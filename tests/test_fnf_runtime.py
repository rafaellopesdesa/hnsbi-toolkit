from __future__ import annotations

import numpy as np
import pytest

from hnsbi.asimov import AsimovBuilder
from hnsbi.fnf import (
    FactorizableDensity,
    FactorizableResidualStack,
    FNFResidualConfig,
    LogQuadraticYieldMorph,
)
from hnsbi.fnf_runtime import FNFSystematic
from hnsbi.intensity import Component, IntensityModel, Parameter
from hnsbi.likelihood import ExtendedUnbinnedLikelihood
from hnsbi.nis import DefensiveMixture, NISAsimovBuilder
from hnsbi.toys import ToyGenerator
from hnsbi.workspace import write_workspace

pytest.importorskip("torch")


class StandardNormal:
    @staticmethod
    def log_prob(values):
        array = np.asarray(values, dtype=np.float64)
        return -0.5 * np.sum(array**2, axis=1)


class NormalSampler:
    def sample(self, n, *, rng=None):
        return (rng or np.random.default_rng()).normal(size=(int(n), 1))


class NormalDensitySampler(NormalSampler):
    @staticmethod
    def log_prob(values):
        x = np.asarray(values, dtype=np.float64)[:, 0]
        return -0.5 * x**2 - 0.5 * np.log(2.0 * np.pi)


class ExponentialTiltFNF:
    parameters = ("shift",)

    @staticmethod
    def shape_factor(values, point):
        shift = float(point["shift"])
        x = np.asarray(values, dtype=np.float64)[:, 0]
        return np.exp(shift * x - 0.5 * shift**2)

    @staticmethod
    def yield_factor(point):
        return float(1.25 ** float(point["shift"]))


def _intensity() -> IntensityModel:
    return IntensityModel(
        components=[Component("signal", 8.0, "mu")],
        parameters=[
            Parameter("mu", 1.0, bounds=(0.0, 5.0)),
            Parameter(
                "shift",
                0.0,
                bounds=(-3.0, 3.0),
                constrained=True,
            ),
        ],
    )


def _fnf(tmp_path) -> tuple[FNFSystematic, object]:
    residual = FactorizableResidualStack(
        FNFResidualConfig(
            n_features=1,
            nuisance_names=("shift",),
            num_layers=1,
            hidden_features=(8,),
        )
    )
    yield_morph = LogQuadraticYieldMorph.from_anchors(
        8.0,
        {"shift": (7.2, 8.8)},
    )
    artifact = residual.save(
        tmp_path / "fnf",
        prefix="signal",
        metadata={
            "component": "signal",
            "features": ["x"],
            "model_name": "signal_shape",
            "nuisance_names": ["shift"],
            "yield_morph": yield_morph.to_dict(),
        },
    )
    return (
        FNFSystematic(
            FactorizableDensity(residual, StandardNormal()),
            yield_morph,
        ),
        artifact,
    )


def test_fnf_yield_morph_accepts_shape_only_nuisances() -> None:
    residual = FactorizableResidualStack(
        FNFResidualConfig(
            n_features=1,
            nuisance_names=("response", "resolution", "theory"),
            num_layers=1,
            hidden_features=(8,),
        )
    )
    yield_morph = LogQuadraticYieldMorph.from_anchors(
        8.0,
        {"response": (7.2, 8.8)},
    )
    fnf = FNFSystematic(
        FactorizableDensity(residual, StandardNormal()),
        yield_morph,
    )

    assert fnf.yield_factor(
        {"response": 0.0, "resolution": 1.0, "theory": -1.0}
    ) == pytest.approx(1.0)
    assert fnf.yield_factor(
        {"response": 1.0, "resolution": -2.0, "theory": 0.5}
    ) == pytest.approx(1.1)


def test_fnf_likelihood_normalizes_shape_and_applies_yield(tmp_path) -> None:
    fnf, _ = _fnf(tmp_path)
    values = np.linspace(-2.0, 2.0, 101).reshape(-1, 1)
    likelihood = ExtendedUnbinnedLikelihood(
        intensity=_intensity(),
        ratios={"signal": np.ones(len(values))},
        event_weights=np.ones(len(values)),
        integration_weights=np.ones(len(values)),
        event_values=values,
        fnf_systematics={"signal": fnf},
        constraints={"shift": {"mean": 0.0, "sigma": 1.0}},
    )
    nominal = likelihood.nll({"mu": 1.0, "shift": 0.0})
    varied = likelihood.nll({"mu": 1.0, "shift": 1.0})
    assert np.isfinite(nominal)
    assert np.isfinite(varied)
    assert varied != pytest.approx(nominal)
    shifted = likelihood.with_auxiliary_observations({"shift": 0.5})
    assert shifted.fnf_systematics["signal"] is fnf
    assert shifted.event_values is likelihood.event_values


def test_fnf_workspace_requires_portable_nominal_density(tmp_path) -> None:
    _, artifact = _fnf(tmp_path)
    intensity = _intensity()
    asimov = AsimovBuilder(
        reference=NormalSampler(),
        ratios={"signal": lambda values: np.ones(len(values))},
        intensity=intensity,
        features=["x"],
    ).build({"mu": 1.0, "shift": 0.0}, n_events=64, seed=9)
    with pytest.raises(ValueError, match="portable FNF workspace requires"):
        write_workspace(
            result=asimov,
            intensity=intensity,
            output_dir=tmp_path / "workspace",
            measurement="measurement",
            poi="mu",
            fnf_manifests={"signal": artifact.manifest_path},
        )


def test_non_nominal_fnf_asimov_closes_and_matches_likelihood_score() -> None:
    fnf = ExponentialTiltFNF()
    point = {"mu": 1.0, "shift": 1.0}
    asimov = AsimovBuilder(
        reference=NormalSampler(),
        ratios={"signal": lambda values: np.ones(len(values))},
        intensity=_intensity(),
        features=["x"],
        fnf_systematics={"signal": fnf},
    ).build(point, n_events=4096, seed=18)

    assert asimov.fnf_components == ("signal",)
    assert asimov.events.expected_count == pytest.approx(10.0, abs=1.0e-12)
    assert set(asimov.events.metadata["fnf_morphs"]) == {"signal"}
    fnf_metadata = asimov.events.metadata["fnf_morphs"]["signal"]
    assert fnf_metadata["fnf_yield_factor"] == pytest.approx(1.25)
    assert fnf_metadata["fnf_shape_partition"] > 0

    likelihood = ExtendedUnbinnedLikelihood(
        intensity=_intensity(),
        ratios=asimov.normalized_ratios,
        event_weights=asimov.events.weights,
        integration_weights=asimov.reference_weights,
        event_values=asimov.events.values,
        fnf_systematics={"signal": fnf},
        constraints={"shift": {"mean": 1.0, "sigma": 1.0}},
    )
    step = 1.0e-4
    plus = likelihood.nll({"mu": 1.0, "shift": 1.0 + step})
    minus = likelihood.nll({"mu": 1.0, "shift": 1.0 - step})
    assert (plus - minus) / (2.0 * step) == pytest.approx(0.0, abs=2.0e-6)


def test_fnf_toy_applies_shape_and_yield_at_non_nominal_point() -> None:
    intensity = IntensityModel(
        components=[Component("signal", 1500.0, "mu")],
        parameters=[
            Parameter("mu", 1.0, bounds=(0.0, 5.0)),
            Parameter("shift", 0.0, bounds=(-2.0, 2.0)),
        ],
    )
    generator = ToyGenerator(
        intensity=intensity,
        features=["x"],
        component_samplers={"signal": NormalSampler()},
        fnf_systematics={"signal": ExponentialTiltFNF()},
    )
    point = {"mu": 1.0, "shift": 0.6}
    toy = generator.generate(point, seed=92)

    expected = 1500.0 * 1.25**0.6
    assert toy.component_expectations["signal"] == pytest.approx(expected)
    assert np.mean(toy.events.values[:, 0]) == pytest.approx(0.6, abs=0.08)
    diagnostics = toy.events.metadata["component_sampling_diagnostics"]["signal"]
    assert diagnostics["method"] == "component_importance_resampling"
    assert diagnostics["fnf_shape_partition"] > 0
    assert diagnostics["fnf_parameters"] == ["shift"]


def test_non_nominal_fnf_nis_asimov_closes() -> None:
    density = NormalDensitySampler()
    proposal = DefensiveMixture(
        reference=density,
        reference_density=density,
        proposal=density,
        proposal_density=density,
        epsilon=0.2,
    )
    asimov = NISAsimovBuilder(
        proposal=proposal,
        ratios={"signal": lambda values: np.ones(len(values))},
        intensity=_intensity(),
        features=["x"],
        fnf_systematics={"signal": ExponentialTiltFNF()},
    ).build({"mu": 1.0, "shift": 1.0}, n_events=2048, seed=41)
    assert asimov.events.expected_count == pytest.approx(10.0, abs=1.0e-12)
    assert asimov.fnf_components == ("signal",)
    assert set(asimov.events.metadata["fnf_morphs"]) == {"signal"}


def test_workspace_rejects_non_nominal_asimov_that_ignored_fnf(tmp_path) -> None:
    fnf, artifact = _fnf(tmp_path)
    intensity = _intensity()
    point = {"mu": 1.0, "shift": 1.0}
    ignored = AsimovBuilder(
        reference=NormalSampler(),
        ratios={"signal": lambda values: np.ones(len(values))},
        intensity=intensity,
        features=["x"],
    ).build(point, n_events=64, seed=4)
    with pytest.raises(ValueError, match="built without applying that FNF"):
        write_workspace(
            result=ignored,
            intensity=intensity,
            output_dir=tmp_path / "ignored",
            measurement="measurement",
            poi="mu",
            fnf_manifests={"signal": artifact.manifest_path},
        )

    applied = AsimovBuilder(
        reference=NormalSampler(),
        ratios={"signal": lambda values: np.ones(len(values))},
        intensity=intensity,
        features=["x"],
        fnf_systematics={"signal": fnf},
    ).build(point, n_events=64, seed=4)
    with pytest.raises(ValueError, match="portable FNF workspace requires"):
        write_workspace(
            result=applied,
            intensity=intensity,
            output_dir=tmp_path / "applied",
            measurement="measurement",
            poi="mu",
            fnf_manifests={"signal": artifact.manifest_path},
        )
