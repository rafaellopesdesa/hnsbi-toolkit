from __future__ import annotations

import numpy as np
import pytest

from hnsbi.asimov import AsimovBuilder
from hnsbi.intensity import Component, IntensityModel, Parameter
from hnsbi.likelihood import ExtendedUnbinnedLikelihood
from hnsbi.nis import DefensiveMixture, NISAsimovBuilder, NISProposalTrainer
from hnsbi.systematics import RuntimeSystematic, SystematicRatioEvaluator
from hnsbi.workspace import write_nsbi_workspace


class StandardNormal:
    def sample(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        generator = np.random.default_rng() if rng is None else rng
        return generator.normal(size=(int(n), 1))

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)[:, 0]
        return -0.5 * (x**2 + np.log(2.0 * np.pi))


def _intensity() -> IntensityModel:
    return IntensityModel(
        [Component("signal", 20.0, "mu")],
        [
            Parameter("mu", 1.0, (0.1, 3.0)),
            Parameter(
                "alpha",
                0.0,
                (-1.0, 1.0),
                constrained=True,
                constraint_mean=0.0,
                constraint_sigma=1.0,
            ),
        ],
    )


def _systematic() -> RuntimeSystematic:
    return RuntimeSystematic(
        parameter="alpha",
        component="signal",
        ratio_up=lambda values: np.exp(0.35 * values[:, 0]),
        ratio_down=lambda values: np.exp(-0.2 * values[:, 0]),
        yield_up=1.25,
        yield_down=0.8,
        interpolation="nsbi_code4p",
    )


def _ratios():
    return {"signal": lambda values: np.exp(0.15 * values[:, 0] - 0.5 * 0.15**2)}


def _assert_workspace_closure(result, tmp_path) -> None:
    truth = {"mu": 1.3, "alpha": 0.4}
    expected_yield = 20.0 * truth["mu"] * _systematic().yield_factor(truth["alpha"])
    assert result.events.expected_count == pytest.approx(expected_yield, abs=1.0e-11)
    assert result.auxiliary_observations == {"alpha": truth["alpha"]}

    evaluator = SystematicRatioEvaluator(
        component="signal",
        nominal_process_ratio=result.normalized_ratios["signal"],
        integration_weights=result.reference_weights,
        anchors=result.systematic_anchors["signal"],
    )
    morph = evaluator.evaluate(truth)
    assert np.sum(result.reference_weights * morph.shape_ratio) == pytest.approx(
        1.0, abs=1.0e-12
    )

    export = write_nsbi_workspace(
        result=result,
        intensity=_intensity(),
        output_dir=tmp_path,
        measurement="measurement",
        poi="mu",
    )
    assert export.schema_version == "2.0"
    assert export.workspace["hnsbi"]["auxiliary_observations"] == {
        "alpha": truth["alpha"]
    }
    modifiers = export.workspace["channels"][0]["samples"][0]["modifiers"]
    assert [modifier["type"] for modifier in modifiers] == [
        "normfactor",
        "normplusshape",
    ]

    likelihood = ExtendedUnbinnedLikelihood.from_workspace(export.path)
    assert likelihood.auxiliary_observations == {"alpha": truth["alpha"]}
    step = 1.0e-5
    for parameter in truth:
        above = dict(truth)
        below = dict(truth)
        above[parameter] += step
        below[parameter] -= step
        derivative = (likelihood.nll(above) - likelihood.nll(below)) / (2.0 * step)
        assert derivative == pytest.approx(0.0, abs=3.0e-7)


def test_direct_asimov_closes_nonzero_systematic_workspace(tmp_path) -> None:
    result = AsimovBuilder(
        reference=StandardNormal(),
        ratios=_ratios(),
        intensity=_intensity(),
        features=("x",),
        systematics={"signal": [_systematic()]},
    ).build({"mu": 1.3, "alpha": 0.4}, n_events=4096, seed=19)

    _assert_workspace_closure(result, tmp_path)


def test_nis_asimov_closes_nonzero_systematic_workspace(tmp_path) -> None:
    reference = StandardNormal()
    proposal = DefensiveMixture(
        reference=reference,
        reference_density=reference,
        proposal=reference,
        proposal_density=reference,
        epsilon=0.15,
    )
    result = NISAsimovBuilder(
        proposal=proposal,
        ratios=_ratios(),
        intensity=_intensity(),
        features=("x",),
        systematics={"signal": [_systematic()]},
    ).build({"mu": 1.3, "alpha": 0.4}, n_events=4096, seed=23)

    _assert_workspace_closure(result, tmp_path)


def test_nis_design_includes_systematic_design_points() -> None:
    class StubTrainer:
        def fit(self, values, *, sample_weights):
            assert len(values) == len(sample_weights)
            return StandardNormal()

    result = NISProposalTrainer(
        reference=StandardNormal(),
        ratios=_ratios(),
        intensity=_intensity(),
        trainer=StubTrainer(),
        systematics={"signal": [_systematic()]},
    ).fit(
        truth_point={"mu": 1.0, "alpha": 0.35},
        design_points=[
            {"mu": 0.6, "alpha": -0.3},
            {"mu": 1.8, "alpha": 0.7},
        ],
        pilot_events=512,
        seed=31,
    )

    assert set(result.systematic_anchors) == {"signal"}
    assert np.isfinite(result.pilot_amplitude).all()
    assert np.any(result.pilot_amplitude > 0)
