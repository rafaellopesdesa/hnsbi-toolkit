from __future__ import annotations

import json

import numpy as np
import pytest

from hnsbi.asimov import AsimovBuilder
from hnsbi.intensity import Component, IntensityModel, Parameter
from hnsbi.likelihood import ExtendedUnbinnedLikelihood
from hnsbi.systematics import RuntimeSystematic, SystematicAnchor
from hnsbi.toys import ToyGenerator
from hnsbi.workspace import load_workspace_model, write_nsbi_workspace


class BalancedSampler:
    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
        values = np.where(np.arange(int(n)) % 2, 1.0, -1.0)
        return values[:, None]


class GridSampler:
    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.linspace(-2.0, 2.0, int(n), dtype=np.float64)[:, None]


def _runtime_systematic() -> RuntimeSystematic:
    return RuntimeSystematic(
        parameter="alpha",
        component="sample",
        ratio_up=lambda values: np.where(values[:, 0] > 0, 9.0, 1.0),
        ratio_down=lambda values: np.where(values[:, 0] < 0, 9.0, 1.0),
        yield_up=1.5,
        yield_down=0.5,
        interpolation="linear",
    )


def test_runtime_systematic_changes_toy_yield_and_shape() -> None:
    intensity = IntensityModel(
        [Component("sample", 1_000.0, "1")],
        [Parameter("alpha", 0.0, (-1.0, 1.0))],
    )
    generator = ToyGenerator(
        intensity=intensity,
        features=("x",),
        component_samplers={"sample": BalancedSampler()},
        systematics={"sample": [_runtime_systematic()]},
    )

    up = generator.generate({"alpha": 1.0}, seed=17, min_pool=10_000)
    down = generator.generate({"alpha": -1.0}, seed=18, min_pool=10_000)

    assert up.component_expectations == {"sample": 1_500.0}
    assert down.component_expectations == {"sample": 500.0}
    assert np.mean(up.events.values[:, 0] > 0) == pytest.approx(0.9, abs=0.03)
    assert np.mean(down.events.values[:, 0] < 0) == pytest.approx(0.9, abs=0.05)


def test_systematic_likelihood_quadrature_is_stationary_at_truth() -> None:
    intensity = IntensityModel(
        [Component("sample", 20.0, "mu")],
        [
            Parameter("mu", 1.0, (0.1, 3.0)),
            Parameter("alpha", 0.0, (-1.0, 1.0)),
        ],
    )
    integration_weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    nominal_ratio = np.asarray([0.5, 0.8, 1.1, 1.7])
    nominal_ratio /= np.sum(integration_weights * nominal_ratio)
    anchor = SystematicAnchor(
        parameter="alpha",
        component="sample",
        ratio_up=np.asarray([0.6, 0.9, 1.3, 1.8]),
        ratio_down=np.asarray([1.7, 1.2, 0.8, 0.5]),
        yield_up=1.25,
        yield_down=0.8,
        interpolation="linear",
    )
    truth = {"mu": 1.3, "alpha": 0.4}
    raw_shape = anchor.raw_shape(truth["alpha"])
    partition = np.sum(integration_weights * nominal_ratio * raw_shape)
    differential = (
        20.0
        * truth["mu"]
        * anchor.yield_factor(truth["alpha"])
        * nominal_ratio
        * raw_shape
        / partition
    )
    likelihood = ExtendedUnbinnedLikelihood(
        intensity=intensity,
        ratios={"sample": nominal_ratio},
        event_weights=integration_weights * differential,
        systematics={"sample": [anchor]},
        integration_weights=integration_weights,
    )

    step = 1.0e-5
    for parameter in truth:
        above = dict(truth)
        below = dict(truth)
        above[parameter] += step
        below[parameter] -= step
        derivative = (likelihood.nll(above) - likelihood.nll(below)) / (2.0 * step)
        assert derivative == pytest.approx(0.0, abs=2.0e-7)


def _systematic_workspace(tmp_path):
    intensity = IntensityModel(
        [Component("signal", 12.0, "mu")],
        [
            Parameter("mu", 1.0, (0.1, 3.0)),
            Parameter("alpha", 0.0, (-1.0, 1.0)),
        ],
    )
    result = AsimovBuilder(
        reference=GridSampler(),
        ratios={"signal": lambda values: np.exp(0.2 * values[:, 0] - 0.5 * 0.2**2)},
        intensity=intensity,
        features=("x",),
    ).build({"mu": 1.0, "alpha": 0.0}, n_events=128)
    x = result.events.values[:, 0]
    anchor = SystematicAnchor(
        parameter="alpha",
        component="signal",
        ratio_up=np.exp(0.15 * x),
        ratio_down=np.exp(-0.1 * x),
        yield_up=1.2,
        yield_down=0.85,
        interpolation="nsbi_code4p",
    )
    modifier = anchor.write_workspace_modifier(tmp_path / "systematics")
    export = write_nsbi_workspace(
        result=result,
        intensity=intensity,
        output_dir=tmp_path / "fit",
        measurement="measurement",
        poi="mu",
        systematic_modifiers={"signal": [modifier]},
    )
    return export, result, intensity, anchor


def test_systematic_workspace_round_trip_preserves_metadata_and_likelihood(
    tmp_path,
) -> None:
    export, result, intensity, anchor = _systematic_workspace(tmp_path)

    recovered = load_workspace_model(export.path)
    specification = recovered.systematics[("signal", "alpha")]
    assert specification.component == "signal"
    assert specification.parameter == "alpha"
    assert specification.interpolation == "nsbi_code4p"
    assert specification.yield_up == pytest.approx(1.2)
    assert specification.yield_down == pytest.approx(0.85)
    assert export.schema_version == "2.0"

    loaded = ExtendedUnbinnedLikelihood.from_workspace(export.path)
    direct = ExtendedUnbinnedLikelihood(
        intensity=intensity,
        ratios=result.normalized_ratios,
        event_weights=result.events.weights,
        systematics={"signal": [anchor]},
        integration_weights=result.reference_weights,
    )
    for alpha in (-0.7, 0.0, 0.8):
        point = {"mu": 1.4, "alpha": alpha}
        assert loaded.nll(point) == pytest.approx(direct.nll(point))


@pytest.mark.parametrize(
    "loader",
    [
        load_workspace_model,
        ExtendedUnbinnedLikelihood.from_workspace,
    ],
    ids=["generative-loader", "likelihood-loader"],
)
def test_systematic_workspace_rejects_manifest_metadata_mismatch(
    tmp_path,
    loader,
) -> None:
    export, _, _, _ = _systematic_workspace(tmp_path)
    workspace = json.loads(export.path.read_text(encoding="utf-8"))
    manifest_value = workspace["channels"][0]["samples"][0]["modifiers"][1]["hnsbi"][
        "manifest"
    ]
    manifest_path = export.path.parent / manifest_value
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["component"] = "background"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata.*component"):
        loader(export.path)
