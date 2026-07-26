from __future__ import annotations

import numpy as np
import pytest

from hnsbi.asimov import AsimovBuilder
from hnsbi.intensity import Component, IntensityModel, Parameter
from hnsbi.likelihood import ExtendedUnbinnedLikelihood, GaussianConstraint
from hnsbi.workspace import write_nsbi_workspace


class NormalSampler:
    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.normal(size=(n, 1))


def test_weighted_asimov_nll_is_stationary_at_truth() -> None:
    model = IntensityModel(
        [Component("signal", 12.0, "mu")],
        [Parameter("mu", 1.0, (0.01, 5.0))],
    )
    result = AsimovBuilder(
        reference=NormalSampler(),
        ratios={"signal": lambda values: np.exp(0.2 * values[:, 0])},
        intensity=model,
        features=["x"],
    ).build({"mu": 1.8}, n_events=2000, seed=2)
    likelihood = ExtendedUnbinnedLikelihood(
        intensity=model,
        ratios=result.normalized_ratios,
        event_weights=result.events.weights,
    )
    step = 1.0e-5
    derivative = (
        likelihood.nll({"mu": 1.8 + step}) - likelihood.nll({"mu": 1.8 - step})
    ) / (2.0 * step)
    assert derivative == pytest.approx(0.0, abs=1.0e-7)
    assert likelihood.nll({"mu": 1.8}) < likelihood.nll({"mu": 1.0})


def test_constraint_and_nonpositive_intensity() -> None:
    model = IntensityModel(
        [Component("sample", 5.0, "mu")],
        [Parameter("mu", 1.0, (-2.0, 3.0), constrained=True)],
    )
    likelihood = ExtendedUnbinnedLikelihood(
        intensity=model,
        ratios={"sample": np.ones(3)},
        constraints={"mu": GaussianConstraint(mean=0.5, sigma=0.25)},
    )
    unconstrained_part = 5.0 - 3.0 * np.log(5.0)
    assert likelihood.nll({"mu": 1.0}) == pytest.approx(unconstrained_part + 2.0)
    assert likelihood.nll({"mu": 0.0}) == float("inf")


def test_formula_workspace_round_trip(tmp_path) -> None:
    model = IntensityModel(
        [Component("signal", 7.0, "mu * exp(alpha)")],
        [
            Parameter("mu", 1.0, (0.01, 5.0)),
            Parameter("alpha", 0.0, (-2.0, 2.0)),
        ],
    )
    result = AsimovBuilder(
        reference=NormalSampler(),
        ratios={"signal": lambda values: np.exp(0.1 * values[:, 0])},
        intensity=model,
        features=["x"],
    ).build({"mu": 1.2, "alpha": 0.1}, n_events=1000, seed=3)
    export = write_nsbi_workspace(
        result=result,
        intensity=model,
        output_dir=tmp_path,
        measurement="measurement",
        poi="mu",
    )
    likelihood = ExtendedUnbinnedLikelihood.from_workspace(export.path)
    point = {"mu": 1.2, "alpha": 0.1}
    expected = ExtendedUnbinnedLikelihood(
        intensity=model,
        ratios=result.normalized_ratios,
        event_weights=result.events.weights,
    )
    assert likelihood.nll(point) == pytest.approx(expected.nll(point))


def test_fit_and_profile_scan_when_scipy_available() -> None:
    pytest.importorskip("scipy")
    model = IntensityModel(
        [Component("signal", 10.0, "mu")],
        [Parameter("mu", 1.0, (0.01, 5.0))],
    )
    likelihood = ExtendedUnbinnedLikelihood(
        intensity=model,
        ratios={"signal": np.ones(17)},
    )
    fit = likelihood.fit()
    assert fit.success
    assert fit.point["mu"] == pytest.approx(1.7, abs=2.0e-5)
    scan = likelihood.profile_scan("mu", [1.0, 1.7, 2.5])
    assert scan.twice_delta_nll[1] == pytest.approx(0.0, abs=1.0e-8)
    assert np.all(scan.twice_delta_nll >= 0.0)
