from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from hnsbi.impacts import (
    compute_pulls,
    covariance_impacts,
    global_observable_impacts,
    plot_impacts,
    plot_pulls,
)
from hnsbi.likelihood import FitResult, GaussianConstraint


class GaussianLikelihood:
    """Analytic y ~ N(mu + theta, sy), t ~ N(theta, st) likelihood."""

    def __init__(
        self,
        *,
        y: float = 3.0,
        sy: float = 2.0,
        t: float = 0.4,
        st: float = 0.5,
    ) -> None:
        self.y = y
        self.sy = sy
        self.t = t
        self.st = st
        self.constraints = {"theta": GaussianConstraint(mean=t, sigma=st)}
        self.auxiliary_observations = {"theta": t}
        self.intensity = SimpleNamespace(
            parameters=(
                SimpleNamespace(name="mu"),
                SimpleNamespace(name="theta"),
            )
        )

    def with_auxiliary_observations(
        self,
        observations: dict[str, float],
    ) -> GaussianLikelihood:
        return GaussianLikelihood(
            y=self.y,
            sy=self.sy,
            t=observations["theta"],
            st=self.st,
        )

    def fit(
        self,
        *,
        initial: dict[str, float] | None = None,
        fixed: dict[str, float] | None = None,
        **kwargs: object,
    ) -> FitResult:
        assert not fixed, "Impact refits must leave theta floating."
        point = {"mu": self.y - self.t, "theta": self.t}
        covariance = np.asarray(
            [
                [self.sy**2 + self.st**2, -(self.st**2)],
                [-(self.st**2), self.st**2],
            ]
        )
        return FitResult(
            point=point,
            nll=0.0,
            success=True,
            message="analytic",
            evaluations=1,
            covariance=covariance,
        )


def test_pulls_and_covariance_impacts_match_analytic_gaussian() -> None:
    likelihood = GaussianLikelihood()
    fit = likelihood.fit()
    pulls = compute_pulls(likelihood, fit)
    assert len(pulls.entries) == 1
    pull = pulls.entries[0]
    assert pull.name == "theta"
    assert pull.pull == pytest.approx(0.0)
    assert pull.postfit_over_prefit == pytest.approx(1.0)

    impacts = covariance_impacts(
        likelihood,
        "mu",
        fit=fit,
        groups={"experimental": ["theta"]},
    )
    impact = impacts.entries[0]
    assert impact.up == pytest.approx(-0.5)
    assert impact.down == pytest.approx(0.5)
    assert impact.magnitude == pytest.approx(0.5)
    assert impacts.total == pytest.approx(0.5)
    assert impacts.groups["experimental"] == pytest.approx(0.5)
    assert impacts.statistical == pytest.approx(2.0)
    assert json.loads(impacts.to_json())["method"] == "covariance"


def test_global_observable_impacts_shift_data_and_float_nuisance() -> None:
    likelihood = GaussianLikelihood()
    result = global_observable_impacts(
        likelihood,
        "mu",
        groups={"experimental": ["theta"]},
    )
    impact = result.entries[0]
    assert impact.up == pytest.approx(-0.5)
    assert impact.down == pytest.approx(0.5)
    assert impact.magnitude == pytest.approx(0.5)
    assert impact.up_point == pytest.approx({"mu": 2.1, "theta": 0.9})
    assert impact.down_point == pytest.approx({"mu": 3.1, "theta": -0.1})
    assert result.total == pytest.approx(0.5)
    assert result.groups["experimental"] == pytest.approx(0.5)
    assert result.method == "global_observable"


def test_global_observable_impacts_reject_fixed_parameter_fit() -> None:
    with pytest.raises(ValueError, match="all parameters to float"):
        global_observable_impacts(
            GaussianLikelihood(),
            "mu",
            fit_kwargs={"fixed": {"theta": 1.0}},
        )


def test_correlation_and_errors_can_reconstruct_covariance() -> None:
    likelihood = GaussianLikelihood()
    covariance = likelihood.fit().covariance
    assert covariance is not None
    errors = {
        "mu": float(np.sqrt(covariance[0, 0])),
        "theta": float(np.sqrt(covariance[1, 1])),
    }
    correlation = covariance / np.outer(
        [errors["mu"], errors["theta"]],
        [errors["mu"], errors["theta"]],
    )
    fit = SimpleNamespace(
        point={"mu": 2.6, "theta": 0.4},
        nll=0.0,
        success=True,
        parameter_names=("mu", "theta"),
        covariance=None,
        correlation=correlation,
        errors=errors,
    )
    result = covariance_impacts(likelihood, "mu", fit=fit)
    assert result.entries[0].magnitude == pytest.approx(0.5)
    assert result.statistical == pytest.approx(2.0)


def test_covariance_shape_and_group_validation() -> None:
    likelihood = GaussianLikelihood()
    bad_fit = FitResult(
        point={"mu": 1.0, "theta": 0.0},
        nll=0.0,
        success=True,
        message="",
        evaluations=1,
        covariance=np.eye(1),
    )
    with pytest.raises(ValueError, match="Covariance shape"):
        compute_pulls(likelihood, bad_fit)
    with pytest.raises(ValueError, match="unknown nuisances"):
        covariance_impacts(
            likelihood,
            "mu",
            groups={"bad": ["missing"]},
        )


def test_plot_helpers_return_axes_and_sort_impacts() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    likelihood = GaussianLikelihood()
    pulls = compute_pulls(likelihood)
    impacts = global_observable_impacts(likelihood, "mu")
    pull_figure, pull_axis = plot_pulls(pulls)
    impact_figure, impact_axis = plot_impacts(impacts)
    assert pull_axis.figure is pull_figure
    assert impact_axis.figure is impact_figure
    assert [tick.get_text() for tick in impact_axis.get_yticklabels()] == ["theta"]

    import matplotlib.pyplot as plt

    plt.close(pull_figure)
    plt.close(impact_figure)
