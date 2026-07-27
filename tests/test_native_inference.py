from __future__ import annotations

import numpy as np
import pytest

from hnsbi.inference import JaxLikelihood, MinuitInference
from hnsbi.intensity import Component, IntensityModel, Parameter
from hnsbi.likelihood import ExtendedUnbinnedLikelihood, GaussianConstraint
from hnsbi.multi_workspace import CombinedLikelihood

jax = pytest.importorskip("jax")
pytest.importorskip("iminuit")


def _counting_likelihood() -> ExtendedUnbinnedLikelihood:
    intensity = IntensityModel(
        [
            Component("signal", 10.0, "mu"),
            Component("background", 20.0, "1"),
        ],
        [
            Parameter("mu", nominal=1.0, bounds=(0.0, 5.0)),
            Parameter(
                "alpha",
                nominal=0.0,
                bounds=(-5.0, 5.0),
                constrained=True,
            ),
        ],
    )
    return ExtendedUnbinnedLikelihood(
        intensity=intensity,
        ratios={
            "signal": np.ones(6),
            "background": np.ones(6),
        },
        event_weights=np.full(6, 5.0),
        constraints={"alpha": GaussianConstraint()},
    )


def test_jax_value_gradient_and_hessian_match_counting_model() -> None:
    likelihood = _counting_likelihood()
    backend = JaxLikelihood(likelihood)
    point = {"mu": 1.0, "alpha": 0.0}

    assert backend.nll(point) == pytest.approx(likelihood.nll(point))
    _, gradient = backend.value_and_grad(point)
    assert gradient == pytest.approx([0.0, 0.0], abs=1.0e-10)
    hessian = backend.hessian(point)
    assert hessian[0, 0] == pytest.approx(10.0 / 3.0)
    assert hessian[1, 1] == pytest.approx(1.0)


def test_minuit_uses_nll_errordef_and_returns_named_hesse_covariance() -> None:
    likelihood = _counting_likelihood()
    fit = MinuitInference(likelihood).fit()

    assert fit.success
    assert fit.backend == "jax+iminuit"
    assert fit.parameter_names == ("mu", "alpha")
    assert fit.point == pytest.approx({"mu": 1.0, "alpha": 0.0}, abs=2.0e-4)
    assert fit.covariance is not None
    # For N=30 and lambda=10*mu+20, d2NLL/dmu2=100/30. Correct
    # errordef=0.5 therefore gives variance 0.3, not 0.6.
    assert fit.covariance[0, 0] == pytest.approx(0.3, rel=2.0e-3)
    assert fit.covariance[1, 1] == pytest.approx(1.0, rel=2.0e-3)
    assert fit.correlation is not None


def test_minuit_profile_and_one_sided_test_statistic() -> None:
    inference = MinuitInference(_counting_likelihood())
    profile = inference.profile_scan("mu", [0.5, 1.0, 1.5])
    assert profile.twice_delta_nll[1] == pytest.approx(0.0, abs=1.0e-8)
    assert np.all(profile.twice_delta_nll >= 0)

    scan = inference.test_statistic_scan("mu", [0.5, 1.0, 1.5])
    assert scan.q[0] == pytest.approx(0.0)
    assert scan.q[1] == pytest.approx(0.0, abs=1.0e-8)
    assert scan.q[2] > 0


def test_combined_channels_share_one_constraint_and_jax_fit() -> None:
    first = _counting_likelihood()
    second = _counting_likelihood()
    combined = CombinedLikelihood({"SR": first, "CR": second})
    point = {"mu": 1.0, "alpha": 1.0}

    assert combined.data_nll(point) == pytest.approx(
        first.data_nll(point) + second.data_nll(point)
    )
    # The global auxiliary constraint is counted once, not once per channel.
    assert combined.constraint_nll(point) == pytest.approx(0.5)
    assert JaxLikelihood(combined).nll(point) == pytest.approx(combined.nll(point))
    fit = combined.fit()
    assert fit.success
    assert fit.point == pytest.approx({"mu": 1.0, "alpha": 0.0}, abs=2.0e-4)
