from __future__ import annotations

import json

import numpy as np
import pytest

from hnsbi.nis_diagnostics import (
    compare_nis_feature_closure,
    compare_nis_target,
    plot_nis_feature_closure,
    plot_nis_target_closure,
)


def test_nis_target_closure_recovers_exact_log_density_relation() -> None:
    amplitude = np.exp(np.linspace(-2.0, 2.0, 101))
    log_proposal_over_reference = np.log(amplitude) + 7.5

    result = compare_nis_target(amplitude, log_proposal_over_reference)

    assert result.count == len(amplitude)
    assert np.isclose(np.mean(result.centered_log_amplitude), 0.0, atol=1.0e-15)
    assert np.isclose(
        np.mean(result.centered_log_proposal_over_reference),
        0.0,
        atol=1.0e-15,
    )
    assert np.isclose(result.correlation, 1.0)
    assert np.isclose(result.slope, 1.0)
    assert np.isclose(result.rmse, 0.0, atol=1.0e-15)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    ("amplitude", "log_ratio", "message"),
    [
        ([], [], "non-empty"),
        ([1.0], [0.0], "at least two"),
        ([1.0, 0.0], [0.0, 1.0], "strictly positive"),
        ([1.0, np.nan], [0.0, 1.0], "finite"),
        ([1.0, 2.0], [0.0], "align"),
        ([1.0, 1.0], [0.0, 1.0], "must not be constant"),
        ([1.0, 2.0], [0.0, 0.0], "must not be constant"),
    ],
)
def test_nis_target_closure_rejects_invalid_inputs(
    amplitude: list[float],
    log_ratio: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_nis_target(amplitude, log_ratio)


def test_nis_target_closure_plot_is_lazy_and_subsamples() -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    amplitude = np.exp(np.linspace(-3.0, 3.0, 1_000))
    result = compare_nis_target(amplitude, 0.8 * np.log(amplitude) + 2.0)

    figure = plot_nis_target_closure(
        result,
        gridsize=25,
        max_points=200,
        rng=np.random.default_rng(9),
    )

    assert len(figure.axes) == 2
    assert figure.axes[0].get_xlabel() == r"Centered $\log A(x)$"
    pyplot.close(figure)


def test_nis_feature_closure_matches_identical_samples() -> None:
    reference = np.asarray(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
        ]
    )
    result = compare_nis_feature_closure(
        reference,
        reference.copy(),
        np.zeros(len(reference)),
        features=("x", "y"),
    )

    assert result.reference_count == 4
    assert result.proposal_count == 4
    assert np.isclose(result.proposal_ess, 4.0)
    assert result.weighted_ks == {"x": 0.0, "y": 0.0}
    np.testing.assert_allclose(result.proposal_weights, np.full(4, 0.25))
    json.dumps(result.to_dict(), allow_nan=False)


def test_nis_feature_closure_retains_exact_zero_weight() -> None:
    reference = np.asarray([[0.0], [1.0], [2.0]])
    proposal = np.asarray([[0.0], [1.0], [100.0]])

    result = compare_nis_feature_closure(
        reference,
        proposal,
        [0.0, 0.0, -np.inf],
        features=("x",),
    )

    assert result.proposal_weights[-1] == 0.0
    assert 0.0 <= result.weighted_ks["x"] <= 1.0
    assert np.isclose(result.proposal_ess, 2.0)


@pytest.mark.parametrize(
    ("reference", "proposal", "log_weights", "features", "message"),
    [
        ([0.0, 1.0], [[0.0], [1.0]], [0.0, 0.0], None, "two-dimensional"),
        (
            [[0.0], [np.nan]],
            [[0.0], [1.0]],
            [0.0, 0.0],
            None,
            "finite",
        ),
        (
            [[0.0], [1.0]],
            [[0.0, 1.0], [1.0, 2.0]],
            [0.0, 0.0],
            None,
            "same features",
        ),
        (
            [[0.0], [1.0]],
            [[0.0], [1.0]],
            [0.0],
            None,
            "align",
        ),
        (
            [[0.0], [1.0]],
            [[0.0], [1.0]],
            [0.0, np.inf],
            None,
            r"not NaN or \+inf",
        ),
        (
            [[0.0], [1.0]],
            [[0.0], [1.0]],
            [-np.inf, -np.inf],
            None,
            "At least one",
        ),
        (
            [[0.0], [1.0]],
            [[0.0], [1.0]],
            [0.0, 0.0],
            ("x", "extra"),
            "one unique",
        ),
    ],
)
def test_nis_feature_closure_rejects_invalid_inputs(
    reference: object,
    proposal: object,
    log_weights: object,
    features: tuple[str, ...] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_nis_feature_closure(
            reference,
            proposal,
            log_weights,
            features=features,
        )


def test_nis_feature_closure_plot_uses_common_marginals() -> None:
    pyplot = pytest.importorskip("matplotlib.pyplot")
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(100, 3))
    proposal = rng.normal(size=(120, 3))
    result = compare_nis_feature_closure(
        reference,
        proposal,
        np.zeros(len(proposal)),
        features=("x", "y", "z"),
    )

    figure = plot_nis_feature_closure(result, bins=12, columns=2)

    assert len(figure.axes) == 4
    assert not figure.axes[-1].get_visible()
    assert figure.axes[0].get_ylabel() == "density"
    pyplot.close(figure)


def test_nis_closure_plotters_reject_invalid_layout_options() -> None:
    target = compare_nis_target([1.0, 2.0], [0.0, 1.0])
    feature = compare_nis_feature_closure(
        [[0.0], [1.0]],
        [[0.0], [1.0]],
        [0.0, 0.0],
    )

    with pytest.raises(ValueError, match="gridsize"):
        plot_nis_target_closure(target, gridsize=1)
    with pytest.raises(ValueError, match="max_points"):
        plot_nis_target_closure(target, max_points=0)
    with pytest.raises(ValueError, match="bins and columns"):
        plot_nis_feature_closure(feature, bins=0)
