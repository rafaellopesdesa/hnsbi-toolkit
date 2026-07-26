from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hnsbi.diagnostics import normalize_log_weights
from hnsbi.flow_diagnostics import (
    classifier_two_sample_test,
    diagnose_flow,
    finite_tail_summary,
)
from hnsbi.nis_diagnostics import (
    compare_nis_epsilons,
    compare_nis_proposal,
    nis_prefix_convergence,
    summarize_nis_log_weights,
)


@pytest.mark.parametrize(
    "invalid",
    [
        np.asarray([0.0, np.nan]),
        np.asarray([0.0, np.inf]),
    ],
)
def test_log_weight_normalization_rejects_nonfinite_failures(invalid) -> None:
    with pytest.raises(ValueError, match=r"NaN or \+inf"):
        normalize_log_weights(invalid)


class _GaussianFlow:
    features = ("x", "y", "z")
    is_conditional = False

    def sample(self, n, *, rng=None, context=None):
        assert context is None
        generator = np.random.default_rng() if rng is None else rng
        values = generator.normal(size=(n, 3))
        values[0, 2] = np.inf
        return values.astype(np.float32)

    def log_prob(self, values, *, context=None):
        assert context is None
        values = np.asarray(values, dtype=np.float64)
        result = -0.5 * np.square(values).sum(axis=1)
        result[1] = -np.inf
        return result


def test_flow_report_has_strict_json_tail_pairwise_and_c2st_fields():
    reference = np.random.default_rng(17).normal(size=(300, 3))
    result = diagnose_flow(
        _GaussianFlow(),
        reference,
        n_generated=250,
        rng=np.random.default_rng(21),
        pairwise_bins=12,
    )

    assert len(result.report.pairwise) == 3
    assert result.report.features[2].generated_distribution.finite_count == 249
    assert result.report.generated_log_prob_distribution.negative_infinity_count >= 1
    assert 0 <= result.report.pairwise[0].histogram_total_variation <= 1
    if result.report.c2st.available:
        assert 0 <= result.report.c2st.auc <= 1
        assert result.report.c2st.reason is None
    else:
        assert math.isnan(result.report.c2st.auc)
        assert result.report.c2st.reason
    json.dumps(result.report.to_dict(), allow_nan=False)


def test_finite_tail_summary_accounts_for_each_nonfinite_kind():
    summary = finite_tail_summary(
        [0.0, 1.0, np.nan, np.inf, -np.inf],
        weights=[1.0, 1.0, 1.0, 1.0, 6.0],
    )

    assert summary.count == 5
    assert summary.finite_count == 2
    assert summary.nan_count == 1
    assert summary.positive_infinity_count == 1
    assert summary.negative_infinity_count == 1
    assert np.isclose(summary.finite_weight_fraction, 0.2)
    assert summary.minimum == 0.0
    assert summary.maximum == 1.0


def test_c2st_is_deterministic_or_gracefully_unavailable():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(120, 2))
    generated = rng.normal(loc=0.4, size=(100, 2))
    first = classifier_two_sample_test(reference, generated, random_state=19)
    second = classifier_two_sample_test(reference, generated, random_state=19)

    assert first == second
    if not first.available:
        assert math.isnan(first.auc)
        assert first.reason


def test_nis_log_weight_comparison_prefixes_and_epsilon_are_serializable():
    reference = np.linspace(-5.0, 2.0, 128)
    proposal = np.linspace(-1.0, 1.0, 128)
    proposal[0] = -np.inf
    summary = summarize_nis_log_weights(proposal)
    comparison = compare_nis_proposal(
        reference_log_weights=reference,
        proposal_log_weights=proposal,
    )
    convergence = nis_prefix_convergence(
        proposal,
        prefix_sizes=[16, 64, 128],
        observables={"x": np.linspace(0.0, 1.0, 128)},
    )
    epsilon = compare_nis_epsilons({0.05: reference, 0.2: proposal})

    assert summary.zero_weight_count == 1
    assert [point.prefix_size for point in convergence.points] == [16, 64, 128]
    assert "x" in convergence.points[-1].observable_estimates
    assert comparison.ess_gain > 0
    assert epsilon.best_epsilon_by_ess_fraction in {0.05, 0.2}
    for payload in (
        summary.to_dict(),
        comparison.to_dict(),
        convergence.to_dict(),
        epsilon.to_dict(),
    ):
        json.dumps(payload, allow_nan=False)
