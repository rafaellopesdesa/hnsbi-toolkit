"""Reference-flow closure summaries and validation plots.

The numerical diagnostics have no plotting dependency.  Matplotlib is loaded
only by the plotting and report-bundle functions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_artifact_manifest
from .flows import ReferenceFlow
from .onnx import require_optional


def _json_compatible(value: Any) -> Any:
    """Return an RFC-8259-compatible representation of a diagnostic value."""

    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _probability_weights(weights: Any | None, rows: int) -> np.ndarray:
    if weights is None:
        return np.full(rows, 1.0 / rows, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(values) != rows:
        raise ValueError("weights must contain one value per reference event.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative.")
    total = float(np.sum(values))
    if not total > 0:
        raise ValueError("weights must have positive sum.")
    return values / total


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, sorted_values)


def _weighted_correlation(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.full((values.shape[1], values.shape[1]), np.nan, dtype=np.float64)
    weights = weights / np.sum(weights)
    mean = np.sum(values * weights[:, None], axis=0)
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(scale, scale)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > 0,
    )
    diagonal = scale > 0
    correlation[np.diag_indices_from(correlation)] = diagonal.astype(float)
    return correlation


def _weighted_ks_distance(
    reference: np.ndarray,
    generated: np.ndarray,
    reference_weights: np.ndarray,
) -> float:
    if len(reference) == 0 or len(generated) == 0:
        return math.nan
    reference_order = np.argsort(reference, kind="stable")
    generated_order = np.argsort(generated, kind="stable")
    reference_values = reference[reference_order]
    generated_values = generated[generated_order]
    reference_cdf = np.cumsum(reference_weights[reference_order])
    generated_cdf = np.arange(1, len(generated_values) + 1, dtype=np.float64) / len(
        generated_values
    )
    points = np.sort(np.concatenate([reference_values, generated_values]))
    reference_indices = np.searchsorted(reference_values, points, side="right")
    generated_indices = np.searchsorted(generated_values, points, side="right")
    reference_at_points = np.where(
        reference_indices > 0, reference_cdf[reference_indices - 1], 0.0
    )
    generated_at_points = np.where(
        generated_indices > 0, generated_cdf[generated_indices - 1], 0.0
    )
    return float(np.max(np.abs(reference_at_points - generated_at_points)))


@dataclass(frozen=True)
class FiniteTailSummary:
    """Finite-value accounting and stable central/tail quantiles."""

    count: int
    finite_count: int
    finite_fraction: float
    finite_weight_fraction: float
    nan_count: int
    negative_infinity_count: int
    positive_infinity_count: int
    minimum: float
    maximum: float
    q001: float
    q01: float
    median: float
    q99: float
    q999: float


def finite_tail_summary(
    values: Any,
    *,
    weights: Any | None = None,
) -> FiniteTailSummary:
    """Summarize finite failures and the 0.1%, 1%, 99%, and 99.9% tails.

    Weights are renormalized over finite entries.  Non-finite values are still
    counted explicitly and never silently enter a moment or quantile.
    """

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0:
        raise ValueError("values must be non-empty.")
    if weights is None:
        probability = np.full(len(array), 1.0 / len(array), dtype=np.float64)
    else:
        raw_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(raw_weights) != len(array):
            raise ValueError("weights must contain one value per entry.")
        if not np.isfinite(raw_weights).all() or np.any(raw_weights < 0):
            raise ValueError("weights must be finite and non-negative.")
        if not float(np.sum(raw_weights)) > 0:
            raise ValueError("weights must have positive sum.")
        probability = raw_weights / np.sum(raw_weights)
    finite = np.isfinite(array)
    finite_values = array[finite]
    finite_weight_sum = float(np.sum(probability[finite]))
    if len(finite_values) and finite_weight_sum > 0:
        finite_weights = probability[finite]
        finite_weights /= finite_weight_sum
        quantiles = _weighted_quantiles(
            finite_values,
            finite_weights,
            np.asarray([0.001, 0.01, 0.5, 0.99, 0.999]),
        )
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
    else:
        quantiles = np.full(5, np.nan)
        minimum = math.nan
        maximum = math.nan
    return FiniteTailSummary(
        count=len(array),
        finite_count=int(np.sum(finite)),
        finite_fraction=float(np.mean(finite)),
        finite_weight_fraction=float(np.sum(probability[finite])),
        nan_count=int(np.sum(np.isnan(array))),
        negative_infinity_count=int(np.sum(np.isneginf(array))),
        positive_infinity_count=int(np.sum(np.isposinf(array))),
        minimum=minimum,
        maximum=maximum,
        q001=float(quantiles[0]),
        q01=float(quantiles[1]),
        median=float(quantiles[2]),
        q99=float(quantiles[3]),
        q999=float(quantiles[4]),
    )


def _weighted_mean_standard_deviation(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return math.nan, math.nan
    finite_values = values[finite].astype(np.float64)
    finite_weights = weights[finite].astype(np.float64)
    finite_weight_sum = float(np.sum(finite_weights))
    if not finite_weight_sum > 0:
        return math.nan, math.nan
    finite_weights /= finite_weight_sum
    mean = float(np.sum(finite_weights * finite_values))
    standard_deviation = float(
        np.sqrt(np.sum(finite_weights * np.square(finite_values - mean)))
    )
    return mean, standard_deviation


def _fraction_outside(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return math.nan, math.nan
    selected = values[finite]
    return (
        float(np.mean(selected < lower)),
        float(np.mean(selected > upper)),
    )


@dataclass(frozen=True)
class ClassifierTwoSampleTest:
    """Deterministic held-out classifier two-sample-test result."""

    requested: bool
    available: bool
    auc: float
    reason: str | None
    reference_train_count: int
    generated_train_count: int
    reference_test_count: int
    generated_test_count: int
    random_state: int


def classifier_two_sample_test(
    reference_values: Any,
    generated_values: Any,
    *,
    reference_weights: Any | None = None,
    random_state: int = 0,
    test_fraction: float = 0.3,
) -> ClassifierTwoSampleTest:
    """Run a deterministic logistic C2ST, or explain why it was unavailable.

    The returned AUC is measured on a held-out stratified split.  Reference
    weights are honored in both fitting and scoring.  ``scikit-learn`` remains
    optional; if it is absent, ``auc`` is NaN and ``reason`` is populated.
    """

    reference = np.asarray(reference_values, dtype=np.float64)
    generated = np.asarray(generated_values, dtype=np.float64)
    if (
        reference.ndim != 2
        or generated.ndim != 2
        or reference.shape[1] != generated.shape[1]
    ):
        raise ValueError(
            "reference_values and generated_values must be two-dimensional "
            "with a common feature count."
        )
    if len(reference) == 0 or len(generated) == 0 or reference.shape[1] == 0:
        raise ValueError("Both samples must contain events and features.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie strictly between zero and one.")
    probability = _probability_weights(reference_weights, len(reference))
    reference_finite = np.isfinite(reference).all(axis=1) & (probability > 0)
    generated_finite = np.isfinite(generated).all(axis=1)
    reference = reference[reference_finite]
    probability = probability[reference_finite]
    generated = generated[generated_finite]
    if len(reference):
        probability /= np.sum(probability)
    if min(len(reference), len(generated)) < 4:
        return ClassifierTwoSampleTest(
            requested=True,
            available=False,
            auc=math.nan,
            reason="C2ST needs at least four finite events from each sample.",
            reference_train_count=0,
            generated_train_count=0,
            reference_test_count=0,
            generated_test_count=0,
            random_state=int(random_state),
        )
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return ClassifierTwoSampleTest(
            requested=True,
            available=False,
            auc=math.nan,
            reason=(
                "scikit-learn is not installed; install the 'lhc' or 'bayes' "
                "extra to enable C2ST."
            ),
            reference_train_count=0,
            generated_train_count=0,
            reference_test_count=0,
            generated_test_count=0,
            random_state=int(random_state),
        )

    generator = np.random.default_rng(random_state)
    reference_order = generator.permutation(len(reference))
    generated_order = generator.permutation(len(generated))
    reference_test_count = min(
        len(reference) - 2,
        max(1, int(round(test_fraction * len(reference)))),
    )
    generated_test_count = min(
        len(generated) - 2,
        max(1, int(round(test_fraction * len(generated)))),
    )
    reference_test = reference_order[:reference_test_count]
    reference_train = reference_order[reference_test_count:]
    generated_test = generated_order[:generated_test_count]
    generated_train = generated_order[generated_test_count:]
    train_values = np.concatenate(
        [reference[reference_train], generated[generated_train]], axis=0
    )
    train_labels = np.concatenate(
        [
            np.zeros(len(reference_train), dtype=np.int8),
            np.ones(len(generated_train), dtype=np.int8),
        ]
    )
    reference_train_weights = probability[reference_train]
    reference_train_weights /= np.sum(reference_train_weights)
    train_weights = np.concatenate(
        [
            reference_train_weights,
            np.full(len(generated_train), 1.0 / len(generated_train)),
        ]
    )
    mean = np.average(train_values, axis=0, weights=train_weights)
    variance = np.average(np.square(train_values - mean), axis=0, weights=train_weights)
    scale = np.sqrt(variance)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    classifier = LogisticRegression(
        solver="liblinear",
        random_state=int(random_state),
        max_iter=1000,
    )
    try:
        classifier.fit(
            (train_values - mean) / scale,
            train_labels,
            sample_weight=train_weights,
        )
        test_values = np.concatenate(
            [reference[reference_test], generated[generated_test]], axis=0
        )
        test_labels = np.concatenate(
            [
                np.zeros(len(reference_test), dtype=np.int8),
                np.ones(len(generated_test), dtype=np.int8),
            ]
        )
        reference_test_weights = probability[reference_test]
        reference_test_weights /= np.sum(reference_test_weights)
        test_weights = np.concatenate(
            [
                reference_test_weights,
                np.full(len(generated_test), 1.0 / len(generated_test)),
            ]
        )
        scores = classifier.decision_function((test_values - mean) / scale)
        auc = float(roc_auc_score(test_labels, scores, sample_weight=test_weights))
    except Exception as error:
        return ClassifierTwoSampleTest(
            requested=True,
            available=False,
            auc=math.nan,
            reason=f"C2ST failed: {type(error).__name__}: {error}",
            reference_train_count=len(reference_train),
            generated_train_count=len(generated_train),
            reference_test_count=len(reference_test),
            generated_test_count=len(generated_test),
            random_state=int(random_state),
        )
    return ClassifierTwoSampleTest(
        requested=True,
        available=True,
        auc=auc,
        reason=None,
        reference_train_count=len(reference_train),
        generated_train_count=len(generated_train),
        reference_test_count=len(reference_test),
        generated_test_count=len(generated_test),
        random_state=int(random_state),
    )


@dataclass(frozen=True)
class FeatureClosure:
    """One-dimensional reference-versus-flow closure statistics."""

    feature: str
    reference_mean: float
    generated_mean: float
    reference_standard_deviation: float
    generated_standard_deviation: float
    standardized_mean_difference: float
    standard_deviation_ratio: float
    weighted_ks_distance: float
    reference_quantiles: tuple[float, float, float]
    generated_quantiles: tuple[float, float, float]
    reference_distribution: FiniteTailSummary
    generated_distribution: FiniteTailSummary
    generated_below_reference_q01_fraction: float
    generated_above_reference_q99_fraction: float


@dataclass(frozen=True)
class PairwiseClosure:
    """Two-dimensional histogram and linear-correlation closure metrics."""

    first_feature: str
    second_feature: str
    reference_finite_count: int
    generated_finite_count: int
    reference_correlation: float
    generated_correlation: float
    correlation_difference: float
    histogram_total_variation: float
    histogram_jensen_shannon_divergence: float
    histogram_bins_per_axis: int


def _pairwise_closure(
    reference: np.ndarray,
    generated: np.ndarray,
    probability: np.ndarray,
    *,
    features: Sequence[str],
    bins: int,
) -> tuple[PairwiseClosure, ...]:
    if bins < 2:
        raise ValueError("pairwise_bins must be at least two.")
    summaries: list[PairwiseClosure] = []
    for first, second in combinations(range(reference.shape[1]), 2):
        reference_pair = reference[:, [first, second]].astype(np.float64)
        generated_pair = generated[:, [first, second]].astype(np.float64)
        reference_finite = np.isfinite(reference_pair).all(axis=1)
        generated_finite = np.isfinite(generated_pair).all(axis=1)
        reference_selected = reference_pair[reference_finite]
        generated_selected = generated_pair[generated_finite]
        reference_probability = probability[reference_finite]
        reference_probability /= np.sum(reference_probability)
        generated_probability = np.full(
            len(generated_selected),
            1.0 / len(generated_selected) if len(generated_selected) else 0.0,
        )
        if len(generated_selected):
            pooled = np.concatenate([reference_selected, generated_selected], axis=0)
            ranges: list[tuple[float, float]] = []
            for column in range(2):
                lower = float(np.min(pooled[:, column]))
                upper = float(np.max(pooled[:, column]))
                if not upper > lower:
                    padding = max(1.0, abs(lower) * 0.01)
                    lower -= padding
                    upper += padding
                ranges.append((lower, upper))
            reference_histogram, _, _ = np.histogram2d(
                reference_selected[:, 0],
                reference_selected[:, 1],
                bins=bins,
                range=ranges,
                weights=reference_probability,
            )
            generated_histogram, _, _ = np.histogram2d(
                generated_selected[:, 0],
                generated_selected[:, 1],
                bins=bins,
                range=ranges,
                weights=generated_probability,
            )
            reference_histogram /= np.sum(reference_histogram)
            generated_histogram /= np.sum(generated_histogram)
            histogram_total_variation = float(
                0.5
                * np.sum(
                    np.abs(reference_histogram - generated_histogram),
                    dtype=np.float64,
                )
            )
            midpoint = 0.5 * (reference_histogram + generated_histogram)
            reference_nonzero = reference_histogram > 0
            generated_nonzero = generated_histogram > 0
            histogram_jensen_shannon = float(
                0.5
                * np.sum(
                    reference_histogram[reference_nonzero]
                    * np.log(
                        reference_histogram[reference_nonzero]
                        / midpoint[reference_nonzero]
                    )
                )
                + 0.5
                * np.sum(
                    generated_histogram[generated_nonzero]
                    * np.log(
                        generated_histogram[generated_nonzero]
                        / midpoint[generated_nonzero]
                    )
                )
            )
            reference_correlation = float(
                _weighted_correlation(reference_selected, reference_probability)[0, 1]
            )
            generated_correlation = float(
                _weighted_correlation(generated_selected, generated_probability)[0, 1]
            )
        else:
            histogram_total_variation = math.nan
            histogram_jensen_shannon = math.nan
            reference_correlation = float(
                _weighted_correlation(reference_selected, reference_probability)[0, 1]
            )
            generated_correlation = math.nan
        summaries.append(
            PairwiseClosure(
                first_feature=str(features[first]),
                second_feature=str(features[second]),
                reference_finite_count=len(reference_selected),
                generated_finite_count=len(generated_selected),
                reference_correlation=reference_correlation,
                generated_correlation=generated_correlation,
                correlation_difference=(generated_correlation - reference_correlation),
                histogram_total_variation=histogram_total_variation,
                histogram_jensen_shannon_divergence=(histogram_jensen_shannon),
                histogram_bins_per_axis=bins,
            )
        )
    return tuple(summaries)


@dataclass(frozen=True)
class FlowDiagnosticReport:
    """Serializable summary of a reference-flow closure test."""

    reference_count: int
    generated_count: int
    features: tuple[FeatureClosure, ...]
    pairwise: tuple[PairwiseClosure, ...]
    c2st: ClassifierTwoSampleTest
    maximum_correlation_difference: float
    correlation_frobenius_difference: float
    reference_log_prob_mean: float
    generated_log_prob_mean: float
    reference_log_prob_standard_deviation: float
    generated_log_prob_standard_deviation: float
    reference_log_prob_distribution: FiniteTailSummary
    generated_log_prob_distribution: FiniteTailSummary
    generated_log_prob_below_reference_q01_fraction: float
    generated_log_prob_above_reference_q99_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(asdict(self))


@dataclass
class FlowDiagnosticResult:
    """Closure report plus the arrays needed for validation plots."""

    report: FlowDiagnosticReport
    reference_values: np.ndarray
    generated_values: np.ndarray
    reference_weights: np.ndarray
    reference_log_prob: np.ndarray
    generated_log_prob: np.ndarray
    reference_correlation: np.ndarray
    generated_correlation: np.ndarray

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(summary.feature for summary in self.report.features)

    def save(
        self,
        directory: str | Path,
        *,
        prefix: str = "flow_validation",
        bins: int = 50,
    ) -> tuple[Path, Path]:
        """Save JSON and validation figures with a checksummed manifest."""

        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        report_path = output_directory / f"{prefix}.json"
        report_path.write_text(
            json.dumps(
                self.report.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        feature_path = output_directory / f"{prefix}.features.png"
        log_prob_path = output_directory / f"{prefix}.log_prob.png"
        correlation_path = output_directory / f"{prefix}.correlation.png"
        feature_figure = plot_feature_closure(self, bins=bins)
        feature_figure.savefig(feature_path, dpi=160, bbox_inches="tight")
        log_prob_figure = plot_log_prob_closure(self, bins=bins)
        log_prob_figure.savefig(log_prob_path, dpi=160, bbox_inches="tight")
        correlation_figure = plot_correlation_closure(self)
        correlation_figure.savefig(correlation_path, dpi=160, bbox_inches="tight")
        pairwise_figure = None
        pairwise_path = output_directory / f"{prefix}.pairwise.png"
        if self.report.pairwise:
            pairwise_figure = plot_pairwise_closure(self, bins=bins)
            pairwise_figure.savefig(pairwise_path, dpi=160, bbox_inches="tight")
        pyplot = require_optional(
            "matplotlib.pyplot", extra="plots", purpose="closing validation plots"
        )
        for figure in (
            feature_figure,
            log_prob_figure,
            correlation_figure,
            pairwise_figure,
        ):
            if figure is not None:
                pyplot.close(figure)
        manifest_path = output_directory / f"{prefix}.manifest.json"
        files = {
            "correlation-plot": correlation_path,
            "feature-closure-plot": feature_path,
            "log-prob-plot": log_prob_path,
            "summary-json": report_path,
        }
        if pairwise_figure is not None:
            files["pairwise-closure-plot"] = pairwise_path
        write_artifact_manifest(
            manifest_path,
            artifact_type="reference-flow-diagnostics",
            files=files,
            metadata={
                "c2st_auc": (
                    self.report.c2st.auc if self.report.c2st.available else None
                ),
                "features": list(self.features),
                "generated_count": self.report.generated_count,
                "reference_count": self.report.reference_count,
            },
        )
        return report_path, manifest_path


def diagnose_flow(
    flow: ReferenceFlow,
    reference_values: Any,
    *,
    weights: Any | None = None,
    n_generated: int | None = None,
    rng: np.random.Generator | None = None,
    reference_context: Any | None = None,
    generated_context: Any | None = None,
    run_c2st: bool = True,
    c2st_random_state: int = 0,
    c2st_test_fraction: float = 0.3,
    pairwise_bins: int = 20,
) -> FlowDiagnosticResult:
    """Compare a fitted flow with its weighted reference sample.

    All classifier, feature-tail, log-density, and pairwise metrics use the
    same generated closure sample.  C2ST is deterministic for a fixed closure
    sample and ``c2st_random_state``.
    """

    reference = np.asarray(reference_values, dtype=np.float32)
    if reference.ndim != 2 or reference.shape[1] != len(flow.features):
        raise ValueError(f"reference_values must have shape (n, {len(flow.features)}).")
    if len(reference) == 0 or not np.isfinite(reference).all():
        raise ValueError("reference_values must be non-empty and finite.")
    probability = _probability_weights(weights, len(reference))
    count = len(reference) if n_generated is None else int(n_generated)
    if count < 1:
        raise ValueError("n_generated must be positive.")
    if generated_context is None and flow.is_conditional:
        generated_context = reference_context
    generated = flow.sample(count, rng=rng, context=generated_context)
    generated = np.asarray(generated, dtype=np.float32)
    if generated.shape != (count, len(flow.features)):
        raise ValueError(
            "flow.sample() returned an array with shape "
            f"{generated.shape}, expected {(count, len(flow.features))}."
        )
    reference_log_prob = flow.log_prob(reference, context=reference_context)
    generated_finite_rows = np.isfinite(generated).all(axis=1)
    generated_log_prob = np.full(count, np.nan, dtype=np.float64)
    if np.any(generated_finite_rows):
        finite_generated_context = generated_context
        if generated_context is not None:
            context_array = np.asarray(generated_context)
            if context_array.ndim >= 1 and len(context_array) == count:
                finite_generated_context = context_array[generated_finite_rows]
        generated_log_prob[generated_finite_rows] = flow.log_prob(
            generated[generated_finite_rows],
            context=finite_generated_context,
        )
    reference_log_prob = np.asarray(reference_log_prob, dtype=np.float64).reshape(-1)
    generated_log_prob = np.asarray(generated_log_prob, dtype=np.float64)
    if len(reference_log_prob) != len(reference):
        raise ValueError(
            "flow.log_prob(reference_values) must return one value per event."
        )
    if len(generated_log_prob) != len(generated):
        raise ValueError(
            "flow.log_prob(generated_values) must return one value per event."
        )
    generated_weights = np.full(count, 1.0 / count, dtype=np.float64)
    probabilities = np.array([0.16, 0.5, 0.84])
    summaries: list[FeatureClosure] = []
    for index, feature in enumerate(flow.features):
        reference_column = reference[:, index].astype(np.float64)
        generated_column = generated[:, index].astype(np.float64)
        reference_distribution = finite_tail_summary(
            reference_column, weights=probability
        )
        generated_distribution = finite_tail_summary(generated_column)
        reference_mean, reference_std = _weighted_mean_standard_deviation(
            reference_column, probability
        )
        generated_mean, generated_std = _weighted_mean_standard_deviation(
            generated_column, generated_weights
        )
        pooled_scale = math_sqrt_average_variance(reference_std, generated_std)
        generated_finite = np.isfinite(generated_column)
        generated_below, generated_above = _fraction_outside(
            generated_column,
            lower=reference_distribution.q01,
            upper=reference_distribution.q99,
        )
        summaries.append(
            FeatureClosure(
                feature=feature,
                reference_mean=reference_mean,
                generated_mean=generated_mean,
                reference_standard_deviation=reference_std,
                generated_standard_deviation=generated_std,
                standardized_mean_difference=(
                    (generated_mean - reference_mean) / pooled_scale
                    if pooled_scale > 0
                    else 0.0
                ),
                standard_deviation_ratio=(
                    generated_std / reference_std if reference_std > 0 else float("nan")
                ),
                weighted_ks_distance=_weighted_ks_distance(
                    reference_column,
                    generated_column[generated_finite],
                    probability,
                ),
                reference_quantiles=tuple(
                    _weighted_quantiles(
                        reference_column, probability, probabilities
                    ).tolist()
                ),
                generated_quantiles=tuple(
                    _weighted_quantiles(
                        generated_column[generated_finite],
                        (
                            generated_weights[generated_finite]
                            / np.sum(generated_weights[generated_finite])
                        ),
                        probabilities,
                    ).tolist()
                )
                if np.any(generated_finite)
                else (math.nan, math.nan, math.nan),
                reference_distribution=reference_distribution,
                generated_distribution=generated_distribution,
                generated_below_reference_q01_fraction=generated_below,
                generated_above_reference_q99_fraction=generated_above,
            )
        )
    reference_correlation = _weighted_correlation(reference, probability)
    generated_correlation = _weighted_correlation(
        generated[generated_finite_rows],
        (
            generated_weights[generated_finite_rows]
            / np.sum(generated_weights[generated_finite_rows])
        )
        if np.any(generated_finite_rows)
        else generated_weights[generated_finite_rows],
    )
    correlation_difference = generated_correlation - reference_correlation
    finite_correlation_difference = correlation_difference[
        np.isfinite(correlation_difference)
    ]
    reference_log_prob_distribution = finite_tail_summary(
        reference_log_prob, weights=probability
    )
    generated_log_prob_distribution = finite_tail_summary(generated_log_prob)
    (
        generated_log_prob_below,
        generated_log_prob_above,
    ) = _fraction_outside(
        generated_log_prob,
        lower=reference_log_prob_distribution.q01,
        upper=reference_log_prob_distribution.q99,
    )
    reference_log_prob_mean, reference_log_prob_std = _weighted_mean_standard_deviation(
        reference_log_prob, probability
    )
    generated_log_prob_mean, generated_log_prob_std = _weighted_mean_standard_deviation(
        generated_log_prob, generated_weights
    )
    if run_c2st:
        c2st = classifier_two_sample_test(
            reference,
            generated,
            reference_weights=probability,
            random_state=c2st_random_state,
            test_fraction=c2st_test_fraction,
        )
    else:
        c2st = ClassifierTwoSampleTest(
            requested=False,
            available=False,
            auc=math.nan,
            reason="C2ST was disabled by the caller.",
            reference_train_count=0,
            generated_train_count=0,
            reference_test_count=0,
            generated_test_count=0,
            random_state=int(c2st_random_state),
        )
    pairwise = _pairwise_closure(
        reference,
        generated,
        probability,
        features=flow.features,
        bins=pairwise_bins,
    )
    report = FlowDiagnosticReport(
        reference_count=len(reference),
        generated_count=count,
        features=tuple(summaries),
        pairwise=pairwise,
        c2st=c2st,
        maximum_correlation_difference=(
            float(np.max(np.abs(finite_correlation_difference)))
            if len(finite_correlation_difference)
            else math.nan
        ),
        correlation_frobenius_difference=(
            float(np.linalg.norm(finite_correlation_difference))
            if len(finite_correlation_difference)
            else math.nan
        ),
        reference_log_prob_mean=reference_log_prob_mean,
        generated_log_prob_mean=generated_log_prob_mean,
        reference_log_prob_standard_deviation=reference_log_prob_std,
        generated_log_prob_standard_deviation=generated_log_prob_std,
        reference_log_prob_distribution=reference_log_prob_distribution,
        generated_log_prob_distribution=generated_log_prob_distribution,
        generated_log_prob_below_reference_q01_fraction=(generated_log_prob_below),
        generated_log_prob_above_reference_q99_fraction=(generated_log_prob_above),
    )
    return FlowDiagnosticResult(
        report=report,
        reference_values=reference,
        generated_values=generated,
        reference_weights=probability,
        reference_log_prob=reference_log_prob,
        generated_log_prob=generated_log_prob,
        reference_correlation=reference_correlation,
        generated_correlation=generated_correlation,
    )


def math_sqrt_average_variance(first: float, second: float) -> float:
    """Return the RMS scale used for a standardized mean difference."""

    return float(np.sqrt(0.5 * (first**2 + second**2)))


def plot_feature_closure(
    result: FlowDiagnosticResult,
    *,
    bins: int = 50,
    columns: int = 3,
) -> Any:
    """Plot weighted reference and generated one-dimensional marginals."""

    if bins < 1 or columns < 1:
        raise ValueError("bins and columns must be positive.")
    pyplot = require_optional(
        "matplotlib.pyplot", extra="plots", purpose="flow validation plots"
    )
    count = len(result.features)
    rows = int(np.ceil(count / columns))
    figure, axes = pyplot.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.0 * rows),
        squeeze=False,
    )
    for index, (axis, feature) in enumerate(
        zip(axes.flat, result.features, strict=False)
    ):
        reference_finite = np.isfinite(result.reference_values[:, index])
        generated_finite = np.isfinite(result.generated_values[:, index])
        pooled = np.concatenate(
            [
                result.reference_values[reference_finite, index],
                result.generated_values[generated_finite, index],
            ]
        )
        edges = np.histogram_bin_edges(pooled, bins=bins)
        axis.hist(
            result.reference_values[reference_finite, index],
            bins=edges,
            weights=result.reference_weights[reference_finite],
            density=True,
            histtype="step",
            linewidth=1.8,
            label="reference",
        )
        axis.hist(
            result.generated_values[generated_finite, index],
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.8,
            label="flow",
        )
        axis.set_xlabel(feature)
        axis.set_ylabel("density")
        axis.legend()
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    figure.tight_layout()
    return figure


def plot_log_prob_closure(result: FlowDiagnosticResult, *, bins: int = 50) -> Any:
    """Plot the fitted log-density distribution on data and generated events."""

    if bins < 1:
        raise ValueError("bins must be positive.")
    pyplot = require_optional(
        "matplotlib.pyplot", extra="plots", purpose="flow validation plots"
    )
    figure, axis = pyplot.subplots(figsize=(6.0, 4.0))
    reference_finite = np.isfinite(result.reference_log_prob)
    generated_finite = np.isfinite(result.generated_log_prob)
    pooled = np.concatenate(
        [
            result.reference_log_prob[reference_finite],
            result.generated_log_prob[generated_finite],
        ]
    )
    if not len(pooled):
        axis.text(
            0.5,
            0.5,
            "no finite log probabilities",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xlabel("log probability")
        axis.set_ylabel("density")
        figure.tight_layout()
        return figure
    edges = np.histogram_bin_edges(pooled, bins=bins)
    axis.hist(
        result.reference_log_prob[reference_finite],
        bins=edges,
        weights=result.reference_weights[reference_finite],
        density=True,
        histtype="step",
        linewidth=1.8,
        label="reference",
    )
    axis.hist(
        result.generated_log_prob[generated_finite],
        bins=edges,
        density=True,
        histtype="step",
        linewidth=1.8,
        label="flow",
    )
    axis.set_xlabel("log probability")
    axis.set_ylabel("density")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_correlation_closure(result: FlowDiagnosticResult) -> Any:
    """Plot reference, generated, and residual correlation matrices."""

    pyplot = require_optional(
        "matplotlib.pyplot", extra="plots", purpose="flow validation plots"
    )
    residual = result.generated_correlation - result.reference_correlation
    figure, axes = pyplot.subplots(1, 3, figsize=(12.0, 3.7))
    images = (
        axes[0].imshow(result.reference_correlation, vmin=-1, vmax=1),
        axes[1].imshow(result.generated_correlation, vmin=-1, vmax=1),
        axes[2].imshow(residual, vmin=-1, vmax=1),
    )
    for axis, title, image in zip(
        axes,
        ("reference", "flow", "flow - reference"),
        images,
        strict=True,
    ):
        axis.set_title(title)
        axis.set_xticks(range(len(result.features)), result.features, rotation=90)
        axis.set_yticks(range(len(result.features)), result.features)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    return figure


def plot_pairwise_closure(
    result: FlowDiagnosticResult,
    *,
    bins: int = 40,
    pairs: Sequence[tuple[str, str]] | None = None,
    max_pairs: int | None = 6,
) -> Any:
    """Plot reference, generated, and residual densities for feature pairs.

    ``pairs`` uses feature names and therefore remains stable if the backing
    array layout changes.  By default the first six pairs are shown to keep
    high-dimensional reports readable; pass ``max_pairs=None`` for all pairs.
    """

    if bins < 2:
        raise ValueError("bins must be at least two.")
    available = {
        (summary.first_feature, summary.second_feature): summary
        for summary in result.report.pairwise
    }
    if pairs is None:
        selected_pairs = list(available)
    else:
        selected_pairs = [tuple(pair) for pair in pairs]
        unknown = [pair for pair in selected_pairs if pair not in available]
        if unknown:
            raise ValueError(f"Unknown or reversed feature pair(s): {unknown}.")
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive or None.")
        selected_pairs = selected_pairs[:max_pairs]
    if not selected_pairs:
        raise ValueError("At least two features are required for a pairwise plot.")

    pyplot = require_optional(
        "matplotlib.pyplot", extra="plots", purpose="pairwise flow validation"
    )
    feature_index = {feature: index for index, feature in enumerate(result.features)}
    figure, axes = pyplot.subplots(
        len(selected_pairs),
        3,
        figsize=(10.8, 3.25 * len(selected_pairs)),
        squeeze=False,
    )
    for row, (first_feature, second_feature) in enumerate(selected_pairs):
        first = feature_index[first_feature]
        second = feature_index[second_feature]
        reference_pair = result.reference_values[:, [first, second]]
        generated_pair = result.generated_values[:, [first, second]]
        reference_finite = np.isfinite(reference_pair).all(axis=1)
        generated_finite = np.isfinite(generated_pair).all(axis=1)
        reference_pair = reference_pair[reference_finite]
        generated_pair = generated_pair[generated_finite]
        if not len(reference_pair) or not len(generated_pair):
            for axis in axes[row]:
                axis.text(
                    0.5,
                    0.5,
                    "no finite pair",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            continue
        pooled = np.concatenate([reference_pair, generated_pair], axis=0)
        ranges: list[tuple[float, float]] = []
        for column in range(2):
            lower = float(np.min(pooled[:, column]))
            upper = float(np.max(pooled[:, column]))
            if not upper > lower:
                padding = max(1.0, abs(lower) * 0.01)
                lower -= padding
                upper += padding
            ranges.append((lower, upper))
        reference_histogram, x_edges, y_edges = np.histogram2d(
            reference_pair[:, 0],
            reference_pair[:, 1],
            bins=bins,
            range=ranges,
            weights=result.reference_weights[reference_finite],
        )
        generated_histogram, _, _ = np.histogram2d(
            generated_pair[:, 0],
            generated_pair[:, 1],
            bins=[x_edges, y_edges],
        )
        reference_histogram /= np.sum(reference_histogram)
        generated_histogram /= np.sum(generated_histogram)
        limit = max(
            float(np.max(reference_histogram)),
            float(np.max(generated_histogram)),
        )
        residual = generated_histogram - reference_histogram
        residual_limit = max(float(np.max(np.abs(residual))), np.finfo(np.float64).eps)
        images = (
            axes[row, 0].pcolormesh(
                x_edges,
                y_edges,
                reference_histogram.T,
                vmin=0,
                vmax=limit,
                shading="auto",
            ),
            axes[row, 1].pcolormesh(
                x_edges,
                y_edges,
                generated_histogram.T,
                vmin=0,
                vmax=limit,
                shading="auto",
            ),
            axes[row, 2].pcolormesh(
                x_edges,
                y_edges,
                residual.T,
                vmin=-residual_limit,
                vmax=residual_limit,
                cmap="coolwarm",
                shading="auto",
            ),
        )
        summary = available[(first_feature, second_feature)]
        titles = (
            "reference",
            "flow",
            (
                "flow - reference\n"
                f"TV@{summary.histogram_bins_per_axis}="
                f"{summary.histogram_total_variation:.3f}, "
                f"JSD={summary.histogram_jensen_shannon_divergence:.3f}"
            ),
        )
        for axis, title, image in zip(axes[row], titles, images, strict=True):
            axis.set_title(title)
            axis.set_xlabel(first_feature)
            axis.set_ylabel(second_feature)
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    return figure
