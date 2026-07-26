"""Validation summaries and plots for neural importance sampling.

The numerical API depends only on NumPy.  Plotting imports Matplotlib lazily,
so production inference and headless validation can use the same reports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .diagnostics import logsumexp, normalize_log_weights, weight_summary
from .onnx import require_optional


def _json_compatible(value: Any) -> Any:
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


def _validated_log_weights(log_weights: Any) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("log_weights must be non-empty.")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("log_weights may be finite or -inf, but not NaN or +inf.")
    if not np.isfinite(values).any():
        raise ValueError("At least one log weight must be finite.")
    return values


def _scalar_logsumexp(values: np.ndarray) -> float:
    return float(np.asarray(logsumexp(values)).reshape(-1)[0])


@dataclass(frozen=True)
class NISLogWeightSummary:
    """A strict, serializable summary of one importance-weight sample."""

    label: str
    count: int
    finite_count: int
    zero_weight_count: int
    finite_fraction: float
    finite_log_weight_mean: float
    finite_log_weight_standard_deviation: float
    finite_log_weight_minimum: float
    finite_log_weight_maximum: float
    finite_log_weight_q01: float
    finite_log_weight_median: float
    finite_log_weight_q99: float
    log_mean_weight: float
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return an RFC-8259-compatible dictionary (non-finite values -> null)."""

        return _json_compatible(asdict(self))


def summarize_nis_log_weights(
    log_weights: Any,
    *,
    label: str = "proposal",
) -> NISLogWeightSummary:
    """Summarize stabilized non-negative importance weights.

    ``-inf`` is retained as an exact zero weight.  NaN and positive infinity
    are rejected because treating them as zeros would hide proposal failures.
    """

    values = _validated_log_weights(log_weights)
    finite = np.isfinite(values)
    finite_values = values[finite]
    quantiles = np.quantile(finite_values, [0.01, 0.5, 0.99])
    metrics = weight_summary(log_weights=values)
    return NISLogWeightSummary(
        label=str(label),
        count=len(values),
        finite_count=int(np.sum(finite)),
        zero_weight_count=int(np.sum(np.isneginf(values))),
        finite_fraction=float(np.mean(finite)),
        finite_log_weight_mean=float(np.mean(finite_values)),
        finite_log_weight_standard_deviation=float(np.std(finite_values)),
        finite_log_weight_minimum=float(np.min(finite_values)),
        finite_log_weight_maximum=float(np.max(finite_values)),
        finite_log_weight_q01=float(quantiles[0]),
        finite_log_weight_median=float(quantiles[1]),
        finite_log_weight_q99=float(quantiles[2]),
        log_mean_weight=_scalar_logsumexp(values) - float(np.log(len(values))),
        metrics={str(key): float(value) for key, value in metrics.items()},
    )


@dataclass(frozen=True)
class NISProposalComparison:
    """Reference-proposal importance-weight comparison."""

    reference: NISLogWeightSummary
    proposal: NISLogWeightSummary
    ess_gain: float
    ess_fraction_gain: float
    max_weight_fraction_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(asdict(self))


def compare_nis_proposal(
    *,
    reference_log_weights: Any,
    proposal_log_weights: Any,
    reference_label: str = "reference",
    proposal_label: str = "proposal",
) -> NISProposalComparison:
    """Compare a learned proposal with direct-reference importance weights."""

    reference = summarize_nis_log_weights(reference_log_weights, label=reference_label)
    proposal = summarize_nis_log_weights(proposal_log_weights, label=proposal_label)
    reference_ess = reference.metrics["ESS"]
    reference_ess_fraction = reference.metrics["ESS_fraction"]
    reference_maximum = reference.metrics["max_weight_fraction"]
    return NISProposalComparison(
        reference=reference,
        proposal=proposal,
        ess_gain=(
            proposal.metrics["ESS"] / reference_ess if reference_ess > 0 else math.nan
        ),
        ess_fraction_gain=(
            proposal.metrics["ESS_fraction"] / reference_ess_fraction
            if reference_ess_fraction > 0
            else math.nan
        ),
        max_weight_fraction_ratio=(
            proposal.metrics["max_weight_fraction"] / reference_maximum
            if reference_maximum > 0
            else math.nan
        ),
    )


@dataclass(frozen=True)
class NISPrefixPoint:
    """One prefix in a deterministic importance-sampling convergence study."""

    prefix_size: int
    log_mean_weight: float
    metrics: dict[str, float]
    observable_estimates: dict[str, float]


@dataclass(frozen=True)
class NISPrefixConvergence:
    """Importance-weight and optional observable estimates over event prefixes."""

    total_count: int
    points: tuple[NISPrefixPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(asdict(self))


def _default_prefix_sizes(count: int) -> tuple[int, ...]:
    start = min(count, 32)
    raw = np.geomspace(start, count, num=min(10, count))
    return tuple(
        int(value)
        for value in np.unique(
            np.concatenate([np.asarray([start]), np.rint(raw), np.asarray([count])])
        )
    )


def nis_prefix_convergence(
    log_weights: Any,
    *,
    prefix_sizes: Sequence[int] | None = None,
    observables: Mapping[str, Any] | None = None,
) -> NISPrefixConvergence:
    """Evaluate ESS, concentration, normalization, and means over prefixes.

    Prefix order is intentionally not shuffled: callers should supply a fixed,
    representative event order (or a reproducibly permuted one).  Observable
    estimates are self-normalized importance-weighted means.
    """

    values = _validated_log_weights(log_weights)
    if prefix_sizes is None:
        sizes = _default_prefix_sizes(len(values))
    else:
        sizes = tuple(sorted({int(size) for size in prefix_sizes}))
        if not sizes or sizes[0] < 1 or sizes[-1] > len(values):
            raise ValueError(
                f"prefix_sizes must be non-empty and lie in [1, {len(values)}]."
            )
    observable_arrays: dict[str, np.ndarray] = {}
    for name, raw in (observables or {}).items():
        array = np.asarray(raw, dtype=np.float64).reshape(-1)
        if len(array) != len(values) or not np.isfinite(array).all():
            raise ValueError(
                f"Observable {name!r} must have one finite value per weight."
            )
        observable_arrays[str(name)] = array
    points: list[NISPrefixPoint] = []
    for size in sizes:
        prefix = values[:size]
        if not np.isfinite(prefix).any():
            raise ValueError(f"The prefix of size {size} contains no positive weight.")
        normalized = normalize_log_weights(prefix)
        estimates = {
            name: float(np.sum(normalized * array[:size]))
            for name, array in observable_arrays.items()
        }
        points.append(
            NISPrefixPoint(
                prefix_size=size,
                log_mean_weight=(_scalar_logsumexp(prefix) - float(np.log(size))),
                metrics={
                    str(key): float(value)
                    for key, value in weight_summary(log_weights=prefix).items()
                },
                observable_estimates=estimates,
            )
        )
    return NISPrefixConvergence(total_count=len(values), points=tuple(points))


@dataclass(frozen=True)
class NISEpsilonPoint:
    """Weight-quality report for one defensive-mixture coefficient."""

    epsilon: float
    summary: NISLogWeightSummary


@dataclass(frozen=True)
class NISEpsilonComparison:
    """Side-by-side defensive-mixture validation."""

    points: tuple[NISEpsilonPoint, ...]
    best_epsilon_by_ess_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(asdict(self))


def compare_nis_epsilons(
    log_weights_by_epsilon: Mapping[float, Any],
) -> NISEpsilonComparison:
    """Compare supplied defensive-mixture log weights using common metrics."""

    if not log_weights_by_epsilon:
        raise ValueError("Provide log weights for at least one epsilon.")
    points: list[NISEpsilonPoint] = []
    seen: set[float] = set()
    for raw_epsilon, values in log_weights_by_epsilon.items():
        epsilon = float(raw_epsilon)
        if not 0 <= epsilon <= 1 or epsilon in seen:
            raise ValueError(
                "Each epsilon must be unique and lie in the closed interval [0, 1]."
            )
        seen.add(epsilon)
        points.append(
            NISEpsilonPoint(
                epsilon=epsilon,
                summary=summarize_nis_log_weights(values, label=f"epsilon={epsilon:g}"),
            )
        )
    points.sort(key=lambda point: point.epsilon)
    best = max(
        points,
        key=lambda point: point.summary.metrics["ESS_fraction"],
    )
    return NISEpsilonComparison(
        points=tuple(points),
        best_epsilon_by_ess_fraction=best.epsilon,
    )


def plot_nis_log_weight_comparison(
    samples: Mapping[str, Any],
    *,
    bins: int = 50,
) -> Any:
    """Plot finite log-weight distributions for reference and proposals."""

    if not samples:
        raise ValueError("Provide at least one named log-weight sample.")
    if bins < 1:
        raise ValueError("bins must be positive.")
    pyplot = require_optional(
        "matplotlib.pyplot",
        extra="plots",
        purpose="NIS log-weight validation",
    )
    figure, axis = pyplot.subplots(figsize=(6.0, 4.0))
    for label, raw in samples.items():
        values = _validated_log_weights(raw)
        axis.hist(
            values[np.isfinite(values)],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=str(label),
        )
    axis.set_xlabel("log importance weight")
    axis.set_ylabel("density")
    axis.legend()
    figure.tight_layout()
    return figure


def plot_nis_prefix_convergence(result: NISPrefixConvergence) -> Any:
    """Plot ESS fraction, largest weight, and mean-weight closure by prefix."""

    pyplot = require_optional(
        "matplotlib.pyplot",
        extra="plots",
        purpose="NIS prefix validation",
    )
    sizes = np.asarray([point.prefix_size for point in result.points])
    ess = np.asarray([point.metrics["ESS_fraction"] for point in result.points])
    maximum = np.asarray(
        [point.metrics["max_weight_fraction"] for point in result.points]
    )
    log_mean = np.asarray([point.log_mean_weight for point in result.points])
    figure, axes = pyplot.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(sizes, ess, marker="o", label="ESS fraction")
    axes[0].plot(sizes, maximum, marker="o", label="max weight fraction")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("raw event count")
    axes[0].set_ylabel("fraction")
    axes[0].legend()
    axes[1].plot(sizes, log_mean, marker="o")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("raw event count")
    axes[1].set_ylabel("log mean weight")
    figure.tight_layout()
    return figure


def plot_nis_epsilon_comparison(result: NISEpsilonComparison) -> Any:
    """Plot ESS and concentration across defensive-mixture coefficients."""

    pyplot = require_optional(
        "matplotlib.pyplot",
        extra="plots",
        purpose="NIS epsilon validation",
    )
    epsilon = np.asarray([point.epsilon for point in result.points])
    ess = np.asarray([point.summary.metrics["ESS_fraction"] for point in result.points])
    maximum = np.asarray(
        [point.summary.metrics["max_weight_fraction"] for point in result.points]
    )
    figure, axis = pyplot.subplots(figsize=(6.0, 4.0))
    axis.plot(epsilon, ess, marker="o", label="ESS fraction")
    axis.plot(epsilon, maximum, marker="o", label="max weight fraction")
    axis.set_xlabel("defensive mixture epsilon")
    axis.set_ylabel("fraction")
    axis.legend()
    figure.tight_layout()
    return figure
