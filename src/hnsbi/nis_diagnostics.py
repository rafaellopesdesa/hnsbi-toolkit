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


def _finite_vector(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(array) == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _finite_matrix(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional array.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _weighted_two_sample_ks(
    reference: np.ndarray,
    proposal: np.ndarray,
    proposal_weights: np.ndarray,
) -> float:
    reference_order = np.argsort(reference, kind="stable")
    proposal_order = np.argsort(proposal, kind="stable")
    sorted_reference = reference[reference_order]
    sorted_proposal = proposal[proposal_order]
    reference_cdf = np.arange(1, len(reference) + 1, dtype=np.float64) / len(reference)
    proposal_cdf = np.cumsum(proposal_weights[proposal_order], dtype=np.float64)
    points = np.sort(np.concatenate([sorted_reference, sorted_proposal]))
    reference_indices = np.searchsorted(sorted_reference, points, side="right")
    proposal_indices = np.searchsorted(sorted_proposal, points, side="right")
    reference_at_points = np.where(
        reference_indices > 0,
        reference_cdf[np.maximum(reference_indices - 1, 0)],
        0.0,
    )
    proposal_at_points = np.where(
        proposal_indices > 0,
        proposal_cdf[np.maximum(proposal_indices - 1, 0)],
        0.0,
    )
    return float(np.max(np.abs(reference_at_points - proposal_at_points)))


@dataclass(frozen=True)
class NISTargetClosure:
    """Centered proposal-to-reference log-density closure against its target."""

    count: int
    centered_log_amplitude: np.ndarray
    centered_log_proposal_over_reference: np.ndarray
    correlation: float
    slope: float
    rmse: float

    def to_dict(self) -> dict[str, Any]:
        """Return finite summary metrics without duplicating plotting arrays."""

        return {
            "count": self.count,
            "correlation": self.correlation,
            "slope": self.slope,
            "rmse": self.rmse,
        }


def compare_nis_target(
    influence_amplitude: Any,
    log_proposal_over_reference: Any,
) -> NISTargetClosure:
    """Compare ``log A`` with ``log(g / q)`` after removing both offsets.

    The influence amplitude must be strictly positive because its logarithm
    defines the proposal-training target.  Constant inputs are rejected:
    neither a regression slope nor a correlation is identifiable in that
    case.
    """

    amplitude = _finite_vector(
        influence_amplitude,
        name="influence_amplitude",
    )
    if np.any(amplitude <= 0):
        raise ValueError("influence_amplitude must be strictly positive.")
    log_ratio = _finite_vector(
        log_proposal_over_reference,
        name="log_proposal_over_reference",
    )
    if len(log_ratio) != len(amplitude):
        raise ValueError(
            "influence_amplitude and log_proposal_over_reference must align."
        )
    if len(amplitude) < 2:
        raise ValueError("Target closure requires at least two events.")
    log_amplitude = np.log(amplitude)
    centered_target = log_amplitude - np.mean(log_amplitude)
    centered_proposal = log_ratio - np.mean(log_ratio)
    target_sum_squares = float(np.sum(centered_target**2, dtype=np.float64))
    proposal_sum_squares = float(np.sum(centered_proposal**2, dtype=np.float64))
    if target_sum_squares <= 0:
        raise ValueError("The log influence amplitude must not be constant.")
    if proposal_sum_squares <= 0:
        raise ValueError("log_proposal_over_reference must not be constant.")
    covariance_sum = float(
        np.sum(centered_target * centered_proposal, dtype=np.float64)
    )
    correlation = covariance_sum / math.sqrt(target_sum_squares * proposal_sum_squares)
    slope = covariance_sum / target_sum_squares
    rmse = float(
        np.sqrt(
            np.mean(
                np.square(centered_proposal - centered_target),
                dtype=np.float64,
            )
        )
    )
    return NISTargetClosure(
        count=len(amplitude),
        centered_log_amplitude=centered_target,
        centered_log_proposal_over_reference=centered_proposal,
        correlation=float(np.clip(correlation, -1.0, 1.0)),
        slope=float(slope),
        rmse=rmse,
    )


@dataclass(frozen=True)
class NISFeatureClosure:
    """Direct-reference versus importance-reweighted feature closure."""

    features: tuple[str, ...]
    reference_values: np.ndarray
    proposal_values: np.ndarray
    proposal_weights: np.ndarray
    weighted_ks: dict[str, float]
    proposal_ess: float

    @property
    def reference_count(self) -> int:
        return len(self.reference_values)

    @property
    def proposal_count(self) -> int:
        return len(self.proposal_values)

    def to_dict(self) -> dict[str, Any]:
        """Return finite closure metrics without duplicating plotting arrays."""

        return {
            "features": list(self.features),
            "reference_count": self.reference_count,
            "proposal_count": self.proposal_count,
            "proposal_ess": self.proposal_ess,
            "weighted_ks": dict(self.weighted_ks),
        }


def compare_nis_feature_closure(
    reference_values: Any,
    proposal_values: Any,
    proposal_log_weights: Any,
    *,
    features: Sequence[str] | None = None,
) -> NISFeatureClosure:
    """Compare direct-reference marginals with reweighted proposal marginals.

    ``proposal_log_weights`` are eventwise ``log(q / g)`` values.  They are
    normalized internally, retaining ``-inf`` as an exact zero weight.
    """

    reference = _finite_matrix(reference_values, name="reference_values")
    proposal = _finite_matrix(proposal_values, name="proposal_values")
    if reference.shape[1] != proposal.shape[1]:
        raise ValueError(
            "reference_values and proposal_values must have the same features."
        )
    log_weights = _validated_log_weights(proposal_log_weights)
    if len(log_weights) != len(proposal):
        raise ValueError("proposal_log_weights must align with proposal_values.")
    if features is None:
        parsed_features = tuple(f"x{index}" for index in range(reference.shape[1]))
    else:
        parsed_features = tuple(features)
        if (
            len(parsed_features) != reference.shape[1]
            or any(
                not isinstance(feature, str) or not feature
                for feature in parsed_features
            )
            or len(set(parsed_features)) != len(parsed_features)
        ):
            raise ValueError(
                "features must contain one unique non-empty name per column."
            )
    proposal_weights = normalize_log_weights(log_weights)
    weighted_ks = {
        feature: _weighted_two_sample_ks(
            reference[:, index],
            proposal[:, index],
            proposal_weights,
        )
        for index, feature in enumerate(parsed_features)
    }
    return NISFeatureClosure(
        features=parsed_features,
        reference_values=reference,
        proposal_values=proposal,
        proposal_weights=proposal_weights,
        weighted_ks=weighted_ks,
        proposal_ess=float(1.0 / np.sum(proposal_weights**2, dtype=np.float64)),
    )


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


def plot_nis_target_closure(
    result: NISTargetClosure,
    *,
    gridsize: int = 70,
    max_points: int | None = 40_000,
    rng: np.random.Generator | None = None,
) -> Any:
    """Plot centered ``log A`` against centered ``log(g / q)``."""

    if gridsize < 2:
        raise ValueError("gridsize must be at least two.")
    if max_points is not None and int(max_points) < 1:
        raise ValueError("max_points must be positive or None.")
    target = result.centered_log_amplitude
    proposal = result.centered_log_proposal_over_reference
    if max_points is not None and len(target) > int(max_points):
        generator = np.random.default_rng(0) if rng is None else rng
        selected = generator.choice(len(target), size=int(max_points), replace=False)
        target = target[selected]
        proposal = proposal[selected]
    absolute = np.abs(np.concatenate([target, proposal]))
    limit = float(np.quantile(absolute, 0.995))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Target-closure plotting range is degenerate.")
    pyplot = require_optional(
        "matplotlib.pyplot",
        extra="plots",
        purpose="NIS target closure",
    )
    figure, axis = pyplot.subplots(figsize=(6.4, 5.5))
    counts = axis.hexbin(
        target,
        proposal,
        gridsize=int(gridsize),
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    axis.plot([-limit, limit], [-limit, limit], color="black", ls="--", lw=1.5)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel(r"Centered $\log A(x)$")
    axis.set_ylabel(r"Centered $\log[g(x)/q(x)]$")
    axis.set_title("Neural proposal versus influence target")
    axis.text(
        0.04,
        0.96,
        rf"$\rho={result.correlation:.3f}$"
        + "\n"
        + rf"slope$={result.slope:.3f}$"
        + "\n"
        + rf"RMS$={result.rmse:.3f}$",
        transform=axis.transAxes,
        va="top",
    )
    figure.colorbar(counts, ax=axis, label="log count")
    figure.tight_layout()
    return figure


def plot_nis_feature_closure(
    result: NISFeatureClosure,
    *,
    bins: int = 40,
    columns: int = 3,
) -> Any:
    """Plot direct and importance-reweighted one-dimensional marginals."""

    if bins < 1 or columns < 1:
        raise ValueError("bins and columns must be positive.")
    pyplot = require_optional(
        "matplotlib.pyplot",
        extra="plots",
        purpose="NIS feature closure",
    )
    rows = int(math.ceil(len(result.features) / columns))
    figure, axes = pyplot.subplots(
        rows,
        columns,
        figsize=(4.0 * columns, 3.0 * rows),
        squeeze=False,
    )
    for index, (axis, feature) in enumerate(
        zip(axes.flat, result.features, strict=False)
    ):
        pooled = np.concatenate(
            [
                result.reference_values[:, index],
                result.proposal_values[:, index],
            ]
        )
        edges = np.histogram_bin_edges(pooled, bins=int(bins))
        axis.hist(
            result.reference_values[:, index],
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="#d55e00",
            label="direct reference",
        )
        axis.hist(
            result.proposal_values[:, index],
            bins=edges,
            weights=result.proposal_weights,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="#0072b2",
            label="proposal, reweighted",
        )
        axis.set_xlabel(feature)
        axis.set_ylabel("density")
        axis.set_title(f"weighted KS = {result.weighted_ks[feature]:.3g}")
        axis.legend(fontsize=8)
    for axis in axes.flat[len(result.features) :]:
        axis.set_visible(False)
    figure.suptitle("Importance-reweighted reference closure")
    figure.tight_layout()
    return figure


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
