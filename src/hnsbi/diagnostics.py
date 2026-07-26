"""Numerical diagnostics shared by frequentist and Bayesian workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish ESS, valid for ordinary or signed quadrature weights."""

    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not len(values):
        return 0.0
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite.")
    denominator = float(np.sum(values**2, dtype=np.float64))
    if denominator == 0.0:
        return 0.0
    numerator = float(np.sum(values, dtype=np.float64)) ** 2
    return numerator / denominator


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    finite_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    result = finite_maximum + np.log(
        np.sum(np.exp(values - finite_maximum), axis=axis, keepdims=True)
    )
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("log_weights may be finite or -inf, but not NaN or +inf.")
    if not np.isfinite(values).any():
        raise ValueError("At least one log weight must be finite.")
    return np.exp(values - logsumexp(values))


def _pareto_k(log_weights: np.ndarray) -> float:
    try:
        from scipy.stats import genpareto
    except ImportError:
        return math.nan
    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        return math.nan
    weights = np.exp(values - np.max(values))
    tail_length = min(
        max(20, int(3.0 * np.sqrt(len(weights)))),
        max(20, len(weights) // 5),
    )
    tail_length = min(tail_length, len(weights) - 1)
    ordered = np.sort(weights)
    threshold = ordered[-tail_length - 1]
    excess = ordered[-tail_length:] - threshold
    if np.ptp(excess) <= np.finfo(float).eps:
        return 0.0
    try:
        shape, _, _ = genpareto.fit(excess, floc=0.0)
    except Exception:
        return math.nan
    return float(shape)


def weight_summary(
    weights: np.ndarray | None = None,
    *,
    log_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Return ESS, concentration, dynamic-range, and Pareto-tail diagnostics."""

    if (weights is None) == (log_weights is None):
        raise ValueError("Provide exactly one of weights or log_weights.")
    if log_weights is not None:
        raw_log = np.asarray(log_weights, dtype=np.float64).reshape(-1)
        normalized = normalize_log_weights(raw_log)
    else:
        raw = np.asarray(weights, dtype=np.float64).reshape(-1)
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Tail diagnostics require finite non-negative weights.")
        total = float(np.sum(raw))
        if not total > 0:
            raise ValueError("weights must have positive sum.")
        normalized = raw / total
        raw_log = np.where(raw > 0, np.log(raw), -np.inf)
    ess = float(1.0 / np.sum(normalized**2))
    positive = normalized[normalized > 0]
    return {
        "count": float(len(normalized)),
        "ESS": ess,
        "ESS_fraction": ess / len(normalized) if len(normalized) else 0.0,
        "max_weight_fraction": float(np.max(normalized))
        if len(normalized)
        else math.nan,
        "weight_dynamic_range": float(np.max(positive) / np.min(positive))
        if len(positive)
        else math.nan,
        "pareto_k": _pareto_k(raw_log),
    }


def ratio_normalization_report(
    ratios: Mapping[str, np.ndarray],
    weights: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    if not ratios:
        raise ValueError("At least one ratio is required.")
    length = len(next(iter(ratios.values())))
    if length < 1:
        raise ValueError("Ratio arrays must be non-empty.")
    if weights is None:
        probability = np.full(length, 1.0 / length)
    else:
        probability = np.asarray(weights, dtype=np.float64).reshape(-1)
        if (
            len(probability) != length
            or not np.isfinite(probability).all()
            or np.any(probability < 0)
        ):
            raise ValueError("weights must align and be finite and non-negative.")
        total = float(np.sum(probability))
        if not total > 0:
            raise ValueError("weights must have positive sum.")
        probability = probability / total
    result: dict[str, dict[str, float]] = {}
    for name, values in ratios.items():
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(array) != length:
            raise ValueError("Ratio arrays must have a common length.")
        if not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("Ratio arrays must be finite and non-negative.")
        mean = float(np.sum(probability * array))
        variance = float(np.sum(probability * (array - mean) ** 2))
        result[name] = {
            "mean": mean,
            "standard_deviation": math.sqrt(max(0.0, variance)),
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }
    return result


def weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("values must be non-empty and finite.")
    if np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError("quantiles must lie in [0, 1].")
    if weights is None:
        weights = np.ones(len(values), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        len(weights) != len(values)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("weights must align and be finite and non-negative.")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    if cumulative[-1] <= 0:
        raise ValueError("weights must have positive sum.")
    cumulative /= cumulative[-1]
    return np.interp(quantiles, cumulative, sorted_values)


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars/arrays recursively for metadata serialization."""

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
