"""Internal array and callable adapters for the Bayesian subsystem."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

Array = np.ndarray


def as_2d(values: Any, name: str) -> Array:
    """Return a finite two-dimensional floating-point array."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    if len(array) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def as_vector(values: Any, length: int, name: str) -> Array:
    """Return one finite value per requested row."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(length, float(array), dtype=np.float64)
    else:
        array = array.reshape(-1)
    if len(array) != length:
        raise ValueError(f"{name} must contain one value per row.")
    if len(array) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def align_rows(
    first: Any,
    second: Any,
    first_name: str,
    second_name: str,
) -> tuple[Array, Array]:
    """Broadcast a one-row array against another row-wise array."""

    left = as_2d(first, first_name)
    right = as_2d(second, second_name)
    if len(left) == len(right):
        return left, right
    if len(left) == 1:
        return np.repeat(left, len(right), axis=0), right
    if len(right) == 1:
        return left, np.repeat(right, len(left), axis=0)
    raise ValueError(
        f"{first_name} and {second_name} must have equal row counts, "
        "or one must contain exactly one row."
    )


def logsumexp(values: Any, axis: int | None = None) -> Array:
    """A small NumPy-only log-sum-exp implementation."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("logsumexp requires at least one value.")
    maximum = np.max(array, axis=axis, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    result = safe_maximum + np.log(
        np.sum(np.exp(array - safe_maximum), axis=axis, keepdims=True)
    )
    if axis is None:
        return np.asarray(result, dtype=np.float64).reshape(())
    return np.squeeze(result, axis=axis)


def logmeanexp(values: Any, axis: int | None = None) -> Array:
    """Return ``log(mean(exp(values)))`` without avoidable overflow."""

    array = np.asarray(values, dtype=np.float64)
    count = array.size if axis is None else array.shape[axis]
    if count < 1:
        raise ValueError("logmeanexp requires at least one value.")
    return logsumexp(array, axis=axis) - np.log(count)


def normalized_log_weights(log_weights: Any) -> tuple[Array, Array]:
    """Return finite-safe normalized weights and their normalized logarithms."""

    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if not len(values):
        raise ValueError("At least one log weight is required.")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("log_weights cannot contain NaN or positive infinity.")
    if not np.isfinite(values).any():
        raise ValueError("At least one log weight must be finite.")
    safe = np.where(np.isfinite(values), values, -np.inf)
    normalized_log = safe - float(logsumexp(safe))
    return np.exp(normalized_log), normalized_log


def call_log_prob(
    distribution: Any,
    values: Any,
    *,
    context: Any | None = None,
    name: str = "log probability",
) -> Array:
    """Evaluate a density-like object with the subsystem's row convention."""

    array = as_2d(values, "values")
    if not hasattr(distribution, "log_prob"):
        raise TypeError("Distribution objects must provide log_prob().")
    if context is None:
        result = distribution.log_prob(array)
    else:
        context_array, array = align_rows(context, array, "context", "values")
        result = distribution.log_prob(array, context=context_array)
    return as_vector(result, len(array), name)


def call_log_ratio(
    evaluator: Any,
    target: Any,
    context: Any,
    *,
    name: str = "log ratio",
) -> Array:
    """Evaluate a log-ratio object or a two-argument callable."""

    target_array, context_array = align_rows(target, context, "target", "context")
    if hasattr(evaluator, "log_ratio"):
        result = evaluator.log_ratio(target_array, context=context_array)
    elif isinstance(evaluator, Callable):
        result = evaluator(target_array, context_array)
    else:
        raise TypeError("Log-ratio evaluators must be callable or provide log_ratio().")
    return as_vector(result, len(target_array), name)


def call_log_term(
    evaluator: Any | None,
    theta: Any,
    *,
    name: str,
) -> Array:
    """Evaluate an optional prior or auxiliary log-density term."""

    theta_array = as_2d(theta, "theta")
    if evaluator is None:
        return np.zeros(len(theta_array), dtype=np.float64)
    if hasattr(evaluator, "log_prob"):
        result = evaluator.log_prob(theta_array)
    elif isinstance(evaluator, Callable):
        result = evaluator(theta_array)
    else:
        result = evaluator
    return as_vector(result, len(theta_array), name)


def call_conditional_samples(
    distribution: Any,
    n: int,
    *,
    context: Any,
    rng: np.random.Generator,
) -> Array:
    """Return conditional samples with shape ``(n_context, n, n_features)``."""

    n = int(n)
    if n < 1:
        raise ValueError("n must be positive.")
    context_array = as_2d(context, "context")
    if not hasattr(distribution, "sample"):
        raise TypeError("Distribution objects must provide sample().")
    values = np.asarray(
        distribution.sample(n, context=context_array, rng=rng),
        dtype=np.float64,
    )
    n_context = len(context_array)
    if values.ndim == 2:
        if n_context == 1 and len(values) == n:
            values = values[None, :, :]
        elif n == 1 and len(values) == n_context:
            values = values[:, None, :]
        elif len(values) == n_context * n:
            values = values.reshape(n_context, n, -1)
        else:
            raise ValueError(
                "Conditional sample output has an ambiguous two-dimensional shape."
            )
    if values.ndim != 3 or values.shape[:2] != (n_context, n):
        raise ValueError(
            "Conditional samples must have shape (n_context, n, n_features)."
        )
    if not np.isfinite(values).all():
        raise ValueError("Conditional samples contain non-finite values.")
    return values


def call_samples(
    distribution: Any,
    n: int,
    *,
    rng: np.random.Generator,
) -> Array:
    """Return unconditional samples with shape ``(n, n_features)``."""

    n = int(n)
    if n < 1:
        raise ValueError("n must be positive.")
    if not hasattr(distribution, "sample"):
        raise TypeError("Distribution objects must provide sample().")
    values = as_2d(distribution.sample(n, rng=rng), "samples")
    if len(values) != n:
        raise ValueError("sample() returned the wrong number of rows.")
    return values
