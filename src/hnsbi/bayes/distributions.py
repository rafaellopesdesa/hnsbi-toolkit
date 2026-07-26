"""Small normalized design distributions used by JSON-driven dual training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IndependentNormal:
    """A diagonal multivariate normal distribution."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float64).reshape(-1)
        if not len(mean) or mean.shape != scale.shape:
            raise ValueError("mean and scale must be non-empty and aligned.")
        if not np.isfinite(mean).all() or np.any(~np.isfinite(scale)):
            raise ValueError("Normal parameters must be finite.")
        if np.any(scale <= 0):
            raise ValueError("Every normal scale must be positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return rng.normal(self.mean, self.scale, size=(int(n), len(self.mean)))

    def log_prob(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.mean):
            raise ValueError(f"values must have shape (n, {len(self.mean)}).")
        standardized = (array - self.mean) / self.scale
        return -0.5 * np.sum(
            standardized**2 + 2.0 * np.log(self.scale) + np.log(2.0 * np.pi),
            axis=1,
        )


@dataclass(frozen=True)
class BoxUniform:
    """A normalized independent uniform distribution on a finite box."""

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=np.float64).reshape(-1)
        high = np.asarray(self.high, dtype=np.float64).reshape(-1)
        if not len(low) or low.shape != high.shape:
            raise ValueError("low and high must be non-empty and aligned.")
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("Uniform bounds must be finite.")
        if np.any(low >= high):
            raise ValueError("Every uniform lower bound must be below its upper.")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return rng.uniform(self.low, self.high, size=(int(n), len(self.low)))

    def log_prob(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.low):
            raise ValueError(f"values must have shape (n, {len(self.low)}).")
        inside = np.all((array >= self.low) & (array <= self.high), axis=1)
        log_density = -float(np.sum(np.log(self.high - self.low)))
        return np.where(inside, log_density, -np.inf)
