"""Small structural interfaces used by backend-independent algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

Array = np.ndarray
ParameterPoint = Mapping[str, float]


@runtime_checkable
class Sampler(Protocol):
    """A normalized distribution from which event arrays can be drawn."""

    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> Array:
        """Return an array with shape ``(n, n_features)``."""


@runtime_checkable
class ConditionalSampler(Protocol):
    """A normalized conditional distribution."""

    def sample(
        self,
        n: int,
        *,
        context: Array,
        rng: np.random.Generator | None = None,
    ) -> Array:
        """Return samples for one or more context rows."""


@runtime_checkable
class Density(Protocol):
    """A tractable normalized density."""

    def log_prob(self, values: Array, **kwargs: Any) -> Array:
        """Evaluate one log density per row."""


@runtime_checkable
class RatioEvaluator(Protocol):
    """An event-wise positive density-ratio estimator."""

    def __call__(self, values: Array) -> Array:
        """Return one non-negative ratio per input row."""
