"""Composition of independent channels with shared physics parameters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .likelihood import (
    ExtendedUnbinnedLikelihood,
    FitResult,
    GaussianConstraint,
    ProfileScanResult,
)


@dataclass
class CombinedLikelihood:
    """Sum channel likelihoods while applying shared constraints exactly once.

    A common nuisance name is therefore correlated across all workspaces even
    when each channel has its own reference density, ratio normalization, FNF,
    and finite quadrature support.
    """

    likelihoods: Mapping[str, ExtendedUnbinnedLikelihood]
    constraints: Mapping[str, GaussianConstraint] | None = None

    def __post_init__(self) -> None:
        channels = dict(self.likelihoods)
        if not channels or any(not name for name in channels):
            raise ValueError("At least one named channel likelihood is required.")
        first = next(iter(channels.values()))
        expected_parameters = tuple(first.intensity.parameters)
        for name, likelihood in channels.items():
            if tuple(likelihood.intensity.parameters) != expected_parameters:
                raise ValueError(
                    f"Channel {name!r} has different shared parameter metadata."
                )
        if self.constraints is None:
            parsed = dict(first.constraints)
            for name, likelihood in channels.items():
                if set(likelihood.constraints) != set(parsed):
                    raise ValueError(
                        f"Channel {name!r} has different constrained parameters."
                    )
                for parameter, constraint in parsed.items():
                    candidate = likelihood.constraints[parameter]
                    if not (
                        np.isclose(candidate.mean, constraint.mean)
                        and np.isclose(candidate.sigma, constraint.sigma)
                    ):
                        raise ValueError(
                            f"Channel {name!r} has a different constraint for "
                            f"{parameter!r}."
                        )
        else:
            parsed = {
                name: (
                    value
                    if isinstance(value, GaussianConstraint)
                    else GaussianConstraint(**value)
                )
                for name, value in self.constraints.items()
            }
        known = {parameter.name for parameter in expected_parameters}
        unknown = set(parsed).difference(known)
        if unknown:
            raise ValueError(
                f"Combined constraints reference unknown parameters {sorted(unknown)}."
            )
        self.likelihoods = channels
        self.constraints = parsed
        self.intensity = first.intensity
        self.auxiliary_observations = {
            name: constraint.mean for name, constraint in parsed.items()
        }

    @property
    def channels(self) -> tuple[ExtendedUnbinnedLikelihood, ...]:
        return tuple(self.likelihoods.values())

    @property
    def observed_count(self) -> float:
        return float(sum(channel.observed_count for channel in self.channels))

    def data_nll(self, point: Mapping[str, float]) -> float:
        validated = self.intensity.validate_point(point)
        return float(sum(channel.data_nll(validated) for channel in self.channels))

    def constraint_nll(self, point: Mapping[str, float]) -> float:
        validated = self.intensity.validate_point(point)
        return float(
            sum(
                constraint.nll(validated[name])
                for name, constraint in self.constraints.items()
            )
        )

    def nll(self, point: Mapping[str, float]) -> float:
        return self.data_nll(point) + self.constraint_nll(point)

    def with_auxiliary_observations(
        self,
        observations: Mapping[str, float],
    ) -> CombinedLikelihood:
        unknown = set(observations).difference(self.constraints)
        if unknown:
            raise ValueError(
                f"Auxiliary observations reference unknown constraints "
                f"{sorted(unknown)}."
            )
        shifted = {
            name: GaussianConstraint(
                mean=float(observations.get(name, constraint.mean)),
                sigma=constraint.sigma,
            )
            for name, constraint in self.constraints.items()
        }
        return CombinedLikelihood(self.likelihoods, constraints=shifted)

    def fit(
        self,
        *,
        initial: Mapping[str, float] | None = None,
        fixed: Mapping[str, float] | None = None,
        use_jax: bool = True,
        **kwargs: Any,
    ) -> FitResult:
        from .inference import MinuitInference

        return MinuitInference(self, use_jax=use_jax).fit(
            initial=initial,
            fixed=fixed,
            **kwargs,
        )

    def profile_scan(
        self,
        parameter: str,
        values: Sequence[float],
        *,
        initial: Mapping[str, float] | None = None,
        use_jax: bool = True,
    ) -> ProfileScanResult:
        from .inference import MinuitInference

        return MinuitInference(self, use_jax=use_jax).profile_scan(
            parameter,
            values,
            initial=initial,
        )
