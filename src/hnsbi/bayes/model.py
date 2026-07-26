"""The five-artifact dual hNPE--hNDE model contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ._array import (
    align_rows,
    as_2d,
    as_vector,
    call_conditional_samples,
    call_log_prob,
    call_log_ratio,
    call_samples,
)

PosteriorRatioReference = Literal["flow", "defensive"]


@dataclass(frozen=True)
class DualModel:
    """Frozen dual posterior- and likelihood-side artifacts.

    Ratio evaluators return *log* density ratios. ``posterior_ratio_reference``
    records whether ``r_p`` was trained against ``q_phi`` itself or against
    ``(1-epsilon) q_phi + epsilon rho``. This provenance is required for
    correct defensive-mixture accounting.
    """

    q_phi: Any
    r_p: Any
    q_eta: Any
    r_c: Any
    z_c: Any
    rho: Any
    posterior_ratio_reference: PosteriorRatioReference = "flow"
    defensive_epsilon: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reference = str(self.posterior_ratio_reference)
        if reference not in {"flow", "defensive"}:
            raise ValueError("posterior_ratio_reference must be 'flow' or 'defensive'.")
        epsilon = float(self.defensive_epsilon)
        if reference == "defensive":
            if not 0.0 < epsilon < 1.0:
                raise ValueError(
                    "A defensive posterior ratio requires 0 < epsilon < 1."
                )
        elif epsilon != 0.0:
            raise ValueError("defensive_epsilon must be zero when r_p targets q_phi.")
        for name, artifact in self.artifacts.items():
            if artifact is None:
                raise ValueError(f"Required Bayesian artifact {name!r} is missing.")
        if not hasattr(self.rho, "sample") or not hasattr(self.rho, "log_prob"):
            raise TypeError("rho must provide normalized sample() and log_prob().")
        object.__setattr__(self, "posterior_ratio_reference", reference)
        object.__setattr__(self, "defensive_epsilon", epsilon)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def artifacts(self) -> dict[str, Any]:
        """Return the five learned objects under stable scientific names."""

        return {
            "q_phi": self.q_phi,
            "r_p": self.r_p,
            "q_eta": self.q_eta,
            "r_c": self.r_c,
            "z_c": self.z_c,
        }

    def log_rho(self, theta: np.ndarray) -> np.ndarray:
        theta = as_2d(theta, "theta")
        return call_log_prob(self.rho, theta, name="log rho")

    def log_q_phi(self, theta: np.ndarray, observation: np.ndarray) -> np.ndarray:
        theta, observation = align_rows(theta, observation, "theta", "observation")
        return call_log_prob(
            self.q_phi,
            theta,
            context=observation,
            name="log q_phi",
        )

    def log_r_p(self, theta: np.ndarray, observation: np.ndarray) -> np.ndarray:
        theta, observation = align_rows(theta, observation, "theta", "observation")
        return call_log_ratio(
            self.r_p,
            theta,
            observation,
            name="log r_p",
        )

    def log_q_eta(self, observation: np.ndarray, theta: np.ndarray) -> np.ndarray:
        observation, theta = align_rows(observation, theta, "observation", "theta")
        return call_log_prob(
            self.q_eta,
            observation,
            context=theta,
            name="log q_eta",
        )

    def log_r_c(self, observation: np.ndarray, theta: np.ndarray) -> np.ndarray:
        observation, theta = align_rows(observation, theta, "observation", "theta")
        return call_log_ratio(
            self.r_c,
            observation,
            theta,
            name="log r_c",
        )

    def log_z_c(self, theta: np.ndarray) -> np.ndarray:
        theta = as_2d(theta, "theta")
        if hasattr(self.z_c, "log_normalization"):
            values = self.z_c.log_normalization(theta)
        elif callable(self.z_c):
            values = self.z_c(theta)
        else:
            raise TypeError("z_c must be callable or provide log_normalization().")
        return as_vector(values, len(theta), "log Z_C")

    def posterior_denominator_log_prob(
        self,
        theta: np.ndarray,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the exact denominator against which ``r_p`` was trained."""

        theta, observation = align_rows(theta, observation, "theta", "observation")
        log_flow = self.log_q_phi(theta, observation)
        if self.posterior_ratio_reference == "flow":
            return log_flow
        log_design = self.log_rho(theta)
        return np.logaddexp(
            np.log1p(-self.defensive_epsilon) + log_flow,
            np.log(self.defensive_epsilon) + log_design,
        )

    def sample_posterior_denominator(
        self,
        n: int,
        *,
        observation: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw from the exact hNPE ratio denominator.

        A single observation returns ``(n, d_theta)``. Multiple observations
        return ``(n_observations, n, d_theta)``.
        """

        rng = np.random.default_rng() if rng is None else rng
        observation = as_2d(observation, "observation")
        flow = call_conditional_samples(
            self.q_phi,
            int(n),
            context=observation,
            rng=rng,
        )
        if self.posterior_ratio_reference == "defensive":
            design = call_samples(
                self.rho,
                len(observation) * int(n),
                rng=rng,
            ).reshape(len(observation), int(n), -1)
            use_design = rng.random((len(observation), int(n))) < self.defensive_epsilon
            flow = np.where(use_design[:, :, None], design, flow)
        return flow[0] if len(observation) == 1 else flow

    def sample_likelihood_reference(
        self,
        n: int,
        *,
        theta: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw observations from the frozen conditional reference flow."""

        rng = np.random.default_rng() if rng is None else rng
        theta = as_2d(theta, "theta")
        draws = call_conditional_samples(
            self.q_eta,
            int(n),
            context=theta,
            rng=rng,
        )
        return draws[0] if len(theta) == 1 else draws

    def log_likelihood(
        self,
        observation: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """Evaluate ``log q_eta + log r_c - log Z_C``."""

        observation, theta = align_rows(observation, theta, "observation", "theta")
        return (
            self.log_q_eta(observation, theta)
            + self.log_r_c(observation, theta)
            - self.log_z_c(theta)
        )

    def log_conditional_correction(
        self,
        observation: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """Evaluate ``log r_c - log Z_C`` for conditional generation."""

        observation, theta = align_rows(observation, theta, "observation", "theta")
        return self.log_r_c(observation, theta) - self.log_z_c(theta)
