"""Backend-independent Bayesian inference with a frozen dual model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..diagnostics import effective_sample_size, weight_summary
from ._array import (
    align_rows,
    as_2d,
    as_vector,
    call_log_term,
    logmeanexp,
    normalized_log_weights,
)
from .model import DualModel


@dataclass
class WeightedSamples:
    """A normalized weighted sample with stable raw log weights."""

    values: np.ndarray
    weights: np.ndarray
    log_weights: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = as_2d(self.values, "values")
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        self.log_weights = np.asarray(self.log_weights, dtype=np.float64).reshape(-1)
        if len(self.values) != len(self.weights) or len(self.values) != len(
            self.log_weights
        ):
            raise ValueError(
                "values, weights, and log_weights must have equal row counts."
            )
        if np.any(self.weights < 0) or not np.isfinite(self.weights).all():
            raise ValueError("weights must be finite and non-negative.")
        total = float(np.sum(self.weights, dtype=np.float64))
        if not np.isclose(total, 1.0, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("weights must be normalized to one.")
        if np.isnan(self.log_weights).any() or np.isposinf(self.log_weights).any():
            raise ValueError("log_weights cannot contain NaN or positive infinity.")
        self.metadata = dict(self.metadata)

    @classmethod
    def from_log_weights(
        cls,
        values: np.ndarray,
        log_weights: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> WeightedSamples:
        weights, normalized_log = normalized_log_weights(log_weights)
        return cls(
            values=values,
            weights=weights,
            log_weights=normalized_log,
            metadata={} if metadata is None else metadata,
        )

    @property
    def ess(self) -> float:
        return effective_sample_size(self.weights)

    @property
    def diagnostics(self) -> dict[str, float]:
        return weight_summary(weights=self.weights)

    def resample(
        self,
        n: int | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        n = len(self.values) if n is None else int(n)
        if n < 1:
            raise ValueError("n must be positive.")
        indices = rng.choice(
            len(self.values),
            size=n,
            replace=True,
            p=self.weights,
        )
        return self.values[indices]


def _proposal_log_prob(
    proposal_log_prob: Any | None,
    model: DualModel,
    theta: np.ndarray,
    observation: np.ndarray,
) -> np.ndarray:
    if proposal_log_prob is None:
        return model.posterior_denominator_log_prob(theta, observation)
    if hasattr(proposal_log_prob, "log_prob"):
        try:
            values = proposal_log_prob.log_prob(theta, context=observation)
        except TypeError:
            values = proposal_log_prob.log_prob(theta)
    elif callable(proposal_log_prob):
        try:
            values = proposal_log_prob(theta, observation)
        except TypeError:
            values = proposal_log_prob(theta)
    else:
        values = proposal_log_prob
    return as_vector(values, len(theta), "proposal log probability")


def prior_auxiliary_log_update(
    theta: np.ndarray,
    *,
    analysis_log_prior: Any,
    design_log_prior: Any,
    auxiliary_log_likelihood: Any | None = None,
    baseline_auxiliary_log_likelihood: Any | None = None,
) -> np.ndarray:
    """Return ``log(pi/rho) + log(f/f0)`` for every parameter row."""

    theta = as_2d(theta, "theta")
    log_pi = call_log_term(analysis_log_prior, theta, name="analysis log prior")
    log_rho = call_log_term(design_log_prior, theta, name="design log prior")
    log_f = call_log_term(
        auxiliary_log_likelihood,
        theta,
        name="auxiliary log likelihood",
    )
    log_f0 = call_log_term(
        baseline_auxiliary_log_likelihood,
        theta,
        name="baseline auxiliary log likelihood",
    )
    return log_pi - log_rho + log_f - log_f0


def hnpe_log_weights(
    model: DualModel,
    theta: np.ndarray,
    observation: np.ndarray,
    *,
    proposal_log_prob: Any | None = None,
    analysis_log_prior: Any | None = None,
    auxiliary_log_likelihood: Any | None = None,
    baseline_auxiliary_log_likelihood: Any | None = None,
) -> np.ndarray:
    """Compute hNPE importance log weights for arbitrary proposal points.

    If ``proposal_log_prob`` is omitted, ``theta`` is assumed to have been
    sampled from the exact denominator against which ``r_p`` was trained.
    """

    theta, observation = align_rows(theta, observation, "theta", "observation")
    denominator = model.posterior_denominator_log_prob(theta, observation)
    proposal = _proposal_log_prob(proposal_log_prob, model, theta, observation)
    log_pi = (
        model.log_rho(theta)
        if analysis_log_prior is None
        else call_log_term(analysis_log_prior, theta, name="analysis log prior")
    )
    update = prior_auxiliary_log_update(
        theta,
        analysis_log_prior=log_pi,
        design_log_prior=model.log_rho(theta),
        auxiliary_log_likelihood=auxiliary_log_likelihood,
        baseline_auxiliary_log_likelihood=(baseline_auxiliary_log_likelihood),
    )
    return denominator + model.log_r_p(theta, observation) + update - proposal


def hnde_log_weights(
    model: DualModel,
    theta: np.ndarray,
    observation: np.ndarray,
    *,
    proposal_log_prob: Any | None = None,
    analysis_log_prior: Any | None = None,
    auxiliary_log_likelihood: Any | None = None,
    baseline_auxiliary_log_likelihood: Any | None = None,
) -> np.ndarray:
    """Compute likelihood-side posterior importance log weights."""

    theta, observation = align_rows(theta, observation, "theta", "observation")
    proposal = _proposal_log_prob(proposal_log_prob, model, theta, observation)
    log_pi = (
        model.log_rho(theta)
        if analysis_log_prior is None
        else call_log_term(analysis_log_prior, theta, name="analysis log prior")
    )
    log_f = call_log_term(
        auxiliary_log_likelihood,
        theta,
        name="auxiliary log likelihood",
    )
    log_f0 = call_log_term(
        baseline_auxiliary_log_likelihood,
        theta,
        name="baseline auxiliary log likelihood",
    )
    return log_pi + log_f - log_f0 + model.log_likelihood(observation, theta) - proposal


def update_posterior_weights(
    base_log_weights: np.ndarray,
    theta: np.ndarray,
    *,
    analysis_log_prior: Any,
    design_log_prior: Any,
    auxiliary_log_likelihood: Any | None = None,
    baseline_auxiliary_log_likelihood: Any | None = None,
) -> np.ndarray:
    """Update design-posterior weights without retraining neural models."""

    base = np.asarray(base_log_weights, dtype=np.float64).reshape(-1)
    theta = as_2d(theta, "theta")
    if len(base) != len(theta):
        raise ValueError("base_log_weights must contain one value per theta row.")
    update = prior_auxiliary_log_update(
        theta,
        analysis_log_prior=analysis_log_prior,
        design_log_prior=design_log_prior,
        auxiliary_log_likelihood=auxiliary_log_likelihood,
        baseline_auxiliary_log_likelihood=(baseline_auxiliary_log_likelihood),
    )
    return normalized_log_weights(base + update)[0]


def geometric_consensus(
    posterior_log_weights: np.ndarray,
    likelihood_log_weights: np.ndarray,
) -> np.ndarray:
    """Return the normalized geometric mean of two normalized weight routes."""

    posterior = np.asarray(posterior_log_weights, dtype=np.float64).reshape(-1)
    likelihood = np.asarray(likelihood_log_weights, dtype=np.float64).reshape(-1)
    if len(posterior) != len(likelihood) or not len(posterior):
        raise ValueError(
            "Both consensus routes must contain the same non-zero row count."
        )
    _, log_p = normalized_log_weights(posterior)
    _, log_l = normalized_log_weights(likelihood)
    return normalized_log_weights(0.5 * (log_p + log_l))[0]


PosteriorRoute = Literal["hnpe", "hnde", "dual"]


def sample_posterior(
    model: DualModel,
    observation: np.ndarray,
    *,
    n: int,
    route: PosteriorRoute = "dual",
    analysis_log_prior: Any | None = None,
    auxiliary_log_likelihood: Any | None = None,
    baseline_auxiliary_log_likelihood: Any | None = None,
    rng: np.random.Generator | None = None,
) -> WeightedSamples:
    """Sample the model's denominator and correct it by either dual route."""

    rng = np.random.default_rng() if rng is None else rng
    observation = as_2d(observation, "observation")
    if len(observation) != 1:
        raise ValueError("sample_posterior requires exactly one observation.")
    theta = model.sample_posterior_denominator(int(n), observation=observation, rng=rng)
    log_p = hnpe_log_weights(
        model,
        theta,
        observation,
        analysis_log_prior=analysis_log_prior,
        auxiliary_log_likelihood=auxiliary_log_likelihood,
        baseline_auxiliary_log_likelihood=(baseline_auxiliary_log_likelihood),
    )
    if route == "hnpe":
        selected = log_p
    elif route in {"hnde", "dual"}:
        log_l = hnde_log_weights(
            model,
            theta,
            observation,
            analysis_log_prior=analysis_log_prior,
            auxiliary_log_likelihood=auxiliary_log_likelihood,
            baseline_auxiliary_log_likelihood=(baseline_auxiliary_log_likelihood),
        )
        if route == "hnde":
            selected = log_l
        else:
            _, normalized_p = normalized_log_weights(log_p)
            _, normalized_l = normalized_log_weights(log_l)
            selected = 0.5 * (normalized_p + normalized_l)
    else:
        raise ValueError("route must be 'hnpe', 'hnde', or 'dual'.")
    return WeightedSamples.from_log_weights(
        theta,
        selected,
        metadata={"route": route},
    )


@dataclass(frozen=True)
class EvidenceEstimate:
    """Monte Carlo estimate of an absolute evidence integral."""

    log_evidence: float
    evidence: float
    relative_mc_error: float
    ess: float
    n: int


def estimate_evidence(
    model: DualModel,
    observation: np.ndarray,
    theta: np.ndarray,
    *,
    integration_log_prob: Any,
    analysis_log_prior: Any | None = None,
    auxiliary_log_likelihood: Any | None = None,
    baseline_auxiliary_log_likelihood: Any | None = None,
) -> EvidenceEstimate:
    """Estimate absolute evidence from normalized proposal samples."""

    theta, observation = align_rows(theta, observation, "theta", "observation")
    log_g = call_log_term(
        integration_log_prob, theta, name="integration log probability"
    )
    log_pi = (
        model.log_rho(theta)
        if analysis_log_prior is None
        else call_log_term(analysis_log_prior, theta, name="analysis log prior")
    )
    log_f = call_log_term(
        auxiliary_log_likelihood,
        theta,
        name="auxiliary log likelihood",
    )
    log_f0 = call_log_term(
        baseline_auxiliary_log_likelihood,
        theta,
        name="baseline auxiliary log likelihood",
    )
    log_integrand = (
        log_pi + log_f - log_f0 + model.log_likelihood(observation, theta) - log_g
    )
    log_value = float(logmeanexp(log_integrand))
    scaled = np.exp(log_integrand - np.max(log_integrand))
    mean_scaled = float(np.mean(scaled))
    relative_error = (
        float(np.std(scaled, ddof=1)) / (math.sqrt(len(scaled)) * mean_scaled)
        if len(scaled) > 1 and mean_scaled > 0
        else 0.0
    )
    normalized, _ = normalized_log_weights(log_integrand)
    return EvidenceEstimate(
        log_evidence=log_value,
        evidence=float(np.exp(log_value)),
        relative_mc_error=relative_error,
        ess=effective_sample_size(normalized),
        n=len(theta),
    )


@dataclass
class PredictiveSamples:
    """Simulator-free posterior-predictive candidates and corrections."""

    observation: np.ndarray
    theta: np.ndarray
    weights: np.ndarray
    log_weights: np.ndarray

    def __post_init__(self) -> None:
        self.observation = as_2d(self.observation, "observation")
        self.theta = as_2d(self.theta, "theta")
        if len(self.observation) != len(self.theta):
            raise ValueError("Predictive observation and theta rows must align.")
        weighted = WeightedSamples(self.observation, self.weights, self.log_weights)
        self.weights = weighted.weights
        self.log_weights = weighted.log_weights

    @property
    def diagnostics(self) -> dict[str, float]:
        return weight_summary(weights=self.weights)

    def resample(
        self,
        n: int | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return WeightedSamples(
            self.observation,
            self.weights,
            self.log_weights,
        ).resample(n, rng=rng)


def posterior_predictive(
    model: DualModel,
    posterior: WeightedSamples,
    *,
    n: int,
    rng: np.random.Generator | None = None,
) -> PredictiveSamples:
    """Generate weighted posterior-predictive observations without a simulator."""

    rng = np.random.default_rng() if rng is None else rng
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive.")
    indices = rng.choice(
        len(posterior.values),
        size=n,
        replace=True,
        p=posterior.weights,
    )
    theta = posterior.values[indices]
    draws = model.sample_likelihood_reference(1, theta=theta, rng=rng)
    if draws.ndim == 3:
        draws = draws[:, 0, :]
    log_correction = model.log_conditional_correction(draws, theta)
    weights, normalized_log = normalized_log_weights(log_correction)
    return PredictiveSamples(
        observation=draws,
        theta=theta,
        weights=weights,
        log_weights=normalized_log,
    )


@dataclass(frozen=True)
class SelectionEstimate:
    """Fixed-parameter selection integrals from conditional-flow samples."""

    theta: np.ndarray
    estimate: np.ndarray
    self_normalized_estimate: np.ndarray
    reference_normalization: np.ndarray
    ess: np.ndarray


def selection_integral(
    model: DualModel,
    theta: np.ndarray,
    indicator: Any,
    *,
    n_reference: int,
    rng: np.random.Generator | None = None,
) -> SelectionEstimate:
    """Estimate selection probability at every supplied parameter point."""

    rng = np.random.default_rng() if rng is None else rng
    theta = as_2d(theta, "theta")
    n_reference = int(n_reference)
    if n_reference < 1:
        raise ValueError("n_reference must be positive.")
    draws = model.sample_likelihood_reference(n_reference, theta=theta, rng=rng)
    if draws.ndim == 2:
        draws = draws[None, :, :]
    estimates = []
    self_normalized = []
    normalizations = []
    ess_values = []
    for index, theta_row in enumerate(theta):
        theta_repeated = np.repeat(theta_row.reshape(1, -1), n_reference, axis=0)
        accepted = np.asarray(indicator(draws[index]), dtype=bool).reshape(-1)
        if len(accepted) != n_reference:
            raise ValueError("indicator must return one Boolean per observation row.")
        log_correction = model.log_conditional_correction(draws[index], theta_repeated)
        log_normalization = float(logmeanexp(log_correction))
        normalization = float(np.exp(log_normalization))
        if accepted.any():
            log_selected_mean = float(
                logmeanexp(log_correction[accepted]) + np.log(np.mean(accepted))
            )
            estimate = float(np.exp(log_selected_mean))
        else:
            estimate = 0.0
        weights, _ = normalized_log_weights(log_correction)
        estimates.append(estimate)
        self_normalized.append(float(np.sum(weights * accepted)))
        normalizations.append(normalization)
        ess_values.append(effective_sample_size(weights))
    return SelectionEstimate(
        theta=theta,
        estimate=np.asarray(estimates),
        self_normalized_estimate=np.asarray(self_normalized),
        reference_normalization=np.asarray(normalizations),
        ess=np.asarray(ess_values),
    )


def population_selection(
    fixed_parameter_efficiency: np.ndarray,
    population_weights: np.ndarray,
) -> float:
    """Integrate fixed-parameter efficiencies over a population measure."""

    efficiency = np.asarray(fixed_parameter_efficiency, dtype=np.float64).reshape(-1)
    weights = np.asarray(population_weights, dtype=np.float64).reshape(-1)
    if len(efficiency) != len(weights) or not len(efficiency):
        raise ValueError("Efficiencies and population weights must align.")
    if (
        not np.isfinite(efficiency).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("Population inputs must be finite with non-negative weights.")
    total = float(np.sum(weights))
    if not total > 0:
        raise ValueError("population_weights must have positive sum.")
    return float(np.sum((weights / total) * efficiency))
