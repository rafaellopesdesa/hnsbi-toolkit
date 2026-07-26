"""Normalization, bridge, and route diagnostics for dual Bayesian models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..diagnostics import effective_sample_size, weight_summary
from ._array import (
    align_rows,
    as_2d,
    logmeanexp,
    normalized_log_weights,
)
from .inference import hnde_log_weights, hnpe_log_weights
from .model import DualModel


@dataclass(frozen=True)
class NormalizationEstimate:
    """A Monte Carlo normalization estimate and proposal diagnostics."""

    value: float
    log_value: float
    standard_error: float
    ess: float
    n: int


def _normalization_estimate(log_values: np.ndarray) -> NormalizationEstimate:
    values = np.asarray(log_values, dtype=np.float64).reshape(-1)
    if not len(values):
        raise ValueError("At least one log value is required.")
    log_value = float(logmeanexp(values))
    maximum = float(np.max(values))
    scaled = np.exp(values - maximum)
    scale = float(np.exp(maximum))
    estimate = scale * float(np.mean(scaled))
    standard_error = (
        scale * float(np.std(scaled, ddof=1)) / np.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    weights, _ = normalized_log_weights(values)
    return NormalizationEstimate(
        value=estimate,
        log_value=log_value,
        standard_error=standard_error,
        ess=effective_sample_size(weights),
        n=len(values),
    )


def posterior_normalization_diagnostic(
    model: DualModel,
    observation: np.ndarray,
    *,
    n_reference: int,
    rng: np.random.Generator | None = None,
) -> NormalizationEstimate:
    """Estimate ``E_denominator[r_P(theta;x)]`` at one observation."""

    rng = np.random.default_rng() if rng is None else rng
    observation = as_2d(observation, "observation")
    if len(observation) != 1:
        raise ValueError("posterior_normalization_diagnostic requires one observation.")
    theta = model.sample_posterior_denominator(
        int(n_reference), observation=observation, rng=rng
    )
    return _normalization_estimate(model.log_r_p(theta, observation))


@dataclass(frozen=True)
class ConditionalNormalizationDiagnostics:
    """Raw and corrected conditional normalization at each parameter row."""

    theta: np.ndarray
    raw_z: np.ndarray
    modeled_z: np.ndarray
    corrected_z: np.ndarray
    corrected_ess: np.ndarray


def conditional_normalization_diagnostic(
    model: DualModel,
    theta: np.ndarray,
    *,
    n_reference: int,
    rng: np.random.Generator | None = None,
) -> ConditionalNormalizationDiagnostics:
    """Estimate raw and corrected ``Z_C(theta)`` on reference-flow draws."""

    rng = np.random.default_rng() if rng is None else rng
    theta = as_2d(theta, "theta")
    draws = model.sample_likelihood_reference(int(n_reference), theta=theta, rng=rng)
    if draws.ndim == 2:
        draws = draws[None, :, :]
    raw_z = []
    modeled_z = np.exp(model.log_z_c(theta))
    corrected_z = []
    ess = []
    for index, theta_row in enumerate(theta):
        repeated = np.repeat(theta_row.reshape(1, -1), int(n_reference), axis=0)
        raw = model.log_r_c(draws[index], repeated)
        corrected = raw - model.log_z_c(repeated)
        raw_z.append(float(np.exp(logmeanexp(raw))))
        corrected_z.append(float(np.exp(logmeanexp(corrected))))
        weights, _ = normalized_log_weights(corrected)
        ess.append(effective_sample_size(weights))
    return ConditionalNormalizationDiagnostics(
        theta=theta,
        raw_z=np.asarray(raw_z),
        modeled_z=np.asarray(modeled_z),
        corrected_z=np.asarray(corrected_z),
        corrected_ess=np.asarray(ess),
    )


@dataclass(frozen=True)
class BridgeDiagnostics:
    """Pointwise posterior/likelihood residual bridge comparison."""

    direct_log_ratio: np.ndarray
    bridge_log_ratio: np.ndarray
    residual: np.ndarray
    median_absolute_residual: float
    rms_residual: float
    mean_residual: float


def bridge_diagnostic(
    model: DualModel,
    theta: np.ndarray,
    observation: np.ndarray,
    *,
    log_design_evidence: float,
) -> BridgeDiagnostics:
    """Check ``r_P = rho L_C / (m_rho q_denominator)`` pointwise."""

    theta, observation = align_rows(theta, observation, "theta", "observation")
    direct = model.log_r_p(theta, observation)
    bridge = (
        model.log_rho(theta)
        + model.log_likelihood(observation, theta)
        - float(log_design_evidence)
        - model.posterior_denominator_log_prob(theta, observation)
    )
    residual = direct - bridge
    return BridgeDiagnostics(
        direct_log_ratio=direct,
        bridge_log_ratio=bridge,
        residual=residual,
        median_absolute_residual=float(np.median(np.abs(residual))),
        rms_residual=float(np.sqrt(np.mean(residual**2))),
        mean_residual=float(np.mean(residual)),
    )


@dataclass(frozen=True)
class RouteDiagnostics:
    """Agreement and importance-tail summaries for the two posterior routes."""

    log_weight_rms: float
    log_weight_mean_difference: float
    hnpe: dict[str, float]
    hnde: dict[str, float]


def route_diagnostic(
    model: DualModel,
    theta: np.ndarray,
    observation: np.ndarray,
    **weight_options: Any,
) -> RouteDiagnostics:
    """Compare separately normalized hNPE and hNDE weights."""

    log_p = hnpe_log_weights(model, theta, observation, **weight_options)
    log_l = hnde_log_weights(model, theta, observation, **weight_options)
    weights_p, normalized_p = normalized_log_weights(log_p)
    weights_l, normalized_l = normalized_log_weights(log_l)
    difference = normalized_p - normalized_l
    centered = difference - np.mean(difference)
    return RouteDiagnostics(
        log_weight_rms=float(np.sqrt(np.mean(centered**2))),
        log_weight_mean_difference=float(np.mean(difference)),
        hnpe=weight_summary(weights=weights_p),
        hnde=weight_summary(weights=weights_l),
    )
