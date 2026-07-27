"""Neural importance proposals and defensive NIS Asimov samples."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .asimov import (
    AsimovResult,
    _draw,
    _evaluate_asimov_intensity,
    _evaluate_ratios,
)
from .data import WeightedEvents
from .diagnostics import (
    effective_sample_size,
    logsumexp,
    normalize_log_weights,
    weight_summary,
)
from .intensity import IntensityModel, RatioNormalizer
from .protocols import Density, RatioEvaluator, Sampler
from .systematics import RuntimeSystematic, SystematicAnchor


def _integrand(
    model: IntensityModel,
    raw_ratios: Mapping[str, np.ndarray],
    normalizers: Mapping[str, float],
    truth_point: Mapping[str, float],
    point: Mapping[str, float],
    *,
    reference_weights: np.ndarray,
    systematics: Mapping[str, Sequence[SystematicAnchor]] | None = None,
    event_values: np.ndarray | None = None,
    fnf_systematics: Mapping[str, Any] | None = None,
) -> np.ndarray:
    normalized = {
        name: np.asarray(values, dtype=np.float64) / normalizers[name]
        for name, values in raw_ratios.items()
    }
    values = (
        np.empty((len(reference_weights), 0), dtype=np.float64)
        if event_values is None
        else np.asarray(event_values)
    )
    h_truth, _, _, _, _ = _evaluate_asimov_intensity(
        intensity=model,
        values=values,
        normalized_ratios=normalized,
        reference_weights=reference_weights,
        point=truth_point,
        systematics=systematics,
        fnf_systematics=fnf_systematics,
    )
    h_point, _, _, _, _ = _evaluate_asimov_intensity(
        intensity=model,
        values=values,
        normalized_ratios=normalized,
        reference_weights=reference_weights,
        point=point,
        systematics=systematics,
        fnf_systematics=fnf_systematics,
    )
    if np.any((h_truth > 0) & (h_point <= 0)):
        raise ValueError(
            "A NIS design point has zero support where the truth intensity is positive."
        )
    return np.where(
        h_truth > 0,
        h_truth * np.log(h_truth / np.maximum(h_point, np.finfo(float).tiny)),
        0.0,
    )


def scan_influence_amplitude(
    model: IntensityModel,
    raw_ratios: Mapping[str, np.ndarray],
    *,
    truth_point: Mapping[str, float],
    design_points: Sequence[Mapping[str, float]],
    reference_weights: np.ndarray | None = None,
    design_weights: np.ndarray | None = None,
    derivative_step: float = 1.0e-4,
    scale: float | None = None,
    systematics: Mapping[str, Sequence[SystematicAnchor]] | None = None,
    event_values: np.ndarray | None = None,
    fnf_systematics: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Influence-function target for a self-normalized likelihood scan.

    The normalizer derivatives are evaluated numerically, which keeps the
    implementation valid for arbitrary safe multiplier formulas and any
    number of process ratios.
    """

    if not design_points:
        raise ValueError("At least one NIS design point is required.")
    arrays = {
        name: np.asarray(values, dtype=np.float64).reshape(-1)
        for name, values in raw_ratios.items()
    }
    length = len(next(iter(arrays.values())))
    if any(len(values) != length for values in arrays.values()):
        raise ValueError("Ratio arrays must be aligned.")
    if event_values is not None:
        event_values = np.asarray(event_values)
        if event_values.ndim != 2 or len(event_values) != length:
            raise ValueError("event_values must align with the ratio arrays.")
    if fnf_systematics and event_values is None:
        raise ValueError("FNF NIS influence evaluation requires event_values.")
    if reference_weights is None:
        probability = np.full(length, 1.0 / length)
    else:
        probability = np.asarray(reference_weights, dtype=np.float64).reshape(-1)
        if len(probability) != length or np.any(probability < 0):
            raise ValueError("reference_weights must align and be non-negative.")
        probability = probability / np.sum(probability)
    if design_weights is None:
        design_probability = np.full(len(design_points), 1.0 / len(design_points))
    else:
        design_probability = np.asarray(design_weights, dtype=np.float64).reshape(-1)
        if len(design_probability) != len(design_points) or np.any(
            design_probability < 0
        ):
            raise ValueError("design_weights must align and be non-negative.")
        design_probability /= np.sum(design_probability)
    means = {
        name: float(np.sum(probability * values)) for name, values in arrays.items()
    }
    if any(value <= 0 for value in means.values()):
        raise ValueError("Every raw ratio must have positive reference mean.")
    if scale is None:
        scale = max(abs(model.expected_yield(truth_point)), 1.0)
    amplitude_squared = np.zeros(length, dtype=np.float64)

    for design_weight, point in zip(design_probability, design_points, strict=True):
        values = _integrand(
            model,
            arrays,
            means,
            truth_point,
            point,
            reference_weights=probability,
            systematics=systematics,
            event_values=event_values,
            fnf_systematics=fnf_systematics,
        )
        integral = float(np.sum(probability * values))
        influence = values - integral
        for name, mean in means.items():
            step = max(abs(mean) * derivative_step, derivative_step)
            plus = dict(means)
            minus = dict(means)
            plus[name] = mean + step
            minus[name] = max(np.finfo(float).tiny, mean - step)
            integral_plus = float(
                np.sum(
                    probability
                    * _integrand(
                        model,
                        arrays,
                        plus,
                        truth_point,
                        point,
                        reference_weights=probability,
                        systematics=systematics,
                        event_values=event_values,
                        fnf_systematics=fnf_systematics,
                    )
                )
            )
            integral_minus = float(
                np.sum(
                    probability
                    * _integrand(
                        model,
                        arrays,
                        minus,
                        truth_point,
                        point,
                        reference_weights=probability,
                        systematics=systematics,
                        event_values=event_values,
                        fnf_systematics=fnf_systematics,
                    )
                )
            )
            derivative = (integral_plus - integral_minus) / (plus[name] - minus[name])
            influence = influence + derivative * (arrays[name] - mean)
        amplitude_squared += float(design_weight) * (influence / float(scale)) ** 2
    return np.sqrt(np.maximum(amplitude_squared, 0.0))


class DefensiveMixture:
    """The exact proposal ``(1-epsilon) g + epsilon q``."""

    def __init__(
        self,
        *,
        reference: Sampler,
        reference_density: Density,
        proposal: Sampler,
        proposal_density: Density,
        epsilon: float,
    ) -> None:
        if not 0.0 < float(epsilon) <= 1.0:
            raise ValueError("epsilon must lie in (0, 1].")
        self.reference = reference
        self.reference_density = reference_density
        self.proposal = proposal
        self.proposal_density = proposal_density
        self.epsilon = float(epsilon)

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        log_q = np.asarray(
            self.reference_density.log_prob(values), dtype=np.float64
        ).reshape(-1)
        log_g = np.asarray(
            self.proposal_density.log_prob(values), dtype=np.float64
        ).reshape(-1)
        if len(log_q) != len(values) or len(log_g) != len(values):
            raise ValueError("Density returned the wrong row count.")
        stacked = np.stack(
            [
                math.log(self.epsilon) + log_q,
                math.log1p(-self.epsilon) + log_g
                if self.epsilon < 1.0
                else np.full_like(log_g, -np.inf),
            ]
        )
        return logsumexp(stacked, axis=0)

    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        n = int(n)
        n_reference = int(rng.binomial(n, self.epsilon))
        n_proposal = n - n_reference
        reference = _draw(self.reference, n_reference, rng) if n_reference else None
        proposal = _draw(self.proposal, n_proposal, rng) if n_proposal else None
        if reference is None:
            values = proposal
        elif proposal is None:
            values = reference
        else:
            if reference.shape[1] != proposal.shape[1]:
                raise ValueError("Mixture samplers have different dimensions.")
            values = np.concatenate([reference, proposal])
        assert values is not None
        return values[rng.permutation(len(values))]

    def sample_with_reference_log_weight(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = rng or np.random.default_rng()
        values = self.sample(n, rng=rng)
        log_q = np.asarray(
            self.reference_density.log_prob(values), dtype=np.float64
        ).reshape(-1)
        log_weight = log_q - self.log_prob(values)
        if np.any(log_weight > -math.log(self.epsilon) + 1.0e-8):
            raise FloatingPointError("Defensive q/g bound was violated.")
        return values, log_weight


@dataclass
class NISDesignResult:
    """Pilot target, trained proposal, and proposal-target diagnostics."""

    proposal: Any
    pilot_values: np.ndarray
    pilot_amplitude: np.ndarray
    training_weights: np.ndarray
    normalizer: RatioNormalizer
    systematic_anchors: dict[str, tuple[SystematicAnchor, ...]]
    fnf_components: tuple[str, ...]
    diagnostics: dict[str, float]


class NISProposalTrainer:
    """Train a weighted flow on the scan-wide influence amplitude."""

    def __init__(
        self,
        *,
        reference: Sampler,
        ratios: Mapping[str, RatioEvaluator],
        intensity: IntensityModel,
        trainer: Callable[..., Any] | Any,
        systematics: Mapping[
            str,
            Sequence[RuntimeSystematic | SystematicAnchor],
        ]
        | None = None,
        fnf_systematics: Mapping[str, Any] | None = None,
    ) -> None:
        self.reference = reference
        self.ratios = dict(ratios)
        self.intensity = intensity
        self.trainer = trainer
        self.systematics = {
            component: tuple(anchors)
            for component, anchors in (systematics or {}).items()
        }
        self.fnf_systematics = dict(fnf_systematics or {})

    def fit(
        self,
        *,
        truth_point: Mapping[str, float],
        design_points: Sequence[Mapping[str, float]],
        pilot_events: int,
        seed: int = 0,
        clip_quantile: float = 0.9999,
        floor_fraction: float = 1.0e-5,
        trainer_kwargs: Mapping[str, Any] | None = None,
    ) -> NISDesignResult:
        if not 0.5 < clip_quantile <= 1.0:
            raise ValueError("clip_quantile must lie in (0.5, 1].")
        rng = np.random.default_rng(seed)
        values = _draw(self.reference, int(pilot_events), rng)
        raw = _evaluate_ratios(self.ratios, values)
        normalizer = RatioNormalizer.fit(raw)
        normalized = normalizer.normalize(raw)
        reference_weights = np.full(len(values), 1.0 / len(values))
        _, _, _, systematic_anchors, _ = _evaluate_asimov_intensity(
            intensity=self.intensity,
            values=values,
            normalized_ratios=normalized,
            reference_weights=reference_weights,
            point=truth_point,
            systematics=self.systematics,
            fnf_systematics=self.fnf_systematics,
        )
        amplitude = scan_influence_amplitude(
            self.intensity,
            raw,
            truth_point=truth_point,
            design_points=design_points,
            reference_weights=reference_weights,
            systematics=systematic_anchors,
            event_values=values,
            fnf_systematics=self.fnf_systematics,
        )
        positive = amplitude[amplitude > 0]
        if not len(positive):
            raise RuntimeError("The NIS influence amplitude is identically zero.")
        floor = float(floor_fraction) * float(np.median(positive))
        ceiling = float(np.quantile(amplitude, clip_quantile))
        training_weights = np.clip(amplitude, floor, ceiling)
        kwargs = dict(trainer_kwargs or {})
        if hasattr(self.trainer, "fit"):
            proposal = self.trainer.fit(
                values, sample_weights=training_weights, **kwargs
            )
        else:
            proposal = self.trainer(values, sample_weights=training_weights, **kwargs)
        diagnostics = {
            "pilot_events": float(len(values)),
            "amplitude_floor": floor,
            "amplitude_ceiling": ceiling,
            "amplitude_median": float(np.median(amplitude)),
            "amplitude_q99": float(np.quantile(amplitude, 0.99)),
        }
        return NISDesignResult(
            proposal=proposal,
            pilot_values=values,
            pilot_amplitude=amplitude,
            training_weights=training_weights,
            normalizer=normalizer,
            systematic_anchors=systematic_anchors,
            fnf_components=tuple(self.fnf_systematics),
            diagnostics=diagnostics,
        )


class NISAsimovBuilder:
    """Build a self-normalized Asimov measure from a defensive proposal."""

    def __init__(
        self,
        *,
        proposal: DefensiveMixture,
        ratios: Mapping[str, RatioEvaluator],
        intensity: IntensityModel,
        features: tuple[str, ...] | list[str],
        systematics: Mapping[
            str,
            Sequence[RuntimeSystematic | SystematicAnchor],
        ]
        | None = None,
        fnf_systematics: Mapping[str, Any] | None = None,
    ) -> None:
        self.proposal = proposal
        self.ratios = dict(ratios)
        self.intensity = intensity
        self.features = tuple(features)
        self.systematics = {
            component: tuple(anchors)
            for component, anchors in (systematics or {}).items()
        }
        self.fnf_systematics = dict(fnf_systematics or {})

    def build(
        self,
        point: Mapping[str, float],
        *,
        n_events: int,
        seed: int = 0,
    ) -> AsimovResult:
        if int(n_events) < 1:
            raise ValueError("n_events must be positive.")
        rng = np.random.default_rng(seed)
        values, log_q_over_g = self.proposal.sample_with_reference_log_weight(
            int(n_events), rng=rng
        )
        reference_weights = normalize_log_weights(log_q_over_g)
        raw = _evaluate_ratios(self.ratios, values)
        normalizer = RatioNormalizer.fit(
            raw,
            reference_weights,
            metadata={
                "mode": "same_support_nis",
                "epsilon": self.proposal.epsilon,
            },
        )
        normalized = normalizer.normalize(raw)
        (
            h,
            expected,
            component_weights,
            systematic_anchors,
            morph_metadata,
        ) = _evaluate_asimov_intensity(
            intensity=self.intensity,
            values=values,
            normalized_ratios=normalized,
            reference_weights=reference_weights,
            point=point,
            systematics=self.systematics,
            fnf_systematics=self.fnf_systematics,
        )
        event_weights = reference_weights * h
        total_weight = float(np.sum(event_weights))
        if not np.isclose(total_weight, expected, rtol=1e-10, atol=1e-12):
            raise RuntimeError(
                "NIS same-support normalization failed yield closure: "
                f"sum(weights)={total_weight}, expected={expected}."
            )
        summary = weight_summary(log_weights=log_q_over_g)
        events = WeightedEvents(
            values=values,
            weights=event_weights,
            features=self.features,
            metadata={
                "kind": "defensive_nis_asimov",
                "seed": int(seed),
                "epsilon": self.proposal.epsilon,
                "requested_raw_count": int(n_events),
                "raw_count": int(n_events),
                "expected_yield_from_model": expected,
                "total_weight": total_weight,
                "yield_closure_residual": total_weight - expected,
                "ESS": effective_sample_size(event_weights),
                "intensity_fingerprint": self.intensity.fingerprint,
                "systematic_morphs": morph_metadata,
                "fnf_morphs": {
                    component: metadata
                    for component, metadata in morph_metadata.items()
                    if "fnf_shape_partition" in metadata
                },
                "auxiliary_observations": {
                    parameter.name: float(point[parameter.name])
                    for parameter in self.intensity.parameters
                    if parameter.constrained
                },
                "reference_weight_diagnostics": summary,
                "max_q_over_g": float(np.max(np.exp(log_q_over_g))),
                "q_over_g_bound": 1.0 / self.proposal.epsilon,
            },
            columns={
                **{f"ratio_{name}": ratio for name, ratio in normalized.items()},
                "reference_weight": reference_weights,
                "log_q_over_g": log_q_over_g,
            },
        )
        return AsimovResult(
            events=events,
            point={name: float(value) for name, value in point.items()},
            normalizer=normalizer,
            raw_ratios=raw,
            normalized_ratios=normalized,
            reference_weights=reference_weights,
            component_weights=component_weights,
            systematic_anchors=systematic_anchors,
            fnf_components=tuple(self.fnf_systematics),
            auxiliary_observations={
                parameter.name: float(point[parameter.name])
                for parameter in self.intensity.parameters
                if parameter.constrained
            },
        )
