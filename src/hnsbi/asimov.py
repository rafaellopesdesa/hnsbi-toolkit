"""Reference-normalized weighted unbinned Asimov construction."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_artifact_manifest
from .data import WeightedEvents
from .diagnostics import effective_sample_size, json_safe
from .intensity import IntensityModel, RatioNormalizer
from .protocols import RatioEvaluator, Sampler
from .systematics import (
    RuntimeSystematic,
    SystematicAnchor,
    SystematicRatioEvaluator,
)


def _draw(
    sampler: Sampler,
    n: int,
    rng: np.random.Generator,
    **kwargs: Any,
) -> np.ndarray:
    try:
        values = sampler.sample(int(n), rng=rng, **kwargs)
    except TypeError:
        values = sampler.sample(int(n), **kwargs)
    values = np.asarray(values)
    if values.ndim != 2 or len(values) != int(n):
        raise ValueError("Sampler returned an array with the wrong shape.")
    if not np.isfinite(values).all():
        raise ValueError("Sampler returned non-finite events.")
    return values


def _evaluate_ratios(
    evaluators: Mapping[str, RatioEvaluator],
    values: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, evaluator in evaluators.items():
        ratio = np.asarray(evaluator(values), dtype=np.float64).reshape(-1)
        if len(ratio) != len(values):
            raise ValueError(f"Ratio evaluator {name!r} returned wrong row count.")
        if not np.isfinite(ratio).all() or np.any(ratio < 0):
            raise ValueError(
                f"Ratio evaluator {name!r} must return finite non-negative values."
            )
        result[name] = ratio
    return result


@dataclass
class AsimovResult:
    """Weighted sample plus every normalization used to construct it."""

    events: WeightedEvents
    point: dict[str, float]
    normalizer: RatioNormalizer
    raw_ratios: dict[str, np.ndarray]
    normalized_ratios: dict[str, np.ndarray]
    reference_weights: np.ndarray
    component_weights: dict[str, np.ndarray]
    systematic_anchors: dict[str, tuple[SystematicAnchor, ...]]
    auxiliary_observations: dict[str, float]

    @property
    def raw_count(self) -> int:
        return self.events.raw_count

    @property
    def ess(self) -> float:
        return self.events.ess

    @property
    def component_ess(self) -> dict[str, float]:
        return {
            name: effective_sample_size(weights)
            for name, weights in self.component_weights.items()
        }

    def write_nsbi_arrays(self, directory: str | Path) -> dict[str, Path]:
        """Write the pre-evaluated arrays consumed by nsbi-common-utils."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        weights_path = directory / "asimov_weights.npy"
        np.save(weights_path, self.events.weights)
        paths["weights"] = weights_path
        reference_weights_path = directory / "reference_weights.npy"
        np.save(reference_weights_path, self.reference_weights)
        paths["reference_weights"] = reference_weights_path
        for name, ratio in self.normalized_ratios.items():
            ratio_path = directory / f"ratio_{name}.npy"
            np.save(ratio_path, ratio)
            paths[f"ratio:{name}"] = ratio_path
        metadata_path = directory / "asimov_metadata.json"
        metadata = {
            **self.events.metadata,
            "point": self.point,
            "ratio_normalizers": dict(self.normalizer.means),
            "ratio_normalizer_standard_errors": dict(self.normalizer.standard_errors),
            "component_ess": self.component_ess,
            "auxiliary_observations": dict(self.auxiliary_observations),
            "systematic_parameters": {
                component: [anchor.parameter for anchor in anchors]
                for component, anchors in self.systematic_anchors.items()
            },
        }
        metadata_path.write_text(
            json.dumps(
                json_safe(metadata),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        paths["metadata"] = metadata_path
        manifest_path = directory / "asimov_arrays.manifest.json"
        files = {
            "event-weights": weights_path,
            "reference-integration-weights": reference_weights_path,
            "metadata": metadata_path,
            **{
                f"process-ratio:{name}": path
                for name, path in (
                    (name, paths[f"ratio:{name}"]) for name in self.normalized_ratios
                )
            },
        }
        write_artifact_manifest(
            manifest_path,
            artifact_type="asimov-array-bundle",
            files=files,
            metadata={
                "features": list(self.events.features),
                "intensity_fingerprint": self.events.metadata.get(
                    "intensity_fingerprint"
                ),
                "rows": self.raw_count,
                "samples": list(self.normalized_ratios),
            },
        )
        paths["manifest"] = manifest_path
        return paths


def _evaluate_asimov_intensity(
    *,
    intensity: IntensityModel,
    values: np.ndarray,
    normalized_ratios: Mapping[str, np.ndarray],
    reference_weights: np.ndarray,
    point: Mapping[str, float],
    systematics: Mapping[
        str,
        Sequence[RuntimeSystematic | SystematicAnchor],
    ]
    | None,
) -> tuple[
    np.ndarray,
    float,
    dict[str, np.ndarray],
    dict[str, tuple[SystematicAnchor, ...]],
    dict[str, dict[str, float]],
]:
    """Evaluate the physical intensity with normalized shape systematics."""

    validated_point = intensity.validate_point(point)
    supplied = {
        component: tuple(anchors) for component, anchors in (systematics or {}).items()
    }
    unknown_components = set(supplied).difference(intensity.component_names)
    if unknown_components:
        raise ValueError(
            f"Systematics reference unknown components {sorted(unknown_components)}."
        )
    known_parameters = {parameter.name for parameter in intensity.parameters}
    evaluators: dict[str, SystematicRatioEvaluator] = {}
    bound_anchors: dict[str, tuple[SystematicAnchor, ...]] = {}
    for component, anchors in supplied.items():
        if not anchors:
            raise ValueError(f"Systematic component {component!r} has no anchors.")
        unknown_parameters = {anchor.parameter for anchor in anchors}.difference(
            known_parameters
        )
        if unknown_parameters:
            raise ValueError(
                f"Systematics for {component!r} reference unknown parameters "
                f"{sorted(unknown_parameters)}."
            )
        evaluator = SystematicRatioEvaluator.bind(
            component=component,
            values=values,
            nominal_process_ratio=normalized_ratios[component],
            integration_weights=reference_weights,
            systematics=anchors,
        )
        evaluators[component] = evaluator
        bound_anchors[component] = evaluator.anchors

    component_yields = intensity.component_yields(validated_point)
    differential = np.zeros(len(values), dtype=np.float64)
    component_weights: dict[str, np.ndarray] = {}
    morph_metadata: dict[str, dict[str, float]] = {}
    for component in intensity.component_names:
        component_yield = component_yields[component]
        shape_ratio = np.asarray(
            normalized_ratios[component], dtype=np.float64
        ).reshape(-1)
        if component in evaluators:
            evaluation = evaluators[component].evaluate(validated_point)
            component_yield *= evaluation.yield_factor
            shape_ratio = evaluation.shape_ratio
            morph_metadata[component] = {
                "shape_partition": evaluation.shape_partition,
                "yield_factor": evaluation.yield_factor,
            }
        contribution = component_yield * shape_ratio
        differential += contribution
        component_weights[component] = reference_weights * contribution
        component_yields[component] = component_yield
    expected = float(sum(component_yields.values()))
    if not np.isfinite(expected) or expected < 0:
        raise ValueError(
            "The systematic-aware intensity has a non-finite or negative "
            f"expected yield ({expected}) at the requested point."
        )
    if not np.isfinite(differential).all() or np.any(differential < 0):
        raise ValueError(
            "The systematic-aware differential intensity is non-finite or "
            "negative on the Asimov support."
        )
    return (
        differential,
        expected,
        component_weights,
        bound_anchors,
        morph_metadata,
    )


class AsimovBuilder:
    """Build a direct reference-flow Asimov measure.

    By default, every ratio is divided by its empirical expectation on the
    *same* support points used by the returned sample. This is the finite
    quadrature convention that enforces exact yield and score closure in
    yield-only signal-strength directions.
    """

    def __init__(
        self,
        *,
        reference: Sampler,
        ratios: Mapping[str, RatioEvaluator],
        intensity: IntensityModel,
        features: tuple[str, ...] | list[str],
        normalizer: RatioNormalizer | None = None,
        systematics: Mapping[
            str,
            Sequence[RuntimeSystematic | SystematicAnchor],
        ]
        | None = None,
    ) -> None:
        self.reference = reference
        self.ratios = dict(ratios)
        self.intensity = intensity
        self.features = tuple(features)
        self.normalizer = normalizer
        self.systematics = {
            component: tuple(anchors)
            for component, anchors in (systematics or {}).items()
        }
        if set(self.ratios) != set(self.intensity.component_names):
            raise ValueError(
                "Ratio evaluators must match the intensity component names."
            )

    def build(
        self,
        point: Mapping[str, float],
        *,
        n_events: int,
        seed: int = 0,
        normalization: str = "sample",
    ) -> AsimovResult:
        n_events = int(n_events)
        if n_events < 1:
            raise ValueError("n_events must be positive.")
        if normalization not in {"sample", "fixed"}:
            raise ValueError("normalization must be 'sample' or 'fixed'.")
        if normalization == "fixed" and self.normalizer is None:
            raise ValueError("Fixed normalization requires a RatioNormalizer.")
        rng = np.random.default_rng(seed)
        values = _draw(self.reference, n_events, rng)
        if values.shape[1] != len(self.features):
            raise ValueError("Reference feature count does not match features.")
        raw = _evaluate_ratios(self.ratios, values)
        reference_weights = np.full(n_events, 1.0 / n_events)
        if normalization == "sample":
            normalizer = RatioNormalizer.fit(
                raw,
                reference_weights,
                metadata={"mode": "same_support"},
            )
        else:
            normalizer = self.normalizer
            assert normalizer is not None
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
        )
        event_weights = reference_weights * h
        total_weight = float(np.sum(event_weights))
        if normalization == "sample" and not np.isclose(
            total_weight,
            expected,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(
                "Same-support Asimov normalization failed yield closure: "
                f"sum(weights)={total_weight}, expected={expected}."
            )
        events = WeightedEvents(
            values=values,
            weights=event_weights,
            features=self.features,
            metadata={
                "kind": "direct_reference_asimov",
                "seed": int(seed),
                "requested_raw_count": n_events,
                "raw_count": n_events,
                "expected_yield_from_model": expected,
                "total_weight": total_weight,
                "yield_closure_residual": total_weight - expected,
                "ESS": effective_sample_size(event_weights),
                "reference_ESS": effective_sample_size(reference_weights),
                "normalization_mode": normalization,
                "intensity_fingerprint": self.intensity.fingerprint,
                "systematic_morphs": morph_metadata,
                "auxiliary_observations": {
                    parameter.name: float(point[parameter.name])
                    for parameter in self.intensity.parameters
                    if parameter.constrained
                },
            },
            columns={f"ratio_{name}": ratio for name, ratio in normalized.items()},
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
            auxiliary_observations={
                parameter.name: float(point[parameter.name])
                for parameter in self.intensity.parameters
                if parameter.constrained
            },
        )
