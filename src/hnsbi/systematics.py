"""Normalized up/down systematic anchors and upstream training orchestration."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_artifact_manifest

_PORTABLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _validate_portable_name(value: str, field: str) -> str:
    value = str(value)
    if not _PORTABLE_NAME.fullmatch(value):
        raise ValueError(
            f"{field} must be a portable name containing only letters, "
            "numbers, '.', '_', or '-', and cannot start with punctuation."
        )
    return value


def _code4p(alpha: float, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """HistFactory strategy-5 interpolation used by nsbi-common-utils."""

    if abs(alpha) > 1.0:
        return up**alpha if alpha > 1.0 else down ** (-alpha)
    log_up = np.log(up)
    log_down = np.log(down)
    up_log = up * log_up
    down_log = -(down * log_down)
    up_log2 = up_log * log_up
    down_log2 = -(down_log * log_down)
    s0 = (up + down) / 2.0
    a0 = (up - down) / 2.0
    s1 = (up_log + down_log) / 2.0
    a1 = (up_log - down_log) / 2.0
    s2 = (up_log2 + down_log2) / 2.0
    a2 = (up_log2 - down_log2) / 2.0
    coefficients = (
        (15.0 * a0 - 7.0 * s1 + a2) / 8.0,
        (-24.0 + 24.0 * s0 - 9.0 * a1 + s2) / 8.0,
        (-5.0 * a0 + 5.0 * s1 - a2) / 4.0,
        (12.0 - 12.0 * s0 + 7.0 * a1 - s2) / 4.0,
        (3.0 * a0 - 3.0 * s1 + a2) / 8.0,
        (-8.0 + 8.0 * s0 - 5.0 * a1 + s2) / 8.0,
    )
    polynomial = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        polynomial = coefficient + alpha * polynomial
    return 1.0 + alpha * polynomial


def _interpolate(
    alpha: float,
    up: np.ndarray,
    down: np.ndarray,
    interpolation: str,
) -> np.ndarray:
    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite.")
    if interpolation == "nsbi_code4p":
        result = _code4p(alpha, up, down)
    elif interpolation == "linear" and alpha >= 0:
        result = 1.0 + alpha * (up - 1.0)
    elif interpolation == "linear":
        result = 1.0 + (-alpha) * (down - 1.0)
    else:
        raise ValueError("interpolation must be 'linear' or 'nsbi_code4p'.")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(
            "Systematic interpolation produced a non-finite or negative factor."
        )
    return result


@dataclass(frozen=True)
class SystematicSpecification:
    """Portable metadata for one nuisance effect in a workspace."""

    parameter: str
    component: str
    yield_up: float
    yield_down: float
    interpolation: str = "nsbi_code4p"

    def __post_init__(self) -> None:
        parameter = _validate_portable_name(self.parameter, "parameter")
        component = _validate_portable_name(self.component, "component")
        if self.interpolation not in {"linear", "nsbi_code4p"}:
            raise ValueError("interpolation must be 'linear' or 'nsbi_code4p'.")
        yield_up = float(self.yield_up)
        yield_down = float(self.yield_down)
        if (
            not np.isfinite(yield_up)
            or not np.isfinite(yield_down)
            or yield_up < 0
            or yield_down < 0
        ):
            raise ValueError(
                "Systematic yield anchors must be finite and non-negative."
            )
        if self.interpolation == "nsbi_code4p" and (yield_up <= 0 or yield_down <= 0):
            raise ValueError("nsbi_code4p requires strictly positive yield anchors.")
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "yield_up", yield_up)
        object.__setattr__(self, "yield_down", yield_down)


@dataclass(frozen=True)
class RuntimeSystematic:
    """Callable up/down variation ratios used for generative toys."""

    parameter: str
    component: str
    ratio_up: Any
    ratio_down: Any
    yield_up: float = 1.0
    yield_down: float = 1.0
    interpolation: str = "nsbi_code4p"

    def __post_init__(self) -> None:
        _validate_portable_name(self.parameter, "parameter")
        _validate_portable_name(self.component, "component")
        if not callable(self.ratio_up) or not callable(self.ratio_down):
            raise TypeError("Runtime systematic ratios must be callable.")
        probe = SystematicAnchor(
            parameter=self.parameter,
            component=self.component,
            ratio_up=np.ones(1),
            ratio_down=np.ones(1),
            yield_up=self.yield_up,
            yield_down=self.yield_down,
            interpolation=self.interpolation,
        )
        object.__setattr__(self, "yield_up", probe.yield_up)
        object.__setattr__(self, "yield_down", probe.yield_down)

    def raw_shape(self, values: Any, alpha: float) -> np.ndarray:
        up, down = self.evaluate_anchors(values)
        return _interpolate(float(alpha), up, down, self.interpolation)

    def yield_factor(self, alpha: float) -> float:
        value = _interpolate(
            float(alpha),
            np.asarray([self.yield_up]),
            np.asarray([self.yield_down]),
            self.interpolation,
        )
        return float(value[0])

    def evaluate_anchors(self, values: Any) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate and validate the learned up/down ratios on one support."""

        rows = len(values)
        up = np.asarray(self.ratio_up(values), dtype=np.float64).reshape(-1)
        down = np.asarray(self.ratio_down(values), dtype=np.float64).reshape(-1)
        if (
            len(up) != rows
            or len(down) != rows
            or not np.isfinite(up).all()
            or not np.isfinite(down).all()
            or np.any(up < 0)
            or np.any(down < 0)
        ):
            raise ValueError(
                "Runtime systematic evaluators must return aligned finite "
                "non-negative ratios."
            )
        if self.interpolation == "nsbi_code4p" and (
            np.any(up <= 0) or np.any(down <= 0)
        ):
            raise ValueError("nsbi_code4p requires strictly positive runtime ratios.")
        return up, down


@dataclass(frozen=True)
class SystematicAnchor:
    """Up/down anchors for one component and nuisance parameter."""

    parameter: str
    component: str
    ratio_up: np.ndarray
    ratio_down: np.ndarray
    yield_up: float = 1.0
    yield_down: float = 1.0
    interpolation: str = "linear"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "parameter",
            _validate_portable_name(self.parameter, "parameter"),
        )
        object.__setattr__(
            self,
            "component",
            _validate_portable_name(self.component, "component"),
        )
        up = np.asarray(self.ratio_up, dtype=np.float64).reshape(-1)
        down = np.asarray(self.ratio_down, dtype=np.float64).reshape(-1)
        if len(up) != len(down) or not len(up):
            raise ValueError("Systematic ratio anchors must be non-empty and aligned.")
        if (
            not np.isfinite(up).all()
            or not np.isfinite(down).all()
            or np.any(up < 0)
            or np.any(down < 0)
        ):
            raise ValueError(
                "Systematic ratio anchors must be finite and non-negative."
            )
        if self.interpolation not in {"linear", "nsbi_code4p"}:
            raise ValueError("interpolation must be 'linear' or 'nsbi_code4p'.")
        if self.interpolation == "nsbi_code4p" and (
            np.any(up <= 0) or np.any(down <= 0)
        ):
            raise ValueError("nsbi_code4p requires strictly positive up/down anchors.")
        yield_up = float(self.yield_up)
        yield_down = float(self.yield_down)
        if (
            not np.isfinite(yield_up)
            or not np.isfinite(yield_down)
            or yield_up < 0
            or yield_down < 0
        ):
            raise ValueError(
                "Systematic yield anchors must be finite and non-negative."
            )
        if self.interpolation == "nsbi_code4p" and (yield_up <= 0 or yield_down <= 0):
            raise ValueError("nsbi_code4p requires strictly positive yield anchors.")
        object.__setattr__(self, "ratio_up", up)
        object.__setattr__(self, "ratio_down", down)
        object.__setattr__(self, "yield_up", yield_up)
        object.__setattr__(self, "yield_down", yield_down)

    def raw_shape(self, alpha: float) -> np.ndarray:
        alpha = float(alpha)
        return _interpolate(alpha, self.ratio_up, self.ratio_down, self.interpolation)

    def normalized_shape(
        self,
        alpha: float,
        *,
        nominal_process_ratio: np.ndarray,
        integration_weights: np.ndarray,
    ) -> np.ndarray:
        """Normalize the interpolated conditional shape on the active support."""

        nominal = np.asarray(nominal_process_ratio, dtype=np.float64).reshape(-1)
        weights = np.asarray(integration_weights, dtype=np.float64).reshape(-1)
        if len(nominal) != len(self.ratio_up) or len(weights) != len(nominal):
            raise ValueError("Nominal ratio, anchors, and weights must align.")
        if np.any(weights < 0) or not np.sum(weights) > 0:
            raise ValueError("integration_weights must be a positive measure.")
        weights = weights / np.sum(weights)
        raw = self.raw_shape(alpha)
        partition = float(np.sum(weights * nominal * raw))
        if not partition > 0:
            raise ValueError("Systematic shape partition is not positive.")
        return raw / partition

    def yield_factor(self, alpha: float) -> float:
        alpha = float(alpha)
        return float(
            _interpolate(
                alpha,
                np.asarray([self.yield_up]),
                np.asarray([self.yield_down]),
                self.interpolation,
            )[0]
        )

    def write_workspace_modifier(self, directory: str | Path) -> dict[str, Any]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        up_path = directory / f"{self.component}_{self.parameter}_up.npy"
        down_path = directory / f"{self.component}_{self.parameter}_down.npy"
        np.save(up_path, self.ratio_up)
        np.save(down_path, self.ratio_down)
        manifest_path = directory / f"{self.component}_{self.parameter}.manifest.json"
        write_artifact_manifest(
            manifest_path,
            artifact_type="systematic-anchor",
            files={
                "up-ratio": up_path,
                "down-ratio": down_path,
            },
            metadata={
                "component": self.component,
                "interpolation": self.interpolation,
                "parameter": self.parameter,
                "rows": len(self.ratio_up),
                "yield_down": self.yield_down,
                "yield_up": self.yield_up,
            },
        )
        return {
            "name": self.parameter,
            "type": "normplusshape",
            "data": {
                "hi_data": [float(self.yield_up)],
                "lo_data": [float(self.yield_down)],
                "hi_ratio": str(up_path),
                "lo_ratio": str(down_path),
            },
            "hnsbi": {
                "shape_normalization": "reference_support",
                "interpolation": self.interpolation,
                "manifest": str(manifest_path),
            },
        }


@dataclass(frozen=True)
class SystematicEvaluation:
    """One component's normalized shape and separated yield morph."""

    component: str
    shape_ratio: np.ndarray
    shape_factor: np.ndarray
    yield_factor: float
    shape_partition: float


class SystematicRatioEvaluator:
    """Evaluate normalized nuisance morphs on a fixed reference quadrature.

    The up/down anchors are conditional-shape ratios relative to the nominal
    process. Each anchor is normalized under the nominal process measure when
    it is bound to the support. At an arbitrary parameter point, all active
    shape factors are interpolated, multiplied, and normalized jointly. This
    is the exact convention used by the native likelihood and Asimov builders.
    """

    def __init__(
        self,
        *,
        component: str,
        nominal_process_ratio: np.ndarray,
        integration_weights: np.ndarray,
        anchors: Sequence[SystematicAnchor],
    ) -> None:
        self.component = _validate_portable_name(component, "component")
        nominal = np.asarray(nominal_process_ratio, dtype=np.float64).reshape(-1)
        weights = np.asarray(integration_weights, dtype=np.float64).reshape(-1)
        if (
            not len(nominal)
            or len(weights) != len(nominal)
            or not np.isfinite(nominal).all()
            or np.any(nominal < 0)
        ):
            raise ValueError(
                "Nominal process ratio must be a non-empty finite "
                "non-negative array aligned with integration_weights."
            )
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not float(np.sum(weights)) > 0
        ):
            raise ValueError(
                "integration_weights must define a finite positive measure."
            )
        parsed = tuple(anchors)
        if not parsed:
            raise ValueError("At least one systematic anchor is required.")
        names: list[str] = []
        for anchor in parsed:
            if anchor.component != self.component:
                raise ValueError(
                    "Systematic anchor component does not match the evaluator."
                )
            if len(anchor.ratio_up) != len(nominal):
                raise ValueError(
                    "Systematic anchor arrays must align with the reference support."
                )
            names.append(anchor.parameter)
        if len(names) != len(set(names)):
            raise ValueError(
                f"Component {self.component!r} repeats a systematic parameter."
            )
        self.nominal_process_ratio = nominal
        self.integration_weights = weights / np.sum(weights)
        self.anchors = parsed

    @classmethod
    def bind(
        cls,
        *,
        component: str,
        values: np.ndarray,
        nominal_process_ratio: np.ndarray,
        integration_weights: np.ndarray,
        systematics: Sequence[RuntimeSystematic | SystematicAnchor],
    ) -> SystematicRatioEvaluator:
        """Evaluate runtime anchors and normalize them on ``values``.

        Pre-evaluated :class:`SystematicAnchor` inputs are normalized again on
        this support. Already-normalized anchors are unchanged up to numerical
        precision, while raw anchors get the required conditional-density
        normalization.
        """

        nominal = np.asarray(nominal_process_ratio, dtype=np.float64).reshape(-1)
        weights = np.asarray(integration_weights, dtype=np.float64).reshape(-1)
        if len(nominal) != len(values) or len(weights) != len(values):
            raise ValueError(
                "Values, nominal process ratio, and integration weights must align."
            )
        if (
            not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not float(np.sum(weights)) > 0
        ):
            raise ValueError(
                "integration_weights must define a finite positive measure."
            )
        probability = weights / np.sum(weights)
        nominal_measure = probability * nominal
        parsed: list[SystematicAnchor] = []
        for systematic in systematics:
            if systematic.component != component:
                raise ValueError("Systematic component does not match its mapping key.")
            if isinstance(systematic, RuntimeSystematic):
                up, down = systematic.evaluate_anchors(values)
            elif isinstance(systematic, SystematicAnchor):
                up = systematic.ratio_up
                down = systematic.ratio_down
            else:
                raise TypeError(
                    "Systematics must be RuntimeSystematic or SystematicAnchor "
                    "instances."
                )
            normalized: dict[str, np.ndarray] = {}
            for label, ratio in (("up", up), ("down", down)):
                ratio = np.asarray(ratio, dtype=np.float64).reshape(-1)
                if len(ratio) != len(values):
                    raise ValueError(
                        f"Systematic {component!r} {label} anchor is not aligned."
                    )
                partition = float(np.sum(nominal_measure * ratio))
                if not np.isfinite(partition) or partition <= 0:
                    raise ValueError(
                        f"Systematic {component!r} {label} anchor has a "
                        "non-positive shape partition."
                    )
                normalized[label] = ratio / partition
            parsed.append(
                SystematicAnchor(
                    parameter=systematic.parameter,
                    component=component,
                    ratio_up=normalized["up"],
                    ratio_down=normalized["down"],
                    yield_up=systematic.yield_up,
                    yield_down=systematic.yield_down,
                    interpolation=systematic.interpolation,
                )
            )
        return cls(
            component=component,
            nominal_process_ratio=nominal,
            integration_weights=probability,
            anchors=parsed,
        )

    def evaluate(self, point: Mapping[str, float]) -> SystematicEvaluation:
        """Return the joint shape/yield morph at one complete parameter point."""

        combined_shape = np.ones_like(self.nominal_process_ratio)
        yield_factor = 1.0
        for anchor in self.anchors:
            if anchor.parameter not in point:
                raise KeyError(f"Missing systematic parameter {anchor.parameter!r}.")
            alpha = float(point[anchor.parameter])
            combined_shape *= anchor.raw_shape(alpha)
            yield_factor *= anchor.yield_factor(alpha)
        partition = float(
            np.sum(
                self.integration_weights * self.nominal_process_ratio * combined_shape
            )
        )
        if not np.isfinite(partition) or partition <= 0:
            raise ValueError(
                f"Systematic shape partition for {self.component!r} is not positive."
            )
        shape_factor = combined_shape / partition
        shape_ratio = self.nominal_process_ratio * shape_factor
        if (
            not np.isfinite(yield_factor)
            or yield_factor < 0
            or not np.isfinite(shape_ratio).all()
            or np.any(shape_ratio < 0)
        ):
            raise ValueError(f"Systematic morph for {self.component!r} is invalid.")
        return SystematicEvaluation(
            component=self.component,
            shape_ratio=shape_ratio,
            shape_factor=shape_factor,
            yield_factor=float(yield_factor),
            shape_partition=partition,
        )


class SystematicsTrainer:
    """Thin interface over the configured density-ratio backend."""

    def __init__(self, ratio_trainer: Any) -> None:
        self.ratio_trainer = ratio_trainer

    def fit_variation(
        self,
        *,
        nominal: Any,
        up: Any,
        down: Any,
        parameter: str,
        component: str,
        output_dir: str | Path,
        features: Sequence[str],
        nominal_weights: Any | None = None,
        up_weights: Any | None = None,
        down_weights: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train ``up/nominal`` and ``down/nominal`` using the same backend.

        The backend is expected to be the nsbi-common-utils adapter, so its
        calibration, overtraining, reweighting, and normalization diagnostics
        remain available on each returned training result.
        """

        output_dir = Path(output_dir)
        up_result = self.ratio_trainer.fit(
            up,
            nominal,
            features=features,
            output_directory=output_dir / "up",
            numerator_weights=up_weights,
            denominator_weights=nominal_weights,
            numerator_name=f"{component}_{parameter}_up",
            denominator_name=f"{component}_nominal",
            **kwargs,
        )
        down_result = self.ratio_trainer.fit(
            down,
            nominal,
            features=features,
            output_directory=output_dir / "down",
            numerator_weights=down_weights,
            denominator_weights=nominal_weights,
            numerator_name=f"{component}_{parameter}_down",
            denominator_name=f"{component}_nominal",
            **kwargs,
        )
        return {"up": up_result, "down": down_result}
