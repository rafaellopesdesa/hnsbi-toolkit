"""Parameter-dependent process intensities and ratio normalization."""

from __future__ import annotations

import hashlib
import json
import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactManifest, write_artifact_manifest
from .expressions import Expression

_PORTABLE_COMPONENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class Parameter:
    """One fit parameter and its workspace metadata."""

    name: str
    nominal: float
    bounds: tuple[float, float] | None = None
    constrained: bool = False
    constraint_mean: float = 0.0
    constraint_sigma: float = 1.0

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.name.isidentifier()
            or keyword.iskeyword(self.name)
        ):
            raise ValueError("Parameter names must be non-keyword Python identifiers.")
        if not np.isfinite(self.nominal):
            raise ValueError("Parameter nominal must be finite.")
        if self.bounds is not None:
            low, high = self.bounds
            if (
                not np.isfinite(low)
                or not np.isfinite(high)
                or not low < high
                or not low <= self.nominal <= high
            ):
                raise ValueError(f"Invalid bounds or nominal value for {self.name!r}.")
        if not np.isfinite(self.constraint_mean):
            raise ValueError("constraint_mean must be finite.")
        if not np.isfinite(self.constraint_sigma) or self.constraint_sigma <= 0:
            raise ValueError("constraint_sigma must be finite and positive.")


@dataclass(frozen=True)
class Component:
    """One physics process in an intensity decomposition."""

    name: str
    nominal_yield: float
    multiplier: Expression | str | float = "1"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _PORTABLE_COMPONENT_NAME.fullmatch(self.name):
            raise ValueError(
                "Component names must be portable names containing only "
                "letters, numbers, '.', '_', or '-', and cannot start with "
                "punctuation."
            )
        if not np.isfinite(self.nominal_yield) or self.nominal_yield < 0:
            raise ValueError("nominal_yield must be finite and non-negative.")
        expression = (
            self.multiplier
            if isinstance(self.multiplier, Expression)
            else Expression.parse(self.multiplier)
        )
        object.__setattr__(self, "multiplier", expression)


@dataclass(frozen=True)
class RatioNormalizer:
    """Monte-Carlo estimates of ``E_reference[raw ratio]``."""

    means: Mapping[str, float]
    standard_errors: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.means:
            raise ValueError("At least one ratio normalization is required.")
        names = tuple(self.means)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Ratio normalizer names must be non-empty strings.")
        for name, value in self.means.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"Ratio normalizer {name!r} must be finite and positive."
                )
        unknown_errors = set(self.standard_errors).difference(names)
        if unknown_errors:
            raise ValueError(
                "Ratio normalizer standard errors reference unknown ratios "
                f"{sorted(unknown_errors)}."
            )
        for name, value in self.standard_errors.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"Ratio normalizer error {name!r} must be finite and non-negative."
                )
        object.__setattr__(
            self,
            "means",
            {name: float(value) for name, value in self.means.items()},
        )
        object.__setattr__(
            self,
            "standard_errors",
            {name: float(value) for name, value in self.standard_errors.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def fit(
        cls,
        ratios: Mapping[str, np.ndarray],
        weights: np.ndarray | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> RatioNormalizer:
        if not ratios:
            raise ValueError("At least one ratio array is required.")
        arrays = {
            name: np.asarray(values, dtype=np.float64).reshape(-1)
            for name, values in ratios.items()
        }
        length = len(next(iter(arrays.values())))
        if length < 1 or any(len(values) != length for values in arrays.values()):
            raise ValueError("Ratio arrays must be non-empty and aligned.")
        for name, values in arrays.items():
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Ratio {name!r} must be finite and non-negative.")
        if weights is None:
            probability = np.full(length, 1.0 / length)
            weight_ess = float(length)
        else:
            probability = np.asarray(weights, dtype=np.float64).reshape(-1)
            if (
                len(probability) != length
                or np.any(probability < 0)
                or not np.isfinite(probability).all()
            ):
                raise ValueError("weights must align, be finite and non-negative.")
            total = float(np.sum(probability))
            if not total > 0:
                raise ValueError("weights must have positive sum.")
            probability = probability / total
            weight_ess = float(1.0 / np.sum(probability**2))
        means: dict[str, float] = {}
        errors: dict[str, float] = {}
        for name, values in arrays.items():
            mean = float(np.sum(probability * values))
            variance = float(np.sum(probability * (values - mean) ** 2))
            if not mean > 0:
                raise ValueError(f"Ratio {name!r} has zero reference mean.")
            means[name] = mean
            sum_squared_weights = float(np.sum(probability**2))
            correction = 1.0 - sum_squared_weights
            errors[name] = (
                float(np.sqrt(max(0.0, variance) * sum_squared_weights / correction))
                if correction > np.finfo(np.float64).eps
                else 0.0
            )
        details = dict(metadata or {})
        details.setdefault("normalization_rows", length)
        details.setdefault("normalization_weight_ess", weight_ess)
        return cls(means, errors, details)

    def normalize(self, ratios: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = set(ratios).difference(self.means)
        if missing:
            raise KeyError(f"Missing normalizers for {sorted(missing)}.")
        return {
            name: np.asarray(values, dtype=np.float64) / self.means[name]
            for name, values in ratios.items()
        }

    def write(
        self,
        path: str | Path,
    ) -> tuple[Path, Path]:
        """Write normalization constants and a checksummed manifest.

        The constants are ordinary JSON because they are numerical metadata,
        not executable model state. The adjacent manifest protects the exact
        file used to normalize deployed density-ratio ensembles.
        """

        output = Path(path)
        if output.suffix != ".json":
            raise ValueError("Ratio normalizer output must end in '.json'.")
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "means": dict(self.means),
            "metadata": dict(self.metadata),
            "standard_errors": dict(self.standard_errors),
        }
        try:
            text = (
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Ratio normalizer metadata must be JSON serializable."
            ) from exc
        output.write_text(text, encoding="utf-8")
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        write_artifact_manifest(
            manifest_path,
            artifact_type="density-ratio-normalizer",
            files={"normalization-json": output},
            metadata={"ratios": list(self.means)},
        )
        return output, manifest_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verify: bool = True,
    ) -> RatioNormalizer:
        """Load persisted constants, verifying their sidecar by default."""

        source = Path(path)
        if verify:
            manifest_path = source.with_suffix(source.suffix + ".manifest.json")
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Ratio normalizer manifest is missing: {manifest_path}"
                )
            manifest = ArtifactManifest.load(manifest_path)
            if manifest.artifact_type != "density-ratio-normalizer":
                raise ValueError(
                    "Unexpected artifact type for ratio normalizer: "
                    f"{manifest.artifact_type!r}."
                )
            manifest.verify(manifest_path.parent)
            records = [
                record
                for record in manifest.files
                if record.kind == "normalization-json"
            ]
            if (
                len(manifest.files) != 1
                or len(records) != 1
                or (manifest_path.parent / records[0].path).resolve()
                != source.resolve()
            ):
                raise ValueError(
                    "Ratio normalizer manifest does not bind the requested JSON file."
                )
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not load ratio normalizer from {source}.") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Ratio normalizer JSON must contain an object.")
        return cls(
            means=payload.get("means", {}),
            standard_errors=payload.get("standard_errors", {}),
            metadata=payload.get("metadata", {}),
        )


class IntensityModel:
    """Extended event intensity relative to a normalized reference.

    For components ``c``,

    ``lambda(x | theta) / q_ref(x) =
    sum_c nominal_yield_c * multiplier_c(theta) * r_c(x)``.
    """

    def __init__(
        self,
        components: Sequence[Component],
        parameters: Sequence[Parameter] = (),
    ) -> None:
        self.components = tuple(components)
        self.parameters = tuple(parameters)
        if not self.components:
            raise ValueError("At least one intensity component is required.")
        component_names = [component.name for component in self.components]
        parameter_names = [parameter.name for parameter in self.parameters]
        if len(set(component_names)) != len(component_names):
            raise ValueError("Component names must be unique.")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("Parameter names must be unique.")
        known = set(parameter_names)
        for component in self.components:
            unknown = component.multiplier.names.difference(known)
            if unknown:
                raise ValueError(
                    f"Component {component.name!r} multiplier references "
                    f"unknown parameters {sorted(unknown)}."
                )

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> IntensityModel:
        raw_parameters = value.get("parameters", ())
        if isinstance(raw_parameters, Mapping):
            parameter_items = [
                {"name": name, **dict(config)}
                for name, config in raw_parameters.items()
            ]
        elif isinstance(raw_parameters, Sequence) and not isinstance(
            raw_parameters, (str, bytes)
        ):
            parameter_items = [dict(config) for config in raw_parameters]
        else:
            raise ValueError(
                "parameters must be a list of objects or a name-to-object map."
            )
        parameters = [
            Parameter(
                name=config["name"],
                nominal=float(config.get("nominal", 0.0)),
                bounds=(
                    tuple(map(float, config["bounds"]))
                    if config.get("bounds") is not None
                    else None
                ),
                constrained=bool(
                    config.get("constraint") is not None
                    or config.get("constrained", False)
                ),
                constraint_mean=float(
                    (config.get("constraint") or {}).get("mean", 0.0)
                ),
                constraint_sigma=float(
                    (config.get("constraint") or {}).get("sigma", 1.0)
                ),
            )
            for config in parameter_items
        ]
        components = [
            Component(
                name=sample["name"],
                nominal_yield=float(sample["nominal_yield"]),
                multiplier=sample.get("multiplier", "1"),
                metadata=sample,
            )
            for sample in value["samples"]
        ]
        return cls(components, parameters)

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.components)

    @property
    def nominal_point(self) -> dict[str, float]:
        return {parameter.name: parameter.nominal for parameter in self.parameters}

    def specification(self) -> dict[str, Any]:
        """Return the deterministic scientific definition of this intensity."""

        return {
            "components": [
                {
                    "name": component.name,
                    "nominal_yield": float(component.nominal_yield),
                    "multiplier": component.multiplier.source,
                }
                for component in self.components
            ],
            "parameters": [
                {
                    "name": parameter.name,
                    "nominal": float(parameter.nominal),
                    "bounds": (
                        None
                        if parameter.bounds is None
                        else [*map(float, parameter.bounds)]
                    ),
                    "constrained": parameter.constrained,
                    "constraint_mean": float(parameter.constraint_mean),
                    "constraint_sigma": float(parameter.constraint_sigma),
                }
                for parameter in self.parameters
            ],
        }

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the ordered intensity definition."""

        payload = json.dumps(
            self.specification(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def validate_point(self, point: Mapping[str, float]) -> dict[str, float]:
        known = {parameter.name for parameter in self.parameters}
        missing = known.difference(point)
        if missing:
            raise ValueError(f"Parameter point is missing {sorted(missing)}.")
        unknown = set(point).difference(known)
        if unknown:
            raise ValueError(
                f"Parameter point contains unknown names {sorted(unknown)}."
            )
        result = {name: float(value) for name, value in point.items()}
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Parameter values must be finite.")
        return result

    def component_yields(self, point: Mapping[str, float]) -> dict[str, float]:
        point = self.validate_point(point)
        yields: dict[str, float] = {}
        for component in self.components:
            multiplier = float(component.multiplier.evaluate(point))
            value = float(component.nominal_yield * multiplier)
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"Component {component.name!r} yield is non-finite."
                )
            yields[component.name] = value
        return yields

    def expected_yield(self, point: Mapping[str, float]) -> float:
        return float(sum(self.component_yields(point).values()))

    def differential_over_reference(
        self,
        ratios: Mapping[str, np.ndarray],
        point: Mapping[str, float],
        *,
        normalizer: RatioNormalizer | None = None,
    ) -> np.ndarray:
        missing = set(self.component_names).difference(ratios)
        if missing:
            raise KeyError(f"Missing component ratios {sorted(missing)}.")
        selected = {
            name: np.asarray(ratios[name], dtype=np.float64).reshape(-1)
            for name in self.component_names
        }
        if normalizer is not None:
            selected = normalizer.normalize(selected)
        length = len(next(iter(selected.values())))
        if any(len(values) != length for values in selected.values()):
            raise ValueError("Component ratio arrays must be aligned.")
        yields = self.component_yields(point)
        result = np.zeros(length, dtype=np.float64)
        for name in self.component_names:
            result += yields[name] * selected[name]
        if not np.isfinite(result).all():
            raise FloatingPointError("The differential intensity is non-finite.")
        return result

    def require_nonnegative(
        self,
        ratios: Mapping[str, np.ndarray],
        point: Mapping[str, float],
        *,
        normalizer: RatioNormalizer | None = None,
    ) -> np.ndarray:
        result = self.differential_over_reference(ratios, point, normalizer=normalizer)
        if np.any(result < 0):
            raise ValueError(
                "A probability intensity must be non-negative on all events."
            )
        return result
