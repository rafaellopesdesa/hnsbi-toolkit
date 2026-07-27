"""Factorizable normalizing flows for parameter-dependent density morphing.

This module implements the residual construction of arXiv:2606.30489.  A
normalized nominal density ``p0`` is pulled back through an invertible,
parameter-dependent residual transform,

``p(x | nuisance) = p0(T_nuisance(x)) * |det dT_nuisance / dx|``.

PyTorch is an optional dependency and is imported only when a residual model
is built, evaluated, trained, or serialized.  Configuration, anchor
validation, deterministic grouped splits, and yield morphing therefore remain
available in a base installation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactManifest, write_artifact_manifest
from .onnx import require_optional

FNF_PAPER_URL = "https://arxiv.org/abs/2606.30489"
FNF_REFERENCE_IMPLEMENTATION = (
    "https://github.com/valsdav/factorizable-normalizing-flow"
)


def _torch(*, purpose: str) -> Any:
    return require_optional("torch", extra="flows", purpose=purpose)


def _as_2d(values: Any, *, columns: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim == 1:
        if columns != 1:
            raise ValueError(f"{name} must have shape (n, {columns}).")
        result = result.reshape(-1, 1)
    if result.ndim != 2 or result.shape[1] != columns:
        raise ValueError(f"{name} must have shape (n, {columns}).")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _as_weights(values: Any | None, *, rows: int, name: str) -> np.ndarray:
    result = (
        np.ones(rows, dtype=np.float32)
        if values is None
        else np.asarray(values, dtype=np.float32).reshape(-1)
    )
    if len(result) != rows:
        raise ValueError(f"{name} must contain one value per event.")
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError(f"{name} must be finite and non-negative.")
    if not float(np.sum(result)) > 0:
        raise ValueError(f"{name} must have positive sum.")
    return result


def _canonical_interactions(
    interactions: Sequence[Sequence[str]],
    nuisance_names: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    positions = {name: index for index, name in enumerate(nuisance_names)}
    result: list[tuple[str, str]] = []
    for supplied in interactions:
        pair = tuple(supplied)
        if len(pair) != 2:
            raise ValueError("Every FNF interaction must contain two parameters.")
        first, second = pair
        if first == second:
            raise ValueError("An FNF interaction cannot repeat a parameter.")
        unknown = set(pair).difference(positions)
        if unknown:
            raise ValueError(f"Unknown interaction parameters {sorted(unknown)}.")
        if positions[first] > positions[second]:
            first, second = second, first
        canonical = (first, second)
        if canonical in result:
            raise ValueError(f"Duplicate FNF interaction {canonical}.")
        result.append(canonical)
    return tuple(result)


@dataclass(frozen=True)
class FNFResidualConfig:
    """Architecture and nuisance basis of a factorizable residual flow."""

    n_features: int
    nuisance_names: tuple[str, ...]
    num_layers: int = 2
    hidden_features: tuple[int, ...] = (64, 64)
    interactions: tuple[tuple[str, str], ...] = ()
    nuisance_centers: tuple[float, ...] = ()
    nuisance_scales: tuple[float, ...] = ()
    quadratic_scale: float = 1.0
    interaction_scale: float = 1.0
    log_scale_clip: float = 2.0
    shift_clip: float | None = None

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.nuisance_names)
        hidden = tuple(int(width) for width in self.hidden_features)
        if self.n_features < 1 or self.num_layers < 1:
            raise ValueError("n_features and num_layers must be positive.")
        if not names or any(not name for name in names):
            raise ValueError("nuisance_names must contain non-empty names.")
        if len(names) != len(set(names)):
            raise ValueError("nuisance_names must be unique.")
        if not hidden or any(width < 1 for width in hidden):
            raise ValueError("hidden_features must contain positive widths.")
        centers = (
            tuple(0.0 for _ in names)
            if not self.nuisance_centers
            else tuple(float(value) for value in self.nuisance_centers)
        )
        scales = (
            tuple(1.0 for _ in names)
            if not self.nuisance_scales
            else tuple(float(value) for value in self.nuisance_scales)
        )
        if len(centers) != len(names) or len(scales) != len(names):
            raise ValueError(
                "nuisance_centers and nuisance_scales must match nuisance_names."
            )
        if not np.isfinite(centers).all():
            raise ValueError("nuisance_centers must be finite.")
        if not np.isfinite(scales).all() or any(value <= 0 for value in scales):
            raise ValueError("nuisance_scales must be finite and positive.")
        if self.quadratic_scale <= 0 or self.interaction_scale <= 0:
            raise ValueError("quadratic_scale and interaction_scale must be positive.")
        if self.log_scale_clip <= 0:
            raise ValueError("log_scale_clip must be positive.")
        if self.shift_clip is not None and self.shift_clip <= 0:
            raise ValueError("shift_clip must be positive when provided.")
        object.__setattr__(self, "nuisance_names", names)
        object.__setattr__(self, "hidden_features", hidden)
        object.__setattr__(self, "nuisance_centers", centers)
        object.__setattr__(self, "nuisance_scales", scales)
        object.__setattr__(
            self,
            "interactions",
            _canonical_interactions(self.interactions, names),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["nuisance_names"] = list(self.nuisance_names)
        result["hidden_features"] = list(self.hidden_features)
        result["interactions"] = [list(pair) for pair in self.interactions]
        result["nuisance_centers"] = list(self.nuisance_centers)
        result["nuisance_scales"] = list(self.nuisance_scales)
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FNFResidualConfig:
        return cls(
            n_features=int(payload["n_features"]),
            nuisance_names=tuple(payload["nuisance_names"]),
            num_layers=int(payload.get("num_layers", 2)),
            hidden_features=tuple(payload.get("hidden_features", (64, 64))),
            interactions=tuple(tuple(pair) for pair in payload.get("interactions", ())),
            nuisance_centers=tuple(payload.get("nuisance_centers", ())),
            nuisance_scales=tuple(payload.get("nuisance_scales", ())),
            quadratic_scale=float(payload.get("quadratic_scale", 1.0)),
            interaction_scale=float(payload.get("interaction_scale", 1.0)),
            log_scale_clip=float(payload.get("log_scale_clip", 2.0)),
            shift_clip=(
                None
                if payload.get("shift_clip") is None
                else float(payload["shift_clip"])
            ),
        )


@dataclass(frozen=True)
class FNFStandardizer:
    """Affine feature coordinates used only inside the residual transform."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float64).reshape(-1)
        if mean.size == 0 or mean.shape != scale.shape:
            raise ValueError("mean and scale must be non-empty and aligned.")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("mean and scale must be finite.")
        if np.any(scale <= 0):
            raise ValueError("Every FNF feature scale must be positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def identity(cls, n_features: int) -> FNFStandardizer:
        if n_features < 1:
            raise ValueError("n_features must be positive.")
        return cls(np.zeros(n_features), np.ones(n_features))

    @classmethod
    def fit_equal_anchor(
        cls,
        arrays: Sequence[np.ndarray],
        weights: Sequence[np.ndarray],
        *,
        minimum_scale: float = 1e-6,
    ) -> FNFStandardizer:
        """Fit moments with equal total importance assigned to every anchor."""

        if not arrays or len(arrays) != len(weights):
            raise ValueError("arrays and weights must be non-empty and aligned.")
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive.")
        validated_arrays = []
        normalized = []
        feature_count = None
        for index, (array, weight) in enumerate(zip(arrays, weights, strict=True)):
            values = np.asarray(array, dtype=np.float64)
            event_weights = np.asarray(weight, dtype=np.float64).reshape(-1)
            if values.ndim != 2 or not len(values):
                raise ValueError(f"arrays[{index}] must be non-empty and 2D.")
            if feature_count is None:
                feature_count = values.shape[1]
            if values.shape[1] != feature_count:
                raise ValueError("All anchor arrays must share a feature count.")
            if len(event_weights) != len(values):
                raise ValueError("Every weight array must align with its anchor.")
            if (
                not np.isfinite(values).all()
                or not np.isfinite(event_weights).all()
                or np.any(event_weights < 0)
                or not float(np.sum(event_weights)) > 0
            ):
                raise ValueError(
                    "Anchor values must be finite and weights finite, "
                    "non-negative, and positive in total."
                )
            validated_arrays.append(values)
            normalized.append(event_weights / np.sum(event_weights))
        mean = np.mean(
            [
                np.sum(array * weight[:, None], axis=0)
                for array, weight in zip(validated_arrays, normalized, strict=True)
            ],
            axis=0,
        )
        second = np.mean(
            [
                np.sum(np.square(array) * weight[:, None], axis=0)
                for array, weight in zip(validated_arrays, normalized, strict=True)
            ],
            axis=0,
        )
        variance = np.maximum(second - np.square(mean), minimum_scale**2)
        return cls(mean=mean, scale=np.sqrt(variance))

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FNFStandardizer:
        return cls(mean=payload["mean"], scale=payload["scale"])


@dataclass(frozen=True)
class FNFAnchor:
    """One simulated nuisance anchor used for residual-flow training."""

    values: Any
    point: Mapping[str, float]
    weights: Any | None = None
    groups: Any | None = None
    name: str = ""


@dataclass(frozen=True)
class FNFTrainingConfig:
    """Optimization and leakage-safe split settings."""

    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    holdout_fraction: float = 0.0
    patience: int = 20
    min_delta: float = 1e-5
    gradient_clip_norm: float | None = 5.0
    steps_per_epoch: int | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative."
            )
        if (
            self.validation_fraction < 0
            or self.holdout_fraction < 0
            or self.validation_fraction + self.holdout_fraction >= 1
        ):
            raise ValueError(
                "validation_fraction and holdout_fraction must be non-negative "
                "and sum to less than one."
            )
        if self.patience < 1 or self.min_delta < 0:
            raise ValueError("patience must be positive and min_delta non-negative.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided.")
        if self.steps_per_epoch is not None and self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive when provided.")


@dataclass(frozen=True)
class FNFSplit:
    """Row indices in deterministic, group-disjoint data partitions."""

    training: tuple[np.ndarray, ...]
    validation: tuple[np.ndarray, ...]
    holdout: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class FNFEpoch:
    """One residual-flow optimization epoch."""

    epoch: int
    training_loss: float
    validation_loss: float


@dataclass(frozen=True)
class LogQuadraticYieldMorph:
    """Positive multi-nuisance yield interpolation through three anchors."""

    nominal_yield: float
    nuisance_names: tuple[str, ...]
    linear: tuple[float, ...]
    quadratic: tuple[float, ...]
    centers: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(self.nuisance_names)
        linear = tuple(float(value) for value in self.linear)
        quadratic = tuple(float(value) for value in self.quadratic)
        centers = (
            tuple(0.0 for _ in names)
            if not self.centers
            else tuple(float(value) for value in self.centers)
        )
        scales = (
            tuple(1.0 for _ in names)
            if not self.scales
            else tuple(float(value) for value in self.scales)
        )
        if not math.isfinite(self.nominal_yield) or self.nominal_yield <= 0:
            raise ValueError("nominal_yield must be finite and positive.")
        if not names or len(names) != len(set(names)):
            raise ValueError("nuisance_names must be non-empty and unique.")
        if not (
            len(linear) == len(quadratic) == len(centers) == len(scales) == len(names)
        ):
            raise ValueError("All yield-morph coefficient arrays must align.")
        if not np.isfinite(linear).all() or not np.isfinite(quadratic).all():
            raise ValueError("Yield-morph coefficients must be finite.")
        if not np.isfinite(centers).all():
            raise ValueError("Yield-morph centers must be finite.")
        if not np.isfinite(scales).all() or any(value <= 0 for value in scales):
            raise ValueError("Yield-morph scales must be finite and positive.")
        object.__setattr__(self, "nuisance_names", names)
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadratic", quadratic)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "scales", scales)

    @classmethod
    def from_anchors(
        cls,
        nominal_yield: float,
        anchors: Mapping[str, Sequence[float]],
        *,
        centers: Mapping[str, float] | None = None,
        scales: Mapping[str, float] | None = None,
    ) -> LogQuadraticYieldMorph:
        """Construct from ``parameter: (down, up)`` positive yield anchors."""

        if not anchors:
            raise ValueError("At least one yield anchor pair is required.")
        names = tuple(anchors)
        linear: list[float] = []
        quadratic: list[float] = []
        for name in names:
            pair = tuple(float(value) for value in anchors[name])
            if len(pair) != 2:
                raise ValueError(f"Yield anchors for {name!r} must be (down, up).")
            down, up = pair
            if (
                not math.isfinite(down)
                or not math.isfinite(up)
                or down <= 0
                or up <= 0
                or nominal_yield <= 0
            ):
                raise ValueError("Log-quadratic yield anchors must be positive.")
            log_down = math.log(down / nominal_yield)
            log_up = math.log(up / nominal_yield)
            linear.append(0.5 * (log_up - log_down))
            quadratic.append(0.5 * (log_up + log_down))
        return cls(
            nominal_yield=float(nominal_yield),
            nuisance_names=names,
            linear=tuple(linear),
            quadratic=tuple(quadratic),
            centers=tuple((centers or {}).get(name, 0.0) for name in names),
            scales=tuple((scales or {}).get(name, 1.0) for name in names),
        )

    def expected_yield(self, point: Mapping[str, float] | None = None) -> float:
        supplied = {} if point is None else dict(point)
        unknown = set(supplied).difference(self.nuisance_names)
        if unknown:
            raise ValueError(f"Unknown yield-morph parameters {sorted(unknown)}.")
        exponent = 0.0
        for name, center, scale, linear, quadratic in zip(
            self.nuisance_names,
            self.centers,
            self.scales,
            self.linear,
            self.quadratic,
            strict=True,
        ):
            coordinate = (float(supplied.get(name, center)) - center) / scale
            exponent += linear * coordinate + quadratic * coordinate**2
        return float(self.nominal_yield * math.exp(exponent))

    def factor(self, point: Mapping[str, float] | None = None) -> float:
        return self.expected_yield(point) / self.nominal_yield

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable yield-morph specification."""

        return {
            "nominal_yield": self.nominal_yield,
            "nuisance_names": list(self.nuisance_names),
            "linear": list(self.linear),
            "quadratic": list(self.quadratic),
            "centers": list(self.centers),
            "scales": list(self.scales),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LogQuadraticYieldMorph:
        """Restore a yield morph from :meth:`to_dict` output."""

        return cls(
            nominal_yield=float(payload["nominal_yield"]),
            nuisance_names=tuple(payload["nuisance_names"]),
            linear=tuple(payload["linear"]),
            quadratic=tuple(payload["quadratic"]),
            centers=tuple(payload.get("centers", ())),
            scales=tuple(payload.get("scales", ())),
        )


def _make_mlp(torch: Any, input_width: int, hidden: Sequence[int], output: int) -> Any:
    nn = torch.nn
    layers: list[Any] = []
    width = input_width
    for next_width in hidden:
        layers.extend([nn.Linear(width, next_width), nn.ELU()])
        width = next_width
    final = nn.Linear(width, output)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    layers.append(final)
    return nn.Sequential(*layers)


def _build_residual_module(config: FNFResidualConfig) -> Any:
    torch = _torch(purpose="building a factorizable residual flow")
    nn = torch.nn
    nuisance_positions = {
        name: index for index, name in enumerate(config.nuisance_names)
    }
    pair_positions = tuple(
        (nuisance_positions[first], nuisance_positions[second])
        for first, second in config.interactions
    )

    class AutoregressiveLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.axis_networks = nn.ModuleList(
                nn.ModuleList(
                    nn.ModuleList(
                        _make_mlp(
                            torch,
                            max(feature, 1),
                            config.hidden_features,
                            4,
                        )
                        for _ in config.nuisance_names
                    )
                    for feature in range(config.n_features)
                )
            )
            self.cross_networks = nn.ModuleList(
                nn.ModuleList(
                    _make_mlp(
                        torch,
                        max(feature, 1),
                        config.hidden_features,
                        2,
                    )
                    for _ in config.interactions
                )
                for feature in range(config.n_features)
            )

        @staticmethod
        def _prefix(values: Any, feature: int) -> Any:
            if feature:
                return values[:, :feature]
            return torch.zeros_like(values[:, :1])

        def _affine_parameters(
            self,
            prefix: Any,
            nuisance: Any,
            feature: int,
        ) -> tuple[Any, Any]:
            raw_scale = torch.zeros_like(nuisance[:, 0])
            raw_shift = torch.zeros_like(nuisance[:, 0])
            for index, network in enumerate(self.axis_networks[feature]):
                coefficients = network(prefix)
                coordinate = nuisance[:, index]
                raw_scale = raw_scale + (
                    coordinate * coefficients[:, 0]
                    + config.quadratic_scale * coordinate.square() * coefficients[:, 1]
                )
                raw_shift = raw_shift + (
                    coordinate * coefficients[:, 2]
                    + config.quadratic_scale * coordinate.square() * coefficients[:, 3]
                )
            for pair, network in zip(
                pair_positions,
                self.cross_networks[feature],
                strict=True,
            ):
                coefficients = network(prefix)
                product = nuisance[:, pair[0]] * nuisance[:, pair[1]]
                raw_scale = (
                    raw_scale + config.interaction_scale * product * coefficients[:, 0]
                )
                raw_shift = (
                    raw_shift + config.interaction_scale * product * coefficients[:, 1]
                )
            log_scale = config.log_scale_clip * torch.tanh(
                raw_scale / config.log_scale_clip
            )
            shift = raw_shift
            if config.shift_clip is not None:
                shift = config.shift_clip * torch.tanh(raw_shift / config.shift_clip)
            return log_scale, shift

        def to_nominal(self, values: Any, nuisance: Any) -> tuple[Any, Any]:
            outputs = []
            log_det = torch.zeros_like(values[:, 0])
            for feature in range(config.n_features):
                prefix = self._prefix(values, feature)
                log_scale, shift = self._affine_parameters(prefix, nuisance, feature)
                output = values[:, feature] * torch.exp(log_scale) + shift
                outputs.append(output.unsqueeze(-1))
                log_det = log_det + log_scale
            return torch.cat(outputs, dim=-1), log_det

        def from_nominal(self, values: Any, nuisance: Any) -> tuple[Any, Any]:
            outputs = []
            log_det = torch.zeros_like(values[:, 0])
            for feature in range(config.n_features):
                if outputs:
                    prefix = torch.cat(outputs, dim=-1)
                else:
                    prefix = torch.zeros_like(values[:, :1])
                log_scale, shift = self._affine_parameters(prefix, nuisance, feature)
                output = (values[:, feature] - shift) * torch.exp(-log_scale)
                outputs.append(output.unsqueeze(-1))
                log_det = log_det - log_scale
            return torch.cat(outputs, dim=-1), log_det

    class ResidualStackModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                AutoregressiveLayer() for _ in range(config.num_layers)
            )
            permutations = []
            for index in range(config.num_layers):
                if index % 2 == 0:
                    permutation = torch.arange(config.n_features)
                else:
                    permutation = torch.arange(config.n_features - 1, -1, -1)
                if index > 1 and config.n_features > 2:
                    permutation = torch.roll(permutation, shifts=index // 2)
                permutations.append(permutation)
            permutation_tensor = torch.stack(permutations).to(dtype=torch.int64)
            self.register_buffer("permutations", permutation_tensor)
            self.register_buffer(
                "inverse_permutations",
                torch.argsort(permutation_tensor, dim=1),
            )
            self.register_buffer(
                "nuisance_centers",
                torch.tensor(config.nuisance_centers, dtype=torch.float32),
            )
            self.register_buffer(
                "nuisance_scales",
                torch.tensor(config.nuisance_scales, dtype=torch.float32),
            )

        def _standardized_nuisance(self, nuisance: Any) -> Any:
            if nuisance.ndim == 1:
                nuisance = nuisance.unsqueeze(0)
            return (nuisance - self.nuisance_centers) / self.nuisance_scales

        def to_nominal(self, values: Any, nuisance: Any) -> tuple[Any, Any]:
            nuisance = self._standardized_nuisance(nuisance)
            if nuisance.shape[0] == 1 and values.shape[0] != 1:
                nuisance = nuisance.expand(values.shape[0], -1)
            result = values
            log_det = torch.zeros_like(values[:, 0])
            for index, layer in enumerate(self.layers):
                permutation = self.permutations[index]
                inverse = self.inverse_permutations[index]
                view = torch.index_select(result, 1, permutation)
                view, increment = layer.to_nominal(view, nuisance)
                result = torch.index_select(view, 1, inverse)
                log_det = log_det + increment
            return result, log_det

        def from_nominal(self, values: Any, nuisance: Any) -> tuple[Any, Any]:
            nuisance = self._standardized_nuisance(nuisance)
            if nuisance.shape[0] == 1 and values.shape[0] != 1:
                nuisance = nuisance.expand(values.shape[0], -1)
            result = values
            log_det = torch.zeros_like(values[:, 0])
            for index in range(len(self.layers) - 1, -1, -1):
                permutation = self.permutations[index]
                inverse = self.inverse_permutations[index]
                view = torch.index_select(result, 1, permutation)
                view, increment = self.layers[index].from_nominal(view, nuisance)
                result = torch.index_select(view, 1, inverse)
                log_det = log_det + increment
            return result, log_det

    return ResidualStackModule()


@dataclass(frozen=True)
class FNFArtifact:
    """Verified native and portable residual-flow files."""

    manifest_path: Path
    checkpoint_path: Path
    state_path: Path
    model_path: Path

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        *,
        verify: bool = True,
    ) -> FNFArtifact:
        path = Path(manifest_path)
        manifest = ArtifactManifest.load(path)
        if manifest.artifact_type != "factorizable-residual-flow":
            raise ValueError(
                "Expected a factorizable-residual-flow manifest, found "
                f"{manifest.artifact_type!r}."
            )
        if verify:
            manifest.verify(path.parent)
        records = {record.kind: record for record in manifest.files}
        required = {"pytorch-checkpoint", "portable-state", "model-config"}
        missing = required.difference(records)
        if missing:
            raise ValueError(f"FNF artifact is missing roles {sorted(missing)}.")
        return cls(
            manifest_path=path,
            checkpoint_path=path.parent / records["pytorch-checkpoint"].path,
            state_path=path.parent / records["portable-state"].path,
            model_path=path.parent / records["model-config"].path,
        )


@dataclass
class FactorizableResidualStack:
    """Native exact-identity stack in original feature coordinates."""

    config: FNFResidualConfig
    standardizer: FNFStandardizer | None = None
    module: Any | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.standardizer is None:
            self.standardizer = FNFStandardizer.identity(self.config.n_features)
        if len(self.standardizer.mean) != self.config.n_features:
            raise ValueError("standardizer must match config.n_features.")
        if self.module is None:
            self.module = _build_residual_module(self.config)
        self.to(self.device)

    @property
    def parameters(self) -> tuple[str, ...]:
        return self.config.nuisance_names

    @property
    def nominal_point(self) -> dict[str, float]:
        return dict(
            zip(
                self.config.nuisance_names,
                self.config.nuisance_centers,
                strict=True,
            )
        )

    def to(self, device: str) -> FactorizableResidualStack:
        self.module.to(device)
        self.device = str(device)
        return self

    def _point_array(
        self,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
        *,
        rows: int,
    ) -> np.ndarray:
        if point is None:
            result = np.asarray(self.config.nuisance_centers, dtype=np.float32).reshape(
                1, -1
            )
        elif isinstance(point, Mapping):
            unknown = set(point).difference(self.config.nuisance_names)
            if unknown:
                raise ValueError(f"Unknown FNF parameters {sorted(unknown)}.")
            columns = []
            for name, center in zip(
                self.config.nuisance_names,
                self.config.nuisance_centers,
                strict=True,
            ):
                value = np.asarray(point.get(name, center), dtype=np.float32)
                if value.ndim == 0:
                    value = value.reshape(1)
                if value.ndim != 1 or len(value) not in {1, rows}:
                    raise ValueError(
                        f"Parameter {name!r} must be scalar or have {rows} rows."
                    )
                if len(value) == 1 and rows != 1:
                    value = np.repeat(value, rows)
                columns.append(value)
            result = np.column_stack(columns)
        else:
            result = np.asarray(point, dtype=np.float32)
            if result.ndim == 1:
                result = result.reshape(1, -1)
        expected = len(self.config.nuisance_names)
        if result.ndim != 2 or result.shape[1] != expected:
            raise ValueError(f"point must have shape (n, {expected}).")
        if len(result) == 1 and rows != 1:
            result = np.repeat(result, rows, axis=0)
        if len(result) != rows:
            raise ValueError("point must have one row or one row per event.")
        if not np.isfinite(result).all():
            raise ValueError("point must contain only finite values.")
        return result

    def _tensor_inputs(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> tuple[Any, Any]:
        torch = _torch(purpose="evaluating a factorizable residual flow")
        if torch.is_tensor(values):
            tensor = values.to(device=self.device, dtype=torch.float32)
            if tensor.ndim == 1 and self.config.n_features == 1:
                tensor = tensor.reshape(-1, 1)
            if tensor.ndim != 2 or tensor.shape[1] != self.config.n_features:
                raise ValueError(
                    f"values must have shape (n, {self.config.n_features})."
                )
        else:
            array = _as_2d(
                values,
                columns=self.config.n_features,
                name="values",
            )
            tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        if torch.is_tensor(point):
            nuisance = point.to(device=self.device, dtype=torch.float32)
            if nuisance.ndim == 1:
                nuisance = nuisance.unsqueeze(0)
            expected = len(self.config.nuisance_names)
            if nuisance.ndim != 2 or nuisance.shape[1] != expected:
                raise ValueError(f"point must have shape (n, {expected}).")
            if nuisance.shape[0] == 1 and len(tensor) != 1:
                nuisance = nuisance.expand(len(tensor), -1)
            if nuisance.shape[0] != len(tensor):
                raise ValueError("point must have one row or one row per event.")
        else:
            point_array = self._point_array(point, rows=len(tensor))
            nuisance = torch.as_tensor(
                point_array, dtype=torch.float32, device=self.device
            )
        return tensor, nuisance

    def to_nominal_tensor(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> tuple[Any, Any]:
        torch = _torch(purpose="evaluating a factorizable residual flow")
        tensor, nuisance = self._tensor_inputs(values, point)
        mean = torch.as_tensor(
            self.standardizer.mean, dtype=torch.float32, device=self.device
        )
        scale = torch.as_tensor(
            self.standardizer.scale, dtype=torch.float32, device=self.device
        )
        standardized = (tensor - mean) / scale
        transformed, log_det = self.module.to_nominal(standardized, nuisance)
        return transformed * scale + mean, log_det

    def from_nominal_tensor(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> tuple[Any, Any]:
        torch = _torch(purpose="evaluating a factorizable residual flow")
        tensor, nuisance = self._tensor_inputs(values, point)
        mean = torch.as_tensor(
            self.standardizer.mean, dtype=torch.float32, device=self.device
        )
        scale = torch.as_tensor(
            self.standardizer.scale, dtype=torch.float32, device=self.device
        )
        standardized = (tensor - mean) / scale
        transformed, log_det = self.module.from_nominal(standardized, nuisance)
        return transformed * scale + mean, log_det

    def to_nominal(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = _torch(purpose="evaluating a factorizable residual flow")
        self.module.eval()
        with torch.no_grad():
            result, log_det = self.to_nominal_tensor(values, point)
        return (
            result.detach().cpu().numpy(),
            log_det.detach().cpu().numpy().reshape(-1),
        )

    def from_nominal(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = _torch(purpose="evaluating a factorizable residual flow")
        self.module.eval()
        with torch.no_grad():
            result, log_det = self.from_nominal_tensor(values, point)
        return (
            result.detach().cpu().numpy(),
            log_det.detach().cpu().numpy().reshape(-1),
        )

    def forward(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alias for the observed-to-nominal transform and its log determinant."""

        return self.to_nominal(values, point)

    def inverse(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alias for the nominal-to-observed transform and its log determinant."""

        return self.from_nominal(values, point)

    def jacobian_logdet_error(
        self,
        values: Any,
        point: Mapping[str, float],
        *,
        max_rows: int = 4,
    ) -> float:
        """Compare analytic and autograd Jacobian determinants."""

        torch = _torch(purpose="checking an FNF Jacobian")
        array = _as_2d(values, columns=self.config.n_features, name="values")[:max_rows]
        errors = []
        self.module.eval()
        for row in array:
            tensor = torch.tensor(
                row, dtype=torch.float32, device=self.device, requires_grad=True
            )

            def transform(single: Any) -> Any:
                transformed, _ = self.to_nominal_tensor(single.unsqueeze(0), point)
                return transformed.squeeze(0)

            jacobian = torch.autograd.functional.jacobian(transform, tensor)
            _, numerical = torch.linalg.slogdet(jacobian)
            _, analytic = self.to_nominal_tensor(tensor.unsqueeze(0), point)
            errors.append(float(torch.abs(numerical - analytic[0]).detach().cpu()))
        return max(errors, default=0.0)

    def save(
        self,
        directory: str | Path,
        *,
        prefix: str = "fnf",
        metadata: Mapping[str, Any] | None = None,
    ) -> FNFArtifact:
        """Write native and backend-neutral states with a checksum manifest."""

        torch = _torch(purpose="saving a factorizable residual flow")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = root / f"{prefix}.pt"
        state_path = root / f"{prefix}.state.npz"
        model_path = root / f"{prefix}.json"
        manifest_path = root / f"{prefix}.manifest.json"
        state = {
            name: value.detach().cpu()
            for name, value in self.module.state_dict().items()
        }
        torch.save({"state_dict": state}, checkpoint_path)
        np.savez_compressed(
            state_path,
            **{name: value.numpy() for name, value in state.items()},
        )
        payload = {
            "artifact_format": 1,
            "config": self.config.to_dict(),
            "metadata": dict(metadata or {}),
            "paper": FNF_PAPER_URL,
            "reference_implementation": FNF_REFERENCE_IMPLEMENTATION,
            "standardizer": self.standardizer.to_dict(),
        }
        model_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_artifact_manifest(
            manifest_path,
            artifact_type="factorizable-residual-flow",
            files={
                "pytorch-checkpoint": checkpoint_path,
                "portable-state": state_path,
                "model-config": model_path,
            },
            metadata={
                "n_features": self.config.n_features,
                "nuisance_names": list(self.config.nuisance_names),
                "interactions": [list(pair) for pair in self.config.interactions],
                **dict(metadata or {}),
            },
        )
        return FNFArtifact.load(manifest_path)

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        *,
        device: str = "cpu",
        verify: bool = True,
    ) -> FactorizableResidualStack:
        """Load from the portable NumPy state after manifest verification."""

        torch = _torch(purpose="loading a factorizable residual flow")
        artifact = FNFArtifact.load(manifest_path, verify=verify)
        payload = json.loads(artifact.model_path.read_text(encoding="utf-8"))
        if payload.get("artifact_format") != 1:
            raise ValueError("Unsupported FNF artifact format.")
        config = FNFResidualConfig.from_dict(payload["config"])
        result = cls(
            config=config,
            standardizer=FNFStandardizer.from_dict(payload["standardizer"]),
            device=device,
        )
        with np.load(artifact.state_path, allow_pickle=False) as portable:
            state = {
                name: torch.as_tensor(
                    portable[name], dtype=result.module.state_dict()[name].dtype
                )
                for name in portable.files
            }
        expected = set(result.module.state_dict())
        if set(state) != expected:
            raise ValueError("Portable FNF state does not match its architecture.")
        structural = {
            "nuisance_centers",
            "nuisance_scales",
            "permutations",
            "inverse_permutations",
        }
        expected_state = result.module.state_dict()
        for name in structural:
            if not torch.equal(state[name], expected_state[name].cpu()):
                raise ValueError(
                    f"Portable FNF structural state {name!r} conflicts "
                    "with model metadata."
                )
        result.module.load_state_dict(state, strict=True)
        return result


@dataclass
class FactorizableDensity:
    """A normalized nominal density transformed by an FNF residual stack."""

    residual: FactorizableResidualStack
    base_density: Any
    base_torch_log_prob: Callable[[Any], Any] | None = None
    base_sampler: Callable[..., Any] | None = None

    @property
    def parameters(self) -> tuple[str, ...]:
        return self.residual.parameters

    @property
    def nominal_point(self) -> dict[str, float]:
        return self.residual.nominal_point

    def to_nominal(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> np.ndarray:
        return self.residual.to_nominal(values, point)[0]

    def from_nominal(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> np.ndarray:
        return self.residual.from_nominal(values, point)[0]

    def forward(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> np.ndarray:
        """Map varied observations to nominal coordinates."""

        return self.to_nominal(values, point)

    def inverse(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
    ) -> np.ndarray:
        """Map nominal coordinates to varied observations."""

        return self.from_nominal(values, point)

    def log_prob(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        transformed, log_det = self.residual.to_nominal(values, point)
        evaluator = getattr(self.base_density, "log_prob", self.base_density)
        base = np.asarray(evaluator(transformed), dtype=np.float64).reshape(-1)
        if len(base) != len(transformed) or not np.isfinite(base).all():
            raise ValueError("base_density.log_prob returned invalid values.")
        return base + log_det

    def torch_log_prob(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None = None,
    ) -> Any:
        if self.base_torch_log_prob is None:
            raise RuntimeError(
                "A differentiable base_torch_log_prob callable is required."
            )
        transformed, log_det = self.residual.to_nominal_tensor(values, point)
        return self.base_torch_log_prob(transformed) + log_det

    def log_ratio(
        self,
        values: Any,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None,
        *,
        reference_point: (
            Mapping[str, Any] | Sequence[float] | np.ndarray | None
        ) = None,
    ) -> np.ndarray:
        """Return ``log p(x|point) - log p(x|reference_point)``."""

        reference = self.nominal_point if reference_point is None else reference_point
        return self.log_prob(values, point) - self.log_prob(values, reference)

    def sample(
        self,
        n: int,
        point: Mapping[str, Any] | Sequence[float] | np.ndarray | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if int(n) < 1:
            raise ValueError("n must be positive.")
        sampler = self.base_sampler or getattr(self.base_density, "sample", None)
        if sampler is None:
            raise RuntimeError("The nominal density does not provide a sampler.")
        generator = np.random.default_rng() if rng is None else rng
        try:
            nominal = sampler(int(n), rng=generator)
        except TypeError:
            nominal = sampler(int(n))
        nominal = _as_2d(
            nominal,
            columns=self.residual.config.n_features,
            name="nominal samples",
        )
        return self.from_nominal(nominal, point)


@dataclass(frozen=True)
class FNFPointDiagnostic:
    """Numerical closure metrics at one nuisance point."""

    point: dict[str, float]
    round_trip_max_abs: float
    logdet_cancellation_max_abs: float
    normalization: float
    normalization_standard_error: float
    importance_ess: float
    jacobian_logdet_max_abs: float | None = None


@dataclass(frozen=True)
class FNFDiagnosticReport:
    """Identity, invertibility, normalization, and Jacobian diagnostics."""

    identity_max_abs: float
    identity_logdet_max_abs: float
    points: tuple[FNFPointDiagnostic, ...]


def diagnose_fnf(
    density: FactorizableDensity,
    nominal_samples: Any,
    points: Sequence[Mapping[str, float]],
    *,
    check_jacobian: bool = False,
    jacobian_rows: int = 4,
) -> FNFDiagnosticReport:
    """Evaluate structural FNF closure without empirical renormalization."""

    values = _as_2d(
        nominal_samples,
        columns=density.residual.config.n_features,
        name="nominal_samples",
    )
    nominal = density.nominal_point
    identity, identity_log_det = density.residual.to_nominal(values, nominal)
    reports = []
    nominal_log_prob = density.log_prob(values, nominal)
    for supplied in points:
        point = {name: float(value) for name, value in supplied.items()}
        transformed, forward_log_det = density.residual.to_nominal(values, point)
        reconstructed, inverse_log_det = density.residual.from_nominal(
            transformed, point
        )
        log_ratio = density.log_prob(values, point) - nominal_log_prob
        ratios = np.exp(np.clip(log_ratio, -700.0, 700.0))
        normalization = float(np.mean(ratios))
        standard_error = (
            float(np.std(ratios, ddof=1) / math.sqrt(len(ratios)))
            if len(ratios) > 1
            else 0.0
        )
        denominator = float(np.sum(np.square(ratios)))
        ess = float(np.sum(ratios) ** 2 / denominator) if denominator > 0 else 0.0
        reports.append(
            FNFPointDiagnostic(
                point=point,
                round_trip_max_abs=float(np.max(np.abs(reconstructed - values))),
                logdet_cancellation_max_abs=float(
                    np.max(np.abs(forward_log_det + inverse_log_det))
                ),
                normalization=normalization,
                normalization_standard_error=standard_error,
                importance_ess=ess,
                jacobian_logdet_max_abs=(
                    density.residual.jacobian_logdet_error(
                        values, point, max_rows=jacobian_rows
                    )
                    if check_jacobian
                    else None
                ),
            )
        )
    return FNFDiagnosticReport(
        identity_max_abs=float(np.max(np.abs(identity - values))),
        identity_logdet_max_abs=float(np.max(np.abs(identity_log_det))),
        points=tuple(reports),
    )


@dataclass
class FNFTrainingResult:
    """A fitted residual stack, optimization history, and deterministic split."""

    residual: FactorizableResidualStack
    history: tuple[FNFEpoch, ...]
    split: FNFSplit
    best_epoch: int
    stopped_early: bool
    features: tuple[str, ...]

    def density(
        self,
        base_density: Any,
        *,
        base_torch_log_prob: Callable[[Any], Any] | None = None,
        base_sampler: Callable[..., Any] | None = None,
    ) -> FactorizableDensity:
        return FactorizableDensity(
            residual=self.residual,
            base_density=base_density,
            base_torch_log_prob=base_torch_log_prob,
            base_sampler=base_sampler,
        )

    def save(
        self,
        directory: str | Path,
        *,
        prefix: str = "fnf",
        metadata: Mapping[str, Any] | None = None,
    ) -> FNFArtifact:
        result_metadata = {
            "best_epoch": self.best_epoch,
            "features": list(self.features),
            "history": [asdict(epoch) for epoch in self.history],
            "stopped_early": self.stopped_early,
            **dict(metadata or {}),
        }
        return self.residual.save(directory, prefix=prefix, metadata=result_metadata)


@dataclass
class _PreparedAnchor:
    name: str
    values: np.ndarray
    point: dict[str, float]
    point_array: np.ndarray
    weights: np.ndarray
    group_keys: tuple[str, ...]


class FNFTrainer:
    """Fit an FNF with equal-anchor weighted maximum likelihood."""

    def __init__(
        self,
        residual_config: FNFResidualConfig,
        training_config: FNFTrainingConfig | None = None,
    ) -> None:
        self.residual_config = residual_config
        self.training_config = training_config or FNFTrainingConfig()

    @staticmethod
    def _stable_group_key(value: Any) -> str:
        if isinstance(value, np.generic):
            value = value.item()
        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except TypeError:
            encoded = repr(value)
        return encoded

    def _prepare_anchors(self, anchors: Sequence[FNFAnchor]) -> list[_PreparedAnchor]:
        if not anchors:
            raise ValueError("At least one FNF training anchor is required.")
        names = self.residual_config.nuisance_names
        centers = dict(
            zip(
                names,
                self.residual_config.nuisance_centers,
                strict=True,
            )
        )
        result = []
        for index, anchor in enumerate(anchors):
            name = anchor.name or f"anchor-{index}"
            values = _as_2d(
                anchor.values,
                columns=self.residual_config.n_features,
                name=f"{name}.values",
            )
            if len(values) < 1:
                raise ValueError(f"{name}.values cannot be empty.")
            weights = _as_weights(
                anchor.weights, rows=len(values), name=f"{name}.weights"
            )
            unknown = set(anchor.point).difference(names)
            if unknown:
                raise ValueError(
                    f"{name} contains unknown parameters {sorted(unknown)}."
                )
            point = {
                parameter: float(anchor.point.get(parameter, centers[parameter]))
                for parameter in names
            }
            if not np.isfinite(tuple(point.values())).all():
                raise ValueError(f"{name}.point must contain finite values.")
            if anchor.groups is None:
                group_keys = tuple(
                    f"implicit:{index}:{row}" for row in range(len(values))
                )
            else:
                groups = np.asarray(anchor.groups, dtype=object).reshape(-1)
                if len(groups) != len(values):
                    raise ValueError(f"{name}.groups must contain one value per event.")
                group_keys = tuple(
                    "provided:" + self._stable_group_key(value) for value in groups
                )
            result.append(
                _PreparedAnchor(
                    name=name,
                    values=values,
                    point=point,
                    point_array=np.asarray(
                        [point[parameter] for parameter in names],
                        dtype=np.float32,
                    ),
                    weights=weights,
                    group_keys=group_keys,
                )
            )
        nonnominal = []
        centers_array = np.asarray(
            self.residual_config.nuisance_centers, dtype=np.float32
        )
        for anchor in result:
            nonnominal.append(
                ~np.isclose(anchor.point_array, centers_array, atol=1e-12)
            )
        active = np.stack(nonnominal)
        if not np.all(np.any(active, axis=0)):
            missing = [
                name
                for name, covered in zip(names, np.any(active, axis=0), strict=True)
                if not covered
            ]
            raise ValueError(
                f"No non-nominal training anchor for parameters {missing}."
            )
        positions = {name: index for index, name in enumerate(names)}
        for first, second in self.residual_config.interactions:
            first_active = active[:, positions[first]]
            second_active = active[:, positions[second]]
            if not np.any(first_active & second_active):
                raise ValueError(
                    f"Interaction {(first, second)} requires a joint anchor "
                    "where both parameters are non-nominal."
                )
        return result

    def validate_anchors(self, anchors: Sequence[FNFAnchor]) -> None:
        """Validate feature, weight, nuisance, and interaction coverage."""

        self._prepare_anchors(anchors)

    def split(self, anchors: Sequence[FNFAnchor], *, seed: int = 0) -> FNFSplit:
        """Return a deterministic split that never separates equal group IDs."""

        prepared = self._prepare_anchors(anchors)
        all_groups = sorted(
            {group for anchor in prepared for group in anchor.group_keys},
            key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest(),
        )
        requested_partitions = (
            1
            + int(self.training_config.validation_fraction > 0)
            + int(self.training_config.holdout_fraction > 0)
        )
        if len(all_groups) < requested_partitions:
            raise ValueError(
                f"At least {requested_partitions} groups are required for the "
                "requested training, validation, and holdout partitions."
            )
        holdout_count = (
            max(
                1,
                int(round(self.training_config.holdout_fraction * len(all_groups))),
            )
            if self.training_config.holdout_fraction
            else 0
        )
        validation_count = (
            max(
                1,
                int(round(self.training_config.validation_fraction * len(all_groups))),
            )
            if self.training_config.validation_fraction
            else 0
        )
        while holdout_count + validation_count >= len(all_groups):
            if validation_count > 1:
                validation_count -= 1
            elif holdout_count > 1:
                holdout_count -= 1
            else:
                raise ValueError(
                    "Split fractions leave no training group; reduce a split "
                    "fraction or provide more groups."
                )
        holdout_groups = set(all_groups[:holdout_count])
        validation_groups = set(
            all_groups[holdout_count : holdout_count + validation_count]
        )
        training = []
        validation = []
        holdout = []
        for anchor in prepared:
            training.append(
                np.asarray(
                    [
                        index
                        for index, group in enumerate(anchor.group_keys)
                        if group not in validation_groups
                        and group not in holdout_groups
                    ],
                    dtype=np.int64,
                )
            )
            validation.append(
                np.asarray(
                    [
                        index
                        for index, group in enumerate(anchor.group_keys)
                        if group in validation_groups
                    ],
                    dtype=np.int64,
                )
            )
            holdout.append(
                np.asarray(
                    [
                        index
                        for index, group in enumerate(anchor.group_keys)
                        if group in holdout_groups
                    ],
                    dtype=np.int64,
                )
            )
        if any(len(indices) == 0 for indices in training):
            raise ValueError(
                "The grouped split leaves an anchor without training rows; "
                "provide more independent groups or smaller split fractions."
            )
        return FNFSplit(
            training=tuple(training),
            validation=tuple(validation),
            holdout=tuple(holdout),
        )

    @staticmethod
    def equal_anchor_loss(
        log_probabilities: Sequence[Any],
        weights: Sequence[Any],
    ) -> Any:
        """Mean of per-anchor normalized weighted negative log likelihoods."""

        if not log_probabilities or len(log_probabilities) != len(weights):
            raise ValueError(
                "log_probabilities and weights must be non-empty and aligned."
            )
        losses = []
        for log_prob, weight in zip(log_probabilities, weights, strict=True):
            if len(log_prob) != len(weight):
                raise ValueError("Every log-probability array must align with weights.")
            losses.append(-(log_prob * weight).sum() / weight.sum())
        return sum(losses) / len(losses)

    def fit(
        self,
        anchors: Sequence[FNFAnchor],
        *,
        base_torch_log_prob: Callable[[Any], Any],
        features: Sequence[str],
        standardizer: FNFStandardizer | None = None,
        seed: int = 0,
    ) -> FNFTrainingResult:
        """Fit the residual while keeping the supplied nominal density frozen.

        ``base_torch_log_prob`` must accept a two-dimensional Torch tensor and
        return one differentiable nominal log density per row.  Parameters of
        the nominal density should have ``requires_grad=False``; gradients with
        respect to its input must remain enabled.
        """

        torch = _torch(purpose="training a factorizable residual flow")
        prepared = self._prepare_anchors(anchors)
        feature_names = tuple(features)
        if len(feature_names) != self.residual_config.n_features:
            raise ValueError("features must match residual_config.n_features.")
        split = self.split(anchors, seed=seed)
        if standardizer is None:
            standardizer = FNFStandardizer.fit_equal_anchor(
                [
                    anchor.values[indices]
                    for anchor, indices in zip(prepared, split.training, strict=True)
                ],
                [
                    anchor.weights[indices]
                    for anchor, indices in zip(prepared, split.training, strict=True)
                ],
            )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        residual = FactorizableResidualStack(
            self.residual_config,
            standardizer=standardizer,
            device=self.training_config.device,
        )
        optimizer = torch.optim.AdamW(
            residual.module.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        rng = np.random.default_rng(seed)
        device = self.training_config.device

        def batches(
            anchor: _PreparedAnchor,
            indices: np.ndarray,
            *,
            shuffled: bool,
        ) -> list[tuple[Any, Any, Any]]:
            selected = rng.permutation(indices) if shuffled else indices
            result = []
            for start in range(0, len(selected), self.training_config.batch_size):
                batch = selected[start : start + self.training_config.batch_size]
                if not len(batch):
                    continue
                values = torch.as_tensor(
                    anchor.values[batch], dtype=torch.float32, device=device
                )
                point = torch.as_tensor(
                    np.repeat(anchor.point_array[None, :], len(batch), axis=0),
                    dtype=torch.float32,
                    device=device,
                )
                weights = torch.as_tensor(
                    anchor.weights[batch], dtype=torch.float32, device=device
                )
                result.append((values, point, weights))
            return result

        def evaluate(partitions: Sequence[np.ndarray]) -> float:
            losses = []
            residual.module.eval()
            for anchor, indices in zip(prepared, partitions, strict=True):
                if not len(indices):
                    continue
                numerator = 0.0
                denominator = 0.0
                for values, point, weight in batches(anchor, indices, shuffled=False):
                    with torch.no_grad():
                        transformed, log_det = residual.to_nominal_tensor(values, point)
                        log_prob = base_torch_log_prob(transformed) + log_det
                    numerator += float((log_prob * weight).sum().detach().cpu())
                    denominator += float(weight.sum().detach().cpu())
                losses.append(-numerator / denominator)
            if not losses:
                return math.nan
            return float(np.mean(losses))

        history = []
        best_loss = math.inf
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        stopped_early = False
        for epoch in range(1, self.training_config.epochs + 1):
            residual.module.train()
            epoch_batches = [
                batches(anchor, indices, shuffled=True)
                for anchor, indices in zip(
                    prepared,
                    split.training,
                    strict=True,
                )
            ]
            natural_steps = max(len(items) for items in epoch_batches)
            steps = (
                natural_steps
                if self.training_config.steps_per_epoch is None
                else self.training_config.steps_per_epoch
            )
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for items in epoch_batches:
                    values, point, weight = items[step % len(items)]
                    transformed, log_det = residual.to_nominal_tensor(values, point)
                    log_prob = base_torch_log_prob(transformed) + log_det
                    losses.append(-(log_prob * weight).sum() / weight.sum())
                loss = sum(losses) / len(losses)
                loss.backward()
                if self.training_config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        residual.module.parameters(),
                        self.training_config.gradient_clip_norm,
                    )
                optimizer.step()
            training_loss = evaluate(split.training)
            validation_loss = evaluate(split.validation)
            if not math.isfinite(validation_loss):
                validation_loss = training_loss
            history.append(
                FNFEpoch(
                    epoch=epoch,
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                )
            )
            if math.isfinite(validation_loss) and (
                validation_loss < best_loss - self.training_config.min_delta
            ):
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in residual.module.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.training_config.patience:
                    stopped_early = True
                    break
        if best_state is None:
            raise RuntimeError("FNF training did not produce a finite checkpoint.")
        residual.module.load_state_dict(best_state)
        return FNFTrainingResult(
            residual=residual,
            history=tuple(history),
            split=split,
            best_epoch=best_epoch,
            stopped_early=stopped_early,
            features=feature_names,
        )


__all__ = [
    "FNFAnchor",
    "FNFArtifact",
    "FNFDiagnosticReport",
    "FNFEpoch",
    "FNFPointDiagnostic",
    "FNFResidualConfig",
    "FNFSplit",
    "FNFStandardizer",
    "FNFTrainer",
    "FNFTrainingConfig",
    "FNFTrainingResult",
    "FactorizableDensity",
    "FactorizableResidualStack",
    "LogQuadraticYieldMorph",
    "diagnose_fnf",
]
