"""Trainable reference normalizing flows with portable ONNX inference.

PyTorch is optional and imported only when a flow is built, trained, loaded, or
exported, so configuration and artifact inspection work in the base
installation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactManifest, write_artifact_manifest
from .onnx import (
    OnnxParityReport,
    OnnxRunner,
    compare_outputs,
    export_torch_onnx,
    manifest_path_for,
    require_optional,
)

_FLOW_ALIASES = {
    "realnvp": "realnvp",
    "real-nvp": "realnvp",
    "affine": "realnvp",
    "quadratic-spline": "quadratic-spline",
    "quadratic_spline": "quadratic-spline",
    "rqs": "quadratic-spline",
    "spline": "quadratic-spline",
}


def _as_2d(values: Any, *, columns: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        if columns != 1:
            raise ValueError(f"{name} must have shape (n, {columns}).")
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (n, {columns}).")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


@dataclass(frozen=True)
class FlowConfig:
    """Architecture of a normalized reference density."""

    n_features: int
    flow_type: str = "realnvp"
    num_transforms: int = 6
    hidden_features: int = 64
    num_blocks: int = 2
    num_bins: int = 8
    tail_bound: float = 4.0
    dropout_probability: float = 0.0
    context_features: int | None = None
    max_log_scale: float = 2.0

    def __post_init__(self) -> None:
        normalized = _FLOW_ALIASES.get(self.flow_type.lower())
        if normalized is None:
            raise ValueError(
                f"Unknown flow_type {self.flow_type!r}; choose 'realnvp' "
                "or 'quadratic-spline'."
            )
        object.__setattr__(self, "flow_type", normalized)
        for name in (
            "n_features",
            "num_transforms",
            "hidden_features",
            "num_blocks",
            "num_bins",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.context_features is not None and self.context_features < 1:
            raise ValueError("context_features must be positive when provided.")
        if self.flow_type == "realnvp" and self.context_features is not None:
            raise ValueError(
                "Conditional context is currently supported by "
                "flow_type='quadratic-spline'; use that architecture for "
                "conditional density estimation."
            )
        if self.num_bins < 2:
            raise ValueError("num_bins must be at least two.")
        if self.tail_bound <= 0:
            raise ValueError("tail_bound must be positive.")
        if not 0 <= self.dropout_probability < 1:
            raise ValueError("dropout_probability must lie in [0, 1).")
        if self.max_log_scale <= 0:
            raise ValueError("max_log_scale must be positive.")


@dataclass(frozen=True)
class FlowTrainingConfig:
    """Weighted maximum-likelihood optimization settings."""

    epochs: int = 200
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    patience: int = 20
    min_delta: float = 1e-5
    learning_rate_factor: float = 0.5
    gradient_clip_norm: float | None = 5.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative."
            )
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie in [0, 1).")
        if self.patience < 1:
            raise ValueError("patience must be positive.")
        if self.min_delta < 0:
            raise ValueError("min_delta cannot be negative.")
        if not 0.0 < self.learning_rate_factor < 1.0:
            raise ValueError("learning_rate_factor must lie in (0, 1).")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive when provided.")


@dataclass(frozen=True)
class AffineStandardizer:
    """Per-feature affine transform fitted with non-negative event weights."""

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(self.scale, dtype=np.float64).reshape(-1)
        if mean.shape != scale.shape or mean.size == 0:
            raise ValueError("mean and scale must be non-empty and aligned.")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("mean and scale must be finite.")
        if np.any(scale <= 0):
            raise ValueError("Every standardization scale must be positive.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(
        cls,
        values: Any,
        *,
        weights: Any | None = None,
        minimum_scale: float = 1e-6,
    ) -> AffineStandardizer:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or len(array) == 0:
            raise ValueError("values must be a non-empty two-dimensional array.")
        if not np.isfinite(array).all():
            raise ValueError("values must contain only finite entries.")
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive.")
        if weights is None:
            event_weights = np.ones(len(array), dtype=np.float64)
        else:
            event_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
            if len(event_weights) != len(array):
                raise ValueError("weights must contain one value per event.")
            if not np.isfinite(event_weights).all() or np.any(event_weights < 0):
                raise ValueError("weights must be finite and non-negative.")
        total = float(np.sum(event_weights))
        if not total > 0:
            raise ValueError("weights must have positive sum.")
        mean = np.sum(array * event_weights[:, None], axis=0) / total
        variance = (
            np.sum(event_weights[:, None] * np.square(array - mean), axis=0) / total
        )
        scale = np.sqrt(np.maximum(variance, minimum_scale**2))
        return cls(mean=mean, scale=scale)

    @property
    def n_features(self) -> int:
        return len(self.mean)

    @property
    def forward_log_abs_det(self) -> float:
        """Log determinant of ``x -> (x - mean) / scale``."""

        return float(-np.sum(np.log(self.scale)))

    def transform(self, values: Any) -> np.ndarray:
        array = _as_2d(values, columns=self.n_features, name="values")
        return ((array - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: Any) -> np.ndarray:
        array = _as_2d(values, columns=self.n_features, name="values")
        return (array * self.scale + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AffineStandardizer:
        return cls(mean=payload["mean"], scale=payload["scale"])


def _build_realnvp(config: FlowConfig) -> Any:
    torch = require_optional("torch", extra="flows", purpose="building a flow")
    nn = torch.nn

    class AffineCoupling(nn.Module):
        def __init__(self, mask: Any) -> None:
            super().__init__()
            self.register_buffer("mask", mask)
            layers: list[Any] = []
            width = config.n_features
            for _ in range(config.num_blocks):
                layers.extend(
                    [
                        nn.Linear(width, config.hidden_features),
                        nn.SiLU(),
                    ]
                )
                if config.dropout_probability:
                    layers.append(nn.Dropout(config.dropout_probability))
                width = config.hidden_features
            final = nn.Linear(width, 2 * config.n_features)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
            self.network = nn.Sequential(*layers)

        def _coupling_parameters(self, conditioned: Any) -> tuple[Any, Any]:
            shift, raw_log_scale = self.network(conditioned).chunk(2, dim=-1)
            transformed = 1.0 - self.mask
            shift = shift * transformed
            log_scale = torch.tanh(raw_log_scale) * config.max_log_scale * transformed
            return shift, log_scale

        def forward(self, values: Any) -> tuple[Any, Any]:
            conditioned = values * self.mask
            shift, log_scale = self._coupling_parameters(conditioned)
            transformed = 1.0 - self.mask
            result = conditioned + transformed * (values * torch.exp(log_scale) + shift)
            return result, log_scale.sum(dim=-1)

        def inverse(self, values: Any) -> tuple[Any, Any]:
            conditioned = values * self.mask
            shift, log_scale = self._coupling_parameters(conditioned)
            transformed = 1.0 - self.mask
            result = conditioned + transformed * (
                (values - shift) * torch.exp(-log_scale)
            )
            return result, -log_scale.sum(dim=-1)

    class RealNVP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            couplings = []
            for index in range(config.num_transforms):
                mask = torch.tensor(
                    [
                        float((feature + index) % 2)
                        for feature in range(config.n_features)
                    ],
                    dtype=torch.float32,
                )
                couplings.append(AffineCoupling(mask))
            self.couplings = nn.ModuleList(couplings)

        def data_to_base(self, values: Any) -> tuple[Any, Any]:
            result = values
            log_det = values.new_zeros(values.shape[0])
            for coupling in self.couplings:
                result, increment = coupling(result)
                log_det = log_det + increment
            return result, log_det

        def base_to_data(self, values: Any) -> tuple[Any, Any]:
            result = values
            log_det = values.new_zeros(values.shape[0])
            for coupling in reversed(self.couplings):
                result, increment = coupling.inverse(result)
                log_det = log_det + increment
            return result, log_det

        def log_prob(self, values: Any) -> Any:
            base, log_det = self.data_to_base(values)
            base_log_prob = -0.5 * (base.square() + math.log(2.0 * math.pi)).sum(dim=-1)
            return base_log_prob + log_det

    return RealNVP()


def _build_quadratic_spline(config: FlowConfig) -> Any:
    """Build an ONNX-portable autoregressive rational-quadratic spline.

    ``nflows`` is excellent for native training, but its spline inverse relies
    on indexing patterns that PyTorch's ONNX exporter specializes to the
    example batch.  The implementation below keeps the same mathematical
    transform while expressing bin selection with comparisons, reductions,
    and ``gather``.  Those operations retain a genuinely dynamic batch axis in
    ONNX Runtime.
    """

    torch = require_optional("torch", extra="flows", purpose="building a spline flow")
    nn = torch.nn
    functional = torch.nn.functional
    minimum_width = min(1e-3, 0.5 / config.num_bins)
    minimum_height = min(1e-3, 0.5 / config.num_bins)
    minimum_derivative = 1e-3
    parameter_count = 3 * config.num_bins - 1
    identity_derivative_bias = math.log(math.expm1(1.0 - minimum_derivative))

    def normalize_parameters(
        raw: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        raw_widths = raw[..., : config.num_bins]
        raw_heights = raw[..., config.num_bins : 2 * config.num_bins]
        raw_derivatives = raw[..., 2 * config.num_bins :]
        widths = minimum_width + (
            1.0 - minimum_width * config.num_bins
        ) * torch.softmax(raw_widths, dim=-1)
        heights = minimum_height + (
            1.0 - minimum_height * config.num_bins
        ) * torch.softmax(raw_heights, dim=-1)
        internal_derivatives = minimum_derivative + functional.softplus(raw_derivatives)
        boundary = torch.ones_like(internal_derivatives[..., :1])
        derivatives = torch.cat((boundary, internal_derivatives, boundary), dim=-1)
        zero_width = torch.zeros_like(widths[..., :1])
        zero_height = torch.zeros_like(heights[..., :1])
        cumulative_widths = torch.cat(
            (zero_width, torch.cumsum(widths, dim=-1)), dim=-1
        )
        cumulative_heights = torch.cat(
            (zero_height, torch.cumsum(heights, dim=-1)), dim=-1
        )
        span = 2.0 * config.tail_bound
        cumulative_widths = cumulative_widths * span - config.tail_bound
        cumulative_heights = cumulative_heights * span - config.tail_bound
        widths = widths * span
        heights = heights * span
        return (
            cumulative_widths,
            cumulative_heights,
            widths,
            heights,
            derivatives,
        )

    def select(values: Any, indices: Any) -> Any:
        return torch.gather(values, -1, indices.unsqueeze(-1)).squeeze(-1)

    def spline(inputs: Any, raw_parameters: Any, *, inverse: bool) -> tuple[Any, Any]:
        (
            cumulative_widths,
            cumulative_heights,
            widths,
            heights,
            derivatives,
        ) = normalize_parameters(raw_parameters)
        knot_locations = cumulative_heights if inverse else cumulative_widths
        bounded = torch.clamp(inputs, min=-config.tail_bound, max=config.tail_bound)
        bin_indices = torch.sum(
            (bounded.unsqueeze(-1) >= knot_locations[..., 1:-1]).to(dtype=torch.int64),
            dim=-1,
        )
        bin_indices = torch.clamp(bin_indices, min=0, max=config.num_bins - 1)
        input_cumulative_widths = select(cumulative_widths, bin_indices)
        input_cumulative_heights = select(cumulative_heights, bin_indices)
        input_widths = select(widths, bin_indices)
        input_heights = select(heights, bin_indices)
        input_delta = input_heights / input_widths
        input_derivatives = select(derivatives, bin_indices)
        input_derivatives_plus_one = select(derivatives, bin_indices + 1)
        derivative_difference = (
            input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
        )

        if inverse:
            shifted = bounded - input_cumulative_heights
            quadratic_a = shifted * derivative_difference + input_heights * (
                input_delta - input_derivatives
            )
            quadratic_b = (
                input_heights * input_derivatives - shifted * derivative_difference
            )
            quadratic_c = -input_delta * shifted
            discriminant = torch.clamp(
                quadratic_b.square() - 4.0 * quadratic_a * quadratic_c,
                min=0.0,
            )
            theta = (2.0 * quadratic_c) / (-quadratic_b - torch.sqrt(discriminant))
        else:
            theta = (bounded - input_cumulative_widths) / input_widths

        theta_one_minus_theta = theta * (1.0 - theta)
        denominator = input_delta + derivative_difference * theta_one_minus_theta
        derivative_numerator = input_delta.square() * (
            input_derivatives_plus_one * theta.square()
            + 2.0 * input_delta * theta_one_minus_theta
            + input_derivatives * (1.0 - theta).square()
        )
        log_abs_det = torch.log(derivative_numerator) - 2.0 * torch.log(denominator)
        if inverse:
            transformed = input_cumulative_widths + theta * input_widths
            log_abs_det = -log_abs_det
        else:
            numerator = input_heights * (
                input_delta * theta.square() + input_derivatives * theta_one_minus_theta
            )
            transformed = input_cumulative_heights + numerator / denominator
        inside = (inputs >= -config.tail_bound) & (inputs <= config.tail_bound)
        return (
            torch.where(inside, transformed, inputs),
            torch.where(inside, log_abs_det, torch.zeros_like(log_abs_det)),
        )

    class Conditioner(nn.Module):
        def __init__(self, feature_index: int) -> None:
            super().__init__()
            context_width = config.context_features or 0
            input_width = max(feature_index + context_width, 1)
            layers: list[Any] = []
            width = input_width
            for _ in range(config.num_blocks):
                layers.extend(
                    [
                        nn.Linear(width, config.hidden_features),
                        nn.SiLU(),
                    ]
                )
                if config.dropout_probability:
                    layers.append(nn.Dropout(config.dropout_probability))
                width = config.hidden_features
            final = nn.Linear(width, parameter_count)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            with torch.no_grad():
                final.bias[2 * config.num_bins :].fill_(identity_derivative_bias)
            layers.append(final)
            self.network = nn.Sequential(*layers)
            self.feature_index = feature_index

        def forward(self, values: Any, context: Any | None) -> Any:
            parts = []
            if self.feature_index:
                parts.append(values[:, : self.feature_index])
            if context is not None:
                parts.append(context)
            if parts:
                conditioned = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
            else:
                # Derive the dummy input from the graph input so its batch
                # dimension remains symbolic during ONNX export.
                conditioned = torch.zeros_like(values[:, :1])
            return self.network(conditioned)

    class AutoregressiveSpline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conditioners = nn.ModuleList(
                Conditioner(index) for index in range(config.n_features)
            )

        def forward(self, values: Any, context: Any | None) -> tuple[Any, Any]:
            outputs = []
            log_abs_det = torch.zeros_like(values[:, 0])
            for index, conditioner in enumerate(self.conditioners):
                output, increment = spline(
                    values[:, index],
                    conditioner(values, context),
                    inverse=False,
                )
                outputs.append(output.unsqueeze(-1))
                log_abs_det = log_abs_det + increment
            return torch.cat(outputs, dim=-1), log_abs_det

        def inverse(self, values: Any, context: Any | None) -> tuple[Any, Any]:
            # Each newly reconstructed feature becomes conditioning input for
            # the next feature. Concatenating a zero-width prefix is avoided
            # because ONNX Runtime handles concrete feature slices better.
            outputs = []
            log_abs_det = torch.zeros_like(values[:, 0])
            for index, conditioner in enumerate(self.conditioners):
                if outputs:
                    prefix = torch.cat(outputs, dim=-1)
                    conditioner_values = torch.cat((prefix, values[:, index:]), dim=-1)
                else:
                    conditioner_values = values
                output, increment = spline(
                    values[:, index],
                    conditioner(conditioner_values, context),
                    inverse=True,
                )
                outputs.append(output.unsqueeze(-1))
                log_abs_det = log_abs_det + increment
            return torch.cat(outputs, dim=-1), log_abs_det

    class QuadraticSplineFlow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transforms = nn.ModuleList(
                AutoregressiveSpline() for _ in range(config.num_transforms)
            )
            self.register_buffer(
                "permutation",
                torch.arange(config.n_features - 1, -1, -1),
            )

        def _context(self, values: Any, context: Any | None) -> Any | None:
            if config.context_features is None:
                return None
            if context is None:
                raise ValueError("context is required by this conditional flow.")
            return context.expand(values.shape[0], -1)

        def data_to_base(
            self, values: Any, context: Any | None = None
        ) -> tuple[Any, Any]:
            result = values
            context = self._context(values, context)
            log_abs_det = torch.zeros_like(values[:, 0])
            for index, transform in enumerate(self.transforms):
                result, increment = transform(result, context)
                log_abs_det = log_abs_det + increment
                if index + 1 < len(self.transforms):
                    result = torch.index_select(result, 1, self.permutation)
            return result, log_abs_det

        def base_to_data(
            self, values: Any, context: Any | None = None
        ) -> tuple[Any, Any]:
            result = values
            context = self._context(values, context)
            log_abs_det = torch.zeros_like(values[:, 0])
            for index in range(len(self.transforms) - 1, -1, -1):
                if index + 1 < len(self.transforms):
                    result = torch.index_select(result, 1, self.permutation)
                result, increment = self.transforms[index].inverse(result, context)
                log_abs_det = log_abs_det + increment
            return result, log_abs_det

        def log_prob(self, values: Any, context: Any | None = None) -> Any:
            base, log_abs_det = self.data_to_base(values, context)
            base_log_prob = -0.5 * (base.square() + math.log(2.0 * math.pi)).sum(dim=-1)
            return base_log_prob + log_abs_det

    return QuadraticSplineFlow()


def _build_model(config: FlowConfig) -> Any:
    if config.flow_type == "realnvp":
        return _build_realnvp(config)
    return _build_quadratic_spline(config)


def _model_log_prob(
    model: Any, values: Any, context: Any | None, config: FlowConfig
) -> Any:
    if config.context_features is None:
        return model.log_prob(values)
    return model.log_prob(values, context=context)


def _model_data_to_base(
    model: Any, values: Any, context: Any | None, config: FlowConfig
) -> tuple[Any, Any]:
    if config.context_features is None:
        return model.data_to_base(values)
    return model.data_to_base(values, context=context)


def _model_base_to_data(
    model: Any, values: Any, context: Any | None, config: FlowConfig
) -> tuple[Any, Any]:
    if config.context_features is None:
        return model.base_to_data(values)
    return model.base_to_data(values, context=context)


@dataclass(frozen=True)
class FlowEpoch:
    """One optimization epoch."""

    epoch: int
    training_loss: float
    validation_loss: float


@dataclass
class FlowTrainingResult:
    """A fitted flow and its optimization history."""

    flow: ReferenceFlow
    history: tuple[FlowEpoch, ...]
    best_epoch: int
    stopped_early: bool

    def save_checkpoint(self, path: str | Path) -> tuple[Path, Path]:
        return self.flow.save(path, training_history=self.history)


@dataclass(frozen=True)
class FlowOnnxBundle:
    """Three deterministic graphs for a reference density."""

    log_prob_path: Path
    base_to_data_path: Path
    data_to_base_path: Path
    manifest_path: Path
    conditional: bool = False
    features: tuple[str, ...] = ()
    context_names: tuple[str, ...] = ()

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        *,
        expected_features: Sequence[str] | None = None,
        verify: bool = True,
    ) -> FlowOnnxBundle:
        """Load and verify a portable bundle and its ordered signatures."""

        path = Path(manifest_path)
        manifest = ArtifactManifest.load(path)
        if manifest.artifact_type != "reference-flow-onnx-bundle":
            raise ValueError(
                f"Expected a reference flow bundle, found {manifest.artifact_type!r}."
            )
        if verify:
            manifest.verify(path.parent)
        features = tuple(manifest.metadata.get("features", ()))
        context_names = tuple(manifest.metadata.get("context_names", ()))
        if not features:
            raise ValueError("Reference flow manifest has no feature signature.")
        if expected_features is not None and tuple(expected_features) != features:
            raise ValueError(
                "Reference flow feature order mismatch: expected "
                f"{tuple(expected_features)}, found {features}."
            )
        records = {record.kind: record for record in manifest.files}
        required = {
            "log-prob-onnx",
            "base-to-data-onnx",
            "data-to-base-onnx",
        }
        missing = required.difference(records)
        if missing:
            raise ValueError(
                f"Reference flow bundle is missing graph roles {sorted(missing)}."
            )
        return cls(
            log_prob_path=path.parent / records["log-prob-onnx"].path,
            base_to_data_path=path.parent / records["base-to-data-onnx"].path,
            data_to_base_path=path.parent / records["data-to-base-onnx"].path,
            manifest_path=path,
            conditional=bool(manifest.metadata.get("conditional", False)),
            features=features,
            context_names=context_names,
        )

    def _inputs(
        self, name: str, values: Any, context: Any | None
    ) -> dict[str, np.ndarray]:
        value_array = _as_2d(values, columns=len(self.features), name=name)
        feed = {name: value_array}
        if self.conditional:
            if context is None:
                raise ValueError("context is required by this conditional flow.")
            context_array = _as_2d(
                context, columns=len(self.context_names), name="context"
            )
            if len(context_array) == 1 and len(value_array) != 1:
                context_array = np.repeat(context_array, len(value_array), axis=0)
            if len(context_array) != len(value_array):
                raise ValueError(
                    "context must have one row or one row per feature vector."
                )
            feed["context"] = context_array
        elif context is not None:
            raise ValueError("context was provided to an unconditional flow.")
        return feed

    def log_prob(self, values: Any, *, context: Any | None = None) -> np.ndarray:
        output = OnnxRunner(self.log_prob_path).run(
            self._inputs("features", values, context)
        )
        return np.asarray(output["log_prob"]).reshape(-1)

    def base_to_data(self, base: Any, *, context: Any | None = None) -> np.ndarray:
        output = OnnxRunner(self.base_to_data_path).run(
            self._inputs("base", base, context)
        )
        return np.asarray(output["features"])

    def data_to_base(self, values: Any, *, context: Any | None = None) -> np.ndarray:
        output = OnnxRunner(self.data_to_base_path).run(
            self._inputs("features", values, context)
        )
        return np.asarray(output["base"])

    def sample(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
        context: Any | None = None,
    ) -> np.ndarray:
        """Sample reproducibly using caller-owned standard-normal base noise."""

        if int(n) < 1:
            raise ValueError("n must be positive.")
        if not self.features:
            raise ValueError("Bundle has no feature signature.")
        generator = np.random.default_rng() if rng is None else rng
        base = generator.standard_normal((int(n), len(self.features)), dtype=np.float32)
        return self.base_to_data(base, context=context)

    def parity(
        self,
        flow: ReferenceFlow,
        values: Any,
        *,
        base: Any | None = None,
        context: Any | None = None,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> dict[str, OnnxParityReport]:
        """Run native-versus-ONNX parity checks for every graph."""

        features = np.asarray(values, dtype=np.float32)
        base_values = (
            flow.data_to_base(features, context=context)
            if base is None
            else np.asarray(base, dtype=np.float32)
        )
        return {
            "log_prob": compare_outputs(
                flow.log_prob(features, context=context),
                self.log_prob(features, context=context),
                atol=atol,
                rtol=rtol,
            ),
            "base_to_data": compare_outputs(
                flow.base_to_data(base_values, context=context),
                self.base_to_data(base_values, context=context),
                atol=atol,
                rtol=rtol,
            ),
            "data_to_base": compare_outputs(
                flow.data_to_base(features, context=context),
                self.data_to_base(features, context=context),
                atol=atol,
                rtol=rtol,
            ),
        }


@dataclass
class ReferenceFlow:
    """A normalized flow density in the original feature coordinates."""

    model: Any
    scaler: AffineStandardizer
    features: tuple[str, ...]
    config: FlowConfig
    context_names: tuple[str, ...] = ()
    device: str = "cpu"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.features = tuple(self.features)
        self.context_names = tuple(self.context_names)
        if len(self.features) != self.config.n_features:
            raise ValueError("features must match config.n_features.")
        if self.scaler.n_features != self.config.n_features:
            raise ValueError("scaler must match config.n_features.")
        expected_context = self.config.context_features or 0
        if len(self.context_names) != expected_context:
            raise ValueError(
                "context_names must match config.context_features exactly."
            )
        self.to(self.device)

    @property
    def is_conditional(self) -> bool:
        return self.config.context_features is not None

    def to(self, device: str) -> ReferenceFlow:
        """Move the native model to *device* and return ``self``."""

        require_optional("torch", extra="flows", purpose="using a reference flow")
        self.model.to(device)
        self.device = str(device)
        return self

    def _context_array(self, context: Any | None, rows: int) -> np.ndarray | None:
        expected = self.config.context_features
        if expected is None:
            if context is not None:
                raise ValueError("context was provided to an unconditional flow.")
            return None
        if context is None:
            raise ValueError("context is required by this conditional flow.")
        values = _as_2d(context, columns=expected, name="context")
        if len(values) == 1 and rows != 1:
            values = np.repeat(values, rows, axis=0)
        if len(values) != rows:
            raise ValueError("context must have one row or one row per feature vector.")
        return values

    def _tensor_inputs(
        self, values: np.ndarray, context: np.ndarray | None
    ) -> tuple[Any, Any | None]:
        torch = require_optional(
            "torch", extra="flows", purpose="evaluating a reference flow"
        )
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        context_tensor = (
            None
            if context is None
            else torch.as_tensor(context, dtype=torch.float32, device=self.device)
        )
        return tensor, context_tensor

    def log_prob(self, values: Any, *, context: Any | None = None) -> np.ndarray:
        """Evaluate the normalized log density in original coordinates."""

        torch = require_optional(
            "torch", extra="flows", purpose="evaluating a reference flow"
        )
        array = _as_2d(values, columns=self.config.n_features, name="values")
        context_array = self._context_array(context, len(array))
        standardized = self.scaler.transform(array)
        tensor, context_tensor = self._tensor_inputs(standardized, context_array)
        self.model.eval()
        with torch.no_grad():
            result = _model_log_prob(self.model, tensor, context_tensor, self.config)
            result = result + self.scaler.forward_log_abs_det
        return result.detach().cpu().numpy().reshape(-1)

    def data_to_base(self, values: Any, *, context: Any | None = None) -> np.ndarray:
        """Map original feature vectors to deterministic standard-normal codes."""

        torch = require_optional(
            "torch", extra="flows", purpose="evaluating a reference flow"
        )
        array = _as_2d(values, columns=self.config.n_features, name="values")
        context_array = self._context_array(context, len(array))
        standardized = self.scaler.transform(array)
        tensor, context_tensor = self._tensor_inputs(standardized, context_array)
        self.model.eval()
        with torch.no_grad():
            result, _ = _model_data_to_base(
                self.model, tensor, context_tensor, self.config
            )
        return result.detach().cpu().numpy()

    def base_to_data(self, base: Any, *, context: Any | None = None) -> np.ndarray:
        """Map standard-normal codes to original feature coordinates."""

        torch = require_optional(
            "torch", extra="flows", purpose="evaluating a reference flow"
        )
        array = _as_2d(base, columns=self.config.n_features, name="base")
        context_array = self._context_array(context, len(array))
        tensor, context_tensor = self._tensor_inputs(array, context_array)
        self.model.eval()
        with torch.no_grad():
            result, _ = _model_base_to_data(
                self.model, tensor, context_tensor, self.config
            )
        return self.scaler.inverse_transform(result.detach().cpu().numpy())

    def sample(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
        context: Any | None = None,
    ) -> np.ndarray:
        """Draw reproducible samples using caller-controlled NumPy base noise."""

        if int(n) < 1:
            raise ValueError("n must be positive.")
        generator = np.random.default_rng() if rng is None else rng
        base = generator.standard_normal(
            (int(n), self.config.n_features), dtype=np.float32
        )
        return self.base_to_data(base, context=context)

    def save(
        self,
        path: str | Path,
        *,
        training_history: Sequence[FlowEpoch] | None = None,
    ) -> tuple[Path, Path]:
        """Save a native checkpoint and its checksum manifest."""

        torch = require_optional(
            "torch", extra="flows", purpose="saving a reference flow"
        )
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "context_names": list(self.context_names),
            "features": list(self.features),
            "metadata": self.metadata,
            "scaler": self.scaler.to_dict(),
            "state_dict": self.model.state_dict(),
            "training_history": [asdict(epoch) for epoch in (training_history or ())],
        }
        torch.save(payload, checkpoint_path)
        manifest_path = manifest_path_for(checkpoint_path)
        write_artifact_manifest(
            manifest_path,
            artifact_type="reference-flow-checkpoint",
            files={"pytorch-checkpoint": checkpoint_path},
            metadata={
                "conditional": self.is_conditional,
                "features": list(self.features),
                "flow_config": asdict(self.config),
            },
        )
        return checkpoint_path, manifest_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        verify: bool = True,
        expected_features: Sequence[str] | None = None,
        allow_unsafe_pickle: bool = False,
    ) -> ReferenceFlow:
        """Load a native checkpoint with integrity and safe-unpickling checks.

        A checksum sidecar is mandatory when ``verify`` is true. PyTorch's
        restricted ``weights_only`` loader is used by default; legacy
        unrestricted pickle loading requires an explicit opt-in.
        """

        torch = require_optional(
            "torch", extra="flows", purpose="loading a reference flow"
        )
        checkpoint_path = Path(path)
        manifest_path = manifest_path_for(checkpoint_path)
        if verify and not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing checkpoint integrity manifest {manifest_path}."
            )
        if manifest_path.is_file():
            manifest = ArtifactManifest.load(manifest_path)
            if manifest.artifact_type != "reference-flow-checkpoint":
                raise ValueError(
                    f"Unexpected checkpoint manifest type {manifest.artifact_type!r}."
                )
            checkpoint_records = [
                record
                for record in manifest.files
                if record.kind == "pytorch-checkpoint"
            ]
            if len(checkpoint_records) != 1:
                raise ValueError(
                    "Checkpoint manifest must contain exactly one "
                    "'pytorch-checkpoint' record."
                )
            recorded = (manifest_path.parent / checkpoint_records[0].path).resolve()
            if recorded != checkpoint_path.resolve():
                raise ValueError(
                    "Checkpoint path does not match its integrity manifest."
                )
            if verify:
                manifest.verify(manifest_path.parent)
        try:
            payload = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
        except Exception:
            if not allow_unsafe_pickle:
                raise ValueError(
                    "Checkpoint cannot be decoded by PyTorch's restricted "
                    "loader. Pass allow_unsafe_pickle=True only for a trusted "
                    "legacy checkpoint."
                ) from None
            payload = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        required = {
            "config",
            "context_names",
            "features",
            "metadata",
            "scaler",
            "state_dict",
        }
        if not isinstance(payload, Mapping) or not required.issubset(payload):
            raise ValueError(
                "Reference-flow checkpoint does not match the expected schema."
            )
        features = tuple(payload["features"])
        if expected_features is not None and tuple(expected_features) != features:
            raise ValueError(
                "Reference flow feature order mismatch: expected "
                f"{tuple(expected_features)}, found {features}."
            )
        config = FlowConfig(**payload["config"])
        model = _build_model(config)
        model.load_state_dict(payload["state_dict"])
        return cls(
            model=model,
            scaler=AffineStandardizer.from_dict(payload["scaler"]),
            features=features,
            config=config,
            context_names=tuple(payload.get("context_names", ())),
            device=device,
            metadata=dict(payload.get("metadata", {})),
        )

    def _onnx_wrappers(self) -> tuple[Any, Any, Any]:
        torch = require_optional(
            "torch", extra="flows", purpose="exporting a reference flow"
        )
        nn = torch.nn
        config = self.config
        model = self.model
        mean = torch.as_tensor(
            self.scaler.mean, dtype=torch.float32, device=self.device
        )
        scale = torch.as_tensor(
            self.scaler.scale, dtype=torch.float32, device=self.device
        )
        log_det = float(self.scaler.forward_log_abs_det)

        class EmbeddedLogProb(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model
                self.register_buffer("mean", mean)
                self.register_buffer("scale", scale)

            def forward(self, features: Any, context: Any | None = None) -> Any:
                standardized = (features - self.mean) / self.scale
                return (
                    _model_log_prob(self.model, standardized, context, config) + log_det
                )

        class EmbeddedDataToBase(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model
                self.register_buffer("mean", mean)
                self.register_buffer("scale", scale)

            def forward(self, features: Any, context: Any | None = None) -> Any:
                standardized = (features - self.mean) / self.scale
                result, _ = _model_data_to_base(
                    self.model, standardized, context, config
                )
                return result

        class EmbeddedBaseToData(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = model
                self.register_buffer("mean", mean)
                self.register_buffer("scale", scale)

            def forward(self, base: Any, context: Any | None = None) -> Any:
                standardized, _ = _model_base_to_data(self.model, base, context, config)
                return standardized * self.scale + self.mean

        return EmbeddedLogProb(), EmbeddedBaseToData(), EmbeddedDataToBase()

    def export_onnx(
        self,
        directory: str | Path,
        *,
        prefix: str = "reference_flow",
        example_values: Any | None = None,
        example_context: Any | None = None,
        opset_version: int = 17,
    ) -> FlowOnnxBundle:
        """Export log-density, base-to-data, and inverse deterministic graphs."""

        torch = require_optional(
            "torch", extra="flows", purpose="exporting a reference flow"
        )
        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        rows = 3
        if example_values is None:
            feature_array = np.tile(self.scaler.mean.astype(np.float32), (rows, 1))
        else:
            feature_array = _as_2d(
                example_values,
                columns=self.config.n_features,
                name="example_values",
            )
            rows = len(feature_array)
        context_array = self._context_array(example_context, rows)
        base_array = self.data_to_base(feature_array, context=context_array)
        feature_tensor = torch.as_tensor(
            feature_array, dtype=torch.float32, device=self.device
        )
        base_tensor = torch.as_tensor(
            base_array, dtype=torch.float32, device=self.device
        )
        input_names = ["features"]
        base_input_names = ["base"]
        feature_inputs: tuple[Any, ...] = (feature_tensor,)
        base_inputs: tuple[Any, ...] = (base_tensor,)
        if context_array is not None:
            context_tensor = torch.as_tensor(
                context_array, dtype=torch.float32, device=self.device
            )
            input_names.append("context")
            base_input_names.append("context")
            feature_inputs += (context_tensor,)
            base_inputs += (context_tensor,)
        log_prob_wrapper, base_wrapper, inverse_wrapper = self._onnx_wrappers()
        paths = {
            "log_prob": output_directory / f"{prefix}.log_prob.onnx",
            "base_to_data": output_directory / f"{prefix}.base_to_data.onnx",
            "data_to_base": output_directory / f"{prefix}.data_to_base.onnx",
        }
        common_metadata = {
            "conditional": self.is_conditional,
            "context_names": list(self.context_names),
            "features": list(self.features),
            "flow_config": asdict(self.config),
            "scaler_embedded": True,
        }
        try:
            export_torch_onnx(
                log_prob_wrapper,
                feature_inputs,
                paths["log_prob"],
                input_names=input_names,
                output_names=["log_prob"],
                artifact_type="reference-flow-log-prob-onnx",
                metadata=common_metadata,
                opset_version=opset_version,
            )
            export_torch_onnx(
                base_wrapper,
                base_inputs,
                paths["base_to_data"],
                input_names=base_input_names,
                output_names=["features"],
                artifact_type="reference-flow-base-to-data-onnx",
                metadata=common_metadata,
                opset_version=opset_version,
            )
            export_torch_onnx(
                inverse_wrapper,
                feature_inputs,
                paths["data_to_base"],
                input_names=input_names,
                output_names=["base"],
                artifact_type="reference-flow-data-to-base-onnx",
                metadata=common_metadata,
                opset_version=opset_version,
            )
        except Exception as exc:
            if self.config.flow_type == "quadratic-spline":
                raise RuntimeError(
                    "The installed torch/onnx combination could not "
                    "export the spline transform. Native checkpoints remain "
                    "valid; install current hnsbi-toolkit[flows] versions and "
                    "retry the ONNX export."
                ) from exc
            raise
        bundle_manifest = output_directory / f"{prefix}.manifest.json"
        files = {
            "log-prob-onnx": paths["log_prob"],
            "log-prob-manifest": manifest_path_for(paths["log_prob"]),
            "base-to-data-onnx": paths["base_to_data"],
            "base-to-data-manifest": manifest_path_for(paths["base_to_data"]),
            "data-to-base-onnx": paths["data_to_base"],
            "data-to-base-manifest": manifest_path_for(paths["data_to_base"]),
        }
        write_artifact_manifest(
            bundle_manifest,
            artifact_type="reference-flow-onnx-bundle",
            files=files,
            metadata=common_metadata,
        )
        return FlowOnnxBundle(
            log_prob_path=paths["log_prob"],
            base_to_data_path=paths["base_to_data"],
            data_to_base_path=paths["data_to_base"],
            manifest_path=bundle_manifest,
            conditional=self.is_conditional,
            features=self.features,
            context_names=self.context_names,
        )


class FlowTrainer:
    """Fit a :class:`ReferenceFlow` with weighted maximum likelihood."""

    def __init__(
        self,
        flow_config: FlowConfig,
        training_config: FlowTrainingConfig | None = None,
    ) -> None:
        self.flow_config = flow_config
        self.training_config = training_config or FlowTrainingConfig()

    @staticmethod
    def _weighted_loss(log_prob: Any, weights: Any) -> Any:
        return -(log_prob * weights).sum() / weights.sum()

    @staticmethod
    def random_split_indices(
        rows: int,
        validation_fraction: float,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact random train/validation split used by ``fit``."""

        rows = int(rows)
        fraction = float(validation_fraction)
        if rows < 2:
            raise ValueError("At least two rows are required for a flow split.")
        if not 0.0 <= fraction < 1.0:
            raise ValueError("validation_fraction must lie in [0, 1).")
        permutation = rng.permutation(rows)
        validation_count = max(1, int(round(fraction * rows))) if fraction > 0 else 0
        return permutation[validation_count:], permutation[:validation_count]

    def fit(
        self,
        values: Any,
        *,
        features: Sequence[str],
        weights: Any | None = None,
        context: Any | None = None,
        context_names: Sequence[str] = (),
        validation_values: Any | None = None,
        validation_weights: Any | None = None,
        validation_context: Any | None = None,
        seed: int = 0,
    ) -> FlowTrainingResult:
        """Train and return a flow without writing implicit artifacts.

        ``validation_values`` supplies a fully external validation measure.
        When present, no training row is selected for validation and the
        scaler is fitted exclusively on ``values``. Without it, the configured
        random validation fraction retains the historical behavior.
        """

        torch = require_optional(
            "torch", extra="flows", purpose="training a reference flow"
        )
        array = _as_2d(values, columns=self.flow_config.n_features, name="values")
        feature_names = tuple(features)
        if len(feature_names) != self.flow_config.n_features:
            raise ValueError("features must match flow_config.n_features.")
        if len(array) < 2:
            raise ValueError("At least two events are required for training.")

        def validated_weights(
            supplied: Any | None,
            *,
            rows: int,
            name: str,
        ) -> np.ndarray:
            result = (
                np.ones(rows, dtype=np.float32)
                if supplied is None
                else np.asarray(supplied, dtype=np.float32).reshape(-1)
            )
            if len(result) != rows:
                raise ValueError(f"{name} must contain one value per event.")
            if not np.isfinite(result).all() or np.any(result < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
            return result

        event_weights = validated_weights(weights, rows=len(array), name="weights")
        if not float(np.sum(event_weights)) > 0:
            raise ValueError("weights must have positive sum.")
        expected_context = self.flow_config.context_features
        names = tuple(context_names)
        if expected_context is None:
            if context is not None or names:
                raise ValueError(
                    "context is only valid for a conditional flow configuration."
                )
            context_array = None
        else:
            if len(names) != expected_context:
                raise ValueError(
                    "context_names must match flow_config.context_features."
                )
            context_array = _as_2d(context, columns=expected_context, name="context")
            if len(context_array) != len(array):
                raise ValueError("context must contain one row per event.")

        rng = np.random.default_rng(seed)
        training_count = len(array)
        external_validation = validation_values is not None
        if external_validation:
            validation_array = _as_2d(
                validation_values,
                columns=self.flow_config.n_features,
                name="validation_values",
            )
            if not len(validation_array):
                raise ValueError("validation_values must contain at least one event.")
            external_weights = validated_weights(
                validation_weights,
                rows=len(validation_array),
                name="validation_weights",
            )
            if expected_context is None:
                if validation_context is not None:
                    raise ValueError(
                        "validation_context is only valid for a conditional flow."
                    )
                external_context = None
            else:
                external_context = _as_2d(
                    validation_context,
                    columns=expected_context,
                    name="validation_context",
                )
                if len(external_context) != len(validation_array):
                    raise ValueError(
                        "validation_context must contain one row per validation event."
                    )
            array = np.concatenate([array, validation_array], axis=0)
            event_weights = np.concatenate([event_weights, external_weights])
            if context_array is not None:
                assert external_context is not None
                context_array = np.concatenate([context_array, external_context])
            training_indices = np.arange(training_count, dtype=np.int64)
            validation_indices = np.arange(training_count, len(array), dtype=np.int64)
        else:
            if validation_weights is not None or validation_context is not None:
                raise ValueError(
                    "validation_weights/validation_context require validation_values."
                )
            training_indices, validation_indices = self.random_split_indices(
                len(array),
                self.training_config.validation_fraction,
                rng=rng,
            )
        validation_count = len(validation_indices)
        if len(training_indices) == 0:
            raise ValueError("validation_fraction leaves no training events.")
        if not float(np.sum(event_weights[training_indices])) > 0:
            raise ValueError("The training split has zero total event weight.")
        if (
            validation_count
            and not float(np.sum(event_weights[validation_indices])) > 0
        ):
            raise ValueError("The validation split has zero total event weight.")

        scaler = AffineStandardizer.fit(
            array[training_indices], weights=event_weights[training_indices]
        )
        standardized = scaler.transform(array)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = self.training_config.device
        model = _build_model(self.flow_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.training_config.learning_rate_factor,
            patience=max(1, self.training_config.patience // 3),
        )

        def tensors(indices: np.ndarray) -> tuple[Any, Any, Any | None]:
            batch_values = torch.as_tensor(
                standardized[indices], dtype=torch.float32, device=device
            )
            batch_weights = torch.as_tensor(
                event_weights[indices], dtype=torch.float32, device=device
            )
            batch_context = (
                None
                if context_array is None
                else torch.as_tensor(
                    context_array[indices], dtype=torch.float32, device=device
                )
            )
            return batch_values, batch_weights, batch_context

        def evaluate(indices: np.ndarray) -> float:
            model.eval()
            with torch.no_grad():
                batch_values, batch_weights, batch_context = tensors(indices)
                log_prob = _model_log_prob(
                    model, batch_values, batch_context, self.flow_config
                )
                loss = self._weighted_loss(log_prob, batch_weights)
            return float(loss.detach().cpu())

        history: list[FlowEpoch] = []
        best_loss = math.inf
        best_epoch = 0
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        stopped_early = False
        for epoch in range(1, self.training_config.epochs + 1):
            model.train()
            shuffled = rng.permutation(training_indices)
            for start in range(0, len(shuffled), self.training_config.batch_size):
                indices = shuffled[start : start + self.training_config.batch_size]
                if not float(np.sum(event_weights[indices])) > 0:
                    continue
                batch_values, batch_weights, batch_context = tensors(indices)
                optimizer.zero_grad(set_to_none=True)
                log_prob = _model_log_prob(
                    model, batch_values, batch_context, self.flow_config
                )
                loss = self._weighted_loss(log_prob, batch_weights)
                loss.backward()
                if self.training_config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.training_config.gradient_clip_norm,
                    )
                optimizer.step()
            training_loss = evaluate(training_indices)
            validation_loss = (
                evaluate(validation_indices) if validation_count else training_loss
            )
            scheduler.step(validation_loss)
            history.append(
                FlowEpoch(
                    epoch=epoch,
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                )
            )
            if validation_loss < best_loss - self.training_config.min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.training_config.patience:
                    stopped_early = True
                    break
        if best_state is None:
            raise RuntimeError("Flow training did not produce a finite checkpoint.")
        model.load_state_dict(best_state)
        flow = ReferenceFlow(
            model=model,
            scaler=scaler,
            features=feature_names,
            config=self.flow_config,
            context_names=names,
            device=device,
            metadata={
                "best_epoch": best_epoch,
                "seed": int(seed),
                "split": {
                    "external_validation": external_validation,
                    "training_rows": len(training_indices),
                    "validation_rows": len(validation_indices),
                },
                "training_config": asdict(self.training_config),
            },
        )
        return FlowTrainingResult(
            flow=flow,
            history=tuple(history),
            best_epoch=best_epoch,
            stopped_early=stopped_early,
        )
