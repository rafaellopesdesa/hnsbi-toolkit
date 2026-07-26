"""Concrete, portable training backend for the dual hNPE--hNDE model.

The module deliberately keeps Torch, nflows, ONNX, and ONNX Runtime behind
method calls.  Importing :mod:`hnsbi.bayes` therefore remains possible in a
NumPy-only environment, while a configured training run produces:

* conditional quadratic-spline flows for ``q_phi`` and ``q_eta``;
* paired density-ratio ensembles trained through :class:`RatioTrainer`;
* an amortized ``theta -> log Z_C(theta)`` Torch regressor;
* one checksummed artifact manifest per learned object;
* a five-object :class:`~hnsbi.bayes.artifacts.DualArtifactManifest`.

Every deployable graph consumes physical-space arrays.  Affine transforms are
embedded in the ONNX graphs, and every export is checked against its native
runtime before it is admitted to the final manifest.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import write_artifact_manifest
from ..flows import (
    FlowConfig,
    FlowOnnxBundle,
    FlowTrainer,
    FlowTrainingConfig,
    FlowTrainingResult,
    ReferenceFlow,
)
from ..onnx import (
    OnnxParityReport,
    OnnxRunner,
    compare_outputs,
    export_torch_onnx,
    require_optional,
)
from ..ratios import (
    RatioBackendResult,
    RatioEnsemble,
    RatioTrainer,
    RatioTrainingBackend,
    RatioTrainingConfig,
    RatioTrainingResult,
)
from ._array import as_2d, logmeanexp
from .artifacts import (
    DualArtifactManifest,
    DualArtifactSpec,
    EnsembleSpec,
    FeatureSignature,
    OnnxGraphSpec,
    TransformSpec,
    create_dual_artifact_manifest,
)
from .data import (
    DualTrainingData,
    PairedClassifierDataset,
    ProposalDataset,
    group_train_validation_split,
)
from .model import DualModel
from .training import DualTrainer


def _device(value: Any) -> str:
    requested = str(value or "cpu")
    if requested != "auto":
        return requested
    torch = require_optional(
        "torch",
        extra="bayes",
        purpose="selecting a Bayesian training device",
    )
    return "cuda" if torch.cuda.is_available() else "cpu"


def _strict_keys(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    location: str,
) -> dict[str, Any]:
    result = dict(value)
    unknown = set(result).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown {location} fields: {sorted(unknown)}.")
    return result


def _flow_training_config(value: Mapping[str, Any]) -> FlowTrainingConfig:
    training = _strict_keys(
        value,
        allowed={
            "epochs",
            "batch_size",
            "learning_rate",
            "validation_fraction",
            "early_stopping_patience",
            "lr_reduction_factor",
            "max_events",
            "seed",
            "device",
        },
        location="flow training",
    )
    return FlowTrainingConfig(
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        validation_fraction=float(training.get("validation_fraction", 0.2)),
        patience=int(training.get("early_stopping_patience", 20)),
        learning_rate_factor=float(training.get("lr_reduction_factor", 0.5)),
        device=str(training.get("device", "cpu")),
    )


@dataclass(frozen=True)
class ConditionalFlowStageConfig:
    """Typed configuration for one conditional flow stage."""

    flow: FlowConfig
    training: FlowTrainingConfig
    target_features: tuple[str, ...]
    context_features: tuple[str, ...]
    seed: int = 0
    max_events: int | None = None
    onnx_opset: int = 17

    def __post_init__(self) -> None:
        targets = tuple(self.target_features)
        contexts = tuple(self.context_features)
        if len(targets) != self.flow.n_features:
            raise ValueError("target_features do not match the flow dimension.")
        if len(contexts) != (self.flow.context_features or 0):
            raise ValueError("context_features do not match the flow context.")
        if self.flow.flow_type != "quadratic-spline":
            raise ValueError(
                "Dual conditional densities require architecture='quadratic_spline'."
            )
        if int(self.onnx_opset) < 17:
            raise ValueError("Bayesian flow ONNX export requires opset >= 17.")
        if self.max_events is not None and int(self.max_events) < 2:
            raise ValueError("max_events must be at least two when provided.")
        object.__setattr__(self, "target_features", targets)
        object.__setattr__(self, "context_features", contexts)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        target_features: Sequence[str],
        context_features: Sequence[str],
    ) -> ConditionalFlowStageConfig:
        """Translate a schema flow mapping without importing heavy runtimes."""

        specification = _strict_keys(
            value,
            allowed={
                "architecture",
                "n_coupling_layers",
                "hidden_features",
                "hidden_layers",
                "scale_clip",
                "spline_num_bins",
                "spline_tail_bound",
                "dropout_probability",
                "training",
                "onnx_opset",
            },
            location="conditional flow",
        )
        targets = tuple(target_features)
        contexts = tuple(context_features)
        training_mapping = dict(specification["training"])
        return cls(
            flow=FlowConfig(
                n_features=len(targets),
                flow_type=str(specification["architecture"]),
                num_transforms=int(specification["n_coupling_layers"]),
                hidden_features=int(specification["hidden_features"]),
                num_blocks=int(specification["hidden_layers"]),
                num_bins=int(specification.get("spline_num_bins", 8)),
                tail_bound=float(specification.get("spline_tail_bound", 4.0)),
                dropout_probability=float(
                    specification.get("dropout_probability", 0.0)
                ),
                context_features=len(contexts),
                max_log_scale=float(specification.get("scale_clip", 2.0)),
            ),
            training=_flow_training_config(training_mapping),
            target_features=targets,
            context_features=contexts,
            seed=int(training_mapping.get("seed", 0)),
            max_events=(
                None
                if training_mapping.get("max_events") is None
                else int(training_mapping["max_events"])
            ),
            onnx_opset=int(specification.get("onnx_opset", 17)),
        )


@dataclass(frozen=True)
class RatioStageConfig:
    """Typed configuration for one paired residual classifier."""

    backend: str
    training: RatioTrainingConfig
    validation_fraction: float
    onnx_opset: int = 17
    normalization: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"native", "nsbi_common_utils"}:
            raise ValueError("Ratio backend must be 'native' or 'nsbi_common_utils'.")
        if self.normalization not in {None, "conditional_reference_mean"}:
            raise ValueError(
                "Bayesian ratio normalization must be "
                "'conditional_reference_mean' when provided."
            )
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("ratio validation_fraction must lie in (0, 1).")
        if int(self.onnx_opset) < 17:
            raise ValueError("Bayesian ratio ONNX export requires opset >= 17.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RatioStageConfig:
        specification = _strict_keys(
            value,
            allowed={
                "backend",
                "ensemble_size",
                "training",
                "normalization",
                "onnx_opset",
            },
            location="ratio model",
        )
        training = _strict_keys(
            specification["training"],
            allowed={
                "epochs",
                "batch_size",
                "learning_rate",
                "hidden_layers",
                "neurons",
                "activation",
                "validation_fraction",
                "holdout_fraction",
                "early_stopping_patience",
                "seed",
            },
            location="ratio training",
        )
        validation_fraction = float(training.get("validation_fraction", 0.2))
        return cls(
            backend=str(specification.get("backend", "native")),
            training=RatioTrainingConfig(
                ensemble_size=int(specification["ensemble_size"]),
                hidden_layers=int(training["hidden_layers"]),
                neurons=int(training["neurons"]),
                epochs=int(training["epochs"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                use_log_loss=True,
                calibration=False,
                validation_fraction=validation_fraction,
                holdout_fraction=float(training.get("holdout_fraction", 0.2)),
                early_stopping=True,
                patience=int(training.get("early_stopping_patience", 20)),
                activation=str(training.get("activation", "swish")),
                seed=int(training.get("seed", 0)),
                run_diagnostics=False,
            ),
            validation_fraction=validation_fraction,
            normalization=specification.get("normalization"),
            onnx_opset=int(specification.get("onnx_opset", 17)),
        )


@dataclass(frozen=True)
class LogNormalizerStageConfig:
    """Typed Monte-Carlo target and MLP configuration for ``log Z_C``."""

    reference_draws_per_context: int
    contexts: int
    hidden_features: int
    hidden_layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    validation_fraction: float = 0.2
    patience: int = 20
    learning_rate_factor: float = 0.5
    seed: int = 0
    device: str = "cpu"
    onnx_opset: int = 17

    def __post_init__(self) -> None:
        for name in (
            "reference_draws_per_context",
            "contexts",
            "hidden_features",
            "hidden_layers",
            "epochs",
            "batch_size",
            "patience",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.reference_draws_per_context < 2:
            raise ValueError("reference_draws_per_context must be at least two.")
        if self.contexts < 2:
            raise ValueError("contexts must be at least two.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0, 1).")
        if not 0.0 < self.learning_rate_factor < 1.0:
            raise ValueError("learning_rate_factor must lie in (0, 1).")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> LogNormalizerStageConfig:
        specification = _strict_keys(
            value,
            allowed={
                "reference_draws_per_context",
                "contexts",
                "hidden_features",
                "hidden_layers",
                "training",
                "onnx_opset",
            },
            location="normalizer",
        )
        training = _strict_keys(
            specification["training"],
            allowed={
                "epochs",
                "batch_size",
                "learning_rate",
                "validation_fraction",
                "early_stopping_patience",
                "lr_reduction_factor",
                "seed",
                "device",
            },
            location="normalizer training",
        )
        return cls(
            reference_draws_per_context=int(
                specification["reference_draws_per_context"]
            ),
            contexts=int(specification["contexts"]),
            hidden_features=int(specification["hidden_features"]),
            hidden_layers=int(specification["hidden_layers"]),
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            validation_fraction=float(training.get("validation_fraction", 0.2)),
            patience=int(training.get("early_stopping_patience", 20)),
            learning_rate_factor=float(training.get("lr_reduction_factor", 0.5)),
            seed=int(training.get("seed", 0)),
            device=str(training.get("device", "cpu")),
            onnx_opset=int(specification.get("onnx_opset", 17)),
        )


@dataclass
class NativeConditionalDensity:
    """Batch-conditional adapter around :class:`ReferenceFlow`."""

    flow: ReferenceFlow

    def log_prob(
        self,
        target: np.ndarray,
        *,
        context: np.ndarray,
    ) -> np.ndarray:
        return self.flow.log_prob(target, context=context)

    def sample(
        self,
        n: int,
        *,
        context: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        contexts = as_2d(context, "context")
        n = int(n)
        if n < 1:
            raise ValueError("n must be positive.")
        generator = np.random.default_rng() if rng is None else rng
        noise = generator.standard_normal(
            (len(contexts), n, self.flow.config.n_features),
            dtype=np.float32,
        )
        flat_context = np.repeat(contexts, n, axis=0)
        result = self.flow.base_to_data(
            noise.reshape(-1, self.flow.config.n_features),
            context=flat_context,
        )
        return result.reshape(len(contexts), n, self.flow.config.n_features)


@dataclass
class NativeLogRatio:
    """Expose a ``[theta, x]`` ratio ensemble under DualModel conventions."""

    ensemble: RatioEnsemble
    artifact_name: str

    def __post_init__(self) -> None:
        if self.artifact_name not in {"r_p", "r_c"}:
            raise ValueError("artifact_name must be 'r_p' or 'r_c'.")

    def log_ratio(
        self,
        target: np.ndarray,
        *,
        context: np.ndarray,
    ) -> np.ndarray:
        target = as_2d(target, "target")
        context = as_2d(context, "context")
        if len(target) != len(context):
            raise ValueError("target and context must be row-aligned.")
        values = (
            np.column_stack([target, context])
            if self.artifact_name == "r_p"
            else np.column_stack([context, target])
        )
        return self.ensemble.log_ratio(values)


@dataclass
class _NativeRatioEvaluator:
    module: Any
    device: str

    def log_ratio(self, values: np.ndarray) -> np.ndarray:
        torch = require_optional(
            "torch", extra="bayes", purpose="evaluating a native ratio"
        )
        array = np.asarray(values, dtype=np.float32)
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        self.module.eval()
        with torch.no_grad():
            result = self.module(tensor)
        return result.detach().cpu().numpy().reshape(-1)

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return np.exp(np.clip(self.log_ratio(values), -80.0, 80.0))


def _activation(torch: Any, name: str) -> Any:
    normalized = str(name).lower()
    if normalized == "relu":
        return torch.nn.ReLU
    if normalized == "swish":
        return torch.nn.SiLU
    if normalized == "tanh":
        return torch.nn.Tanh
    raise ValueError(f"Unsupported native ratio activation {name!r}.")


@dataclass
class NativeTorchRatioBackend:
    """Torch binary-classifier backend used through :class:`RatioTrainer`."""

    theta_features: tuple[str, ...]
    observation_features: tuple[str, ...]
    artifact_name: str
    validation_pairs: PairedClassifierDataset
    opset_version: int = 17
    device: str = "cpu"
    name: str = field(default="native_torch", init=False)

    def train_member(
        self,
        *,
        numerator_values: np.ndarray,
        denominator_values: np.ndarray,
        numerator_weights: np.ndarray,
        denominator_weights: np.ndarray,
        features: tuple[str, ...],
        output_directory: Path,
        member_index: int,
        numerator_name: str,
        denominator_name: str,
        config: RatioTrainingConfig,
    ) -> RatioBackendResult:
        """Train one balanced classifier and export a physical-input graph."""

        del numerator_name, denominator_name
        torch = require_optional(
            "torch", extra="bayes", purpose="training a native density ratio"
        )
        n_theta = len(self.theta_features)
        expected = self.theta_features + self.observation_features
        if tuple(features) != expected:
            raise ValueError("Native dual ratios require ordered features [theta, x].")
        values = np.concatenate([numerator_values, denominator_values], axis=0).astype(
            np.float32
        )
        labels = np.concatenate(
            [
                np.ones(len(numerator_values), dtype=np.float32),
                np.zeros(len(denominator_values), dtype=np.float32),
            ]
        )
        weights = np.concatenate(
            [numerator_weights, denominator_weights], axis=0
        ).astype(np.float32)
        validation_values, validation_labels, _ = self.validation_pairs.stacked()
        validation_values = validation_values.astype(np.float32)
        validation_labels = validation_labels.astype(np.float32)
        mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(values, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(scale, np.float32(1.0e-6))
        activation = _activation(torch, config.activation)

        class EmbeddedRatio(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
                self.register_buffer(
                    "scale", torch.as_tensor(scale, dtype=torch.float32)
                )
                layers: list[Any] = []
                width = len(features)
                for _ in range(config.hidden_layers):
                    layers.extend(
                        [
                            torch.nn.Linear(width, config.neurons),
                            activation(),
                        ]
                    )
                    width = config.neurons
                layers.append(torch.nn.Linear(width, 1))
                self.network = torch.nn.Sequential(*layers)

            def forward(self, rows: Any) -> Any:
                return self.network((rows - self.mean) / self.scale).reshape(-1)

        torch.manual_seed(config.seed + member_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed + member_index)
        device = self.device
        model = EmbeddedRatio().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=1.0e-5
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=min(float(config.learning_rate_factor), 0.99),
            patience=max(1, config.patience // 3),
        )
        rng = np.random.default_rng(config.seed + member_index)
        training_tensor = torch.as_tensor(values, device=device)
        label_tensor = torch.as_tensor(labels, device=device)
        weight_tensor = torch.as_tensor(weights, device=device)
        validation_tensor = torch.as_tensor(validation_values, device=device)
        validation_label_tensor = torch.as_tensor(validation_labels, device=device)
        best_state: dict[str, Any] | None = None
        best_loss = math.inf
        epochs_without_improvement = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, config.epochs + 1):
            model.train()
            permutation = rng.permutation(len(values))
            for start in range(0, len(values), config.batch_size):
                indices = torch.as_tensor(
                    permutation[start : start + config.batch_size],
                    dtype=torch.long,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(training_tensor[indices])
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, label_tensor[indices], reduction="none"
                )
                batch_weights = weight_tensor[indices]
                loss = (losses * batch_weights).sum() / batch_weights.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                train_logits = model(training_tensor)
                train_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    train_logits, label_tensor, reduction="none"
                )
                train_loss = float(
                    (train_losses * weight_tensor).sum() / weight_tensor.sum()
                )
                validation_loss = float(
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        model(validation_tensor),
                        validation_label_tensor,
                    )
                )
            scheduler.step(validation_loss)
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1.0e-6:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if (
                    config.early_stopping
                    and epochs_without_improvement >= config.patience
                ):
                    break
        if best_state is None:  # pragma: no cover - at least one epoch is required
            raise RuntimeError("Native ratio training produced no checkpoint.")
        model.load_state_dict(best_state)
        evaluator = _NativeRatioEvaluator(model, device)
        output_directory.mkdir(parents=True, exist_ok=True)
        history_path = output_directory / "training_history.json"
        history_path.write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        class SplitInputs(torch.nn.Module):
            def __init__(self, fitted: Any) -> None:
                super().__init__()
                self.fitted = fitted

            def forward(self, theta: Any, x: Any) -> Any:
                return self.fitted(torch.cat([theta, x], dim=1))

        graph_path = output_directory / "log_ratio.onnx"
        rows = min(32, len(values))
        theta_example = torch.as_tensor(
            values[:rows, :n_theta], dtype=torch.float32, device=device
        )
        x_example = torch.as_tensor(
            values[:rows, n_theta:], dtype=torch.float32, device=device
        )
        export_torch_onnx(
            SplitInputs(model),
            (theta_example, x_example),
            graph_path,
            input_names=["theta", "x"],
            output_names=["log_ratio"],
            artifact_type="dual-native-log-ratio-onnx",
            metadata={
                "artifact_name": self.artifact_name,
                "feature_order": list(features),
                "member": member_index,
                "scaler_embedded": True,
            },
            opset_version=self.opset_version,
        )
        portable = (
            OnnxRunner(graph_path)
            .run(
                {
                    "theta": values[:rows, :n_theta],
                    "x": values[:rows, n_theta:],
                }
            )["log_ratio"]
            .reshape(-1)
        )
        parity = compare_outputs(evaluator.log_ratio(values[:rows]), portable)
        parity.assert_close()
        parity_path = output_directory / "onnx_parity.json"
        parity_path.write_text(
            json.dumps(parity.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return RatioBackendResult(
            evaluator=evaluator,
            files={
                "dual-log-ratio-onnx": graph_path,
                "onnx-parity": parity_path,
                "training-history": history_path,
            },
            metadata={
                "onnx_parity": parity.to_dict(),
                "best_validation_loss": best_loss,
                "epochs": len(history),
            },
        )


@dataclass
class TorchLogNormalizer:
    """Native ``theta -> log Z_C(theta)`` evaluator."""

    module: Any
    device: str

    def log_normalization(self, theta: np.ndarray) -> np.ndarray:
        torch = require_optional("torch", extra="bayes", purpose="evaluating log Z_C")
        values = as_2d(theta, "theta").astype(np.float32)
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self.module.eval()
        with torch.no_grad():
            result = self.module(tensor)
        return result.detach().cpu().numpy().reshape(-1)

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        return self.log_normalization(theta)


@dataclass(frozen=True)
class FlowStageArtifacts:
    training: FlowTrainingResult
    checkpoint_path: Path
    onnx_bundle: FlowOnnxBundle
    parity: Mapping[str, OnnxParityReport]
    artifact_manifest_path: Path


@dataclass(frozen=True)
class RatioStageArtifacts:
    training: RatioTrainingResult
    parity: Mapping[int, OnnxParityReport]
    graph_paths: tuple[Path, ...]
    artifact_manifest_path: Path


@dataclass(frozen=True)
class NormalizerStageArtifacts:
    model: TorchLogNormalizer
    contexts: np.ndarray
    log_targets: np.ndarray
    graph_path: Path
    parity: OnnxParityReport
    artifact_manifest_path: Path


@dataclass(frozen=True)
class NativeDualTrainingArtifacts:
    """Complete native run, including the portable five-object manifest."""

    model: DualModel
    manifest: DualArtifactManifest
    manifest_path: Path
    stages: Mapping[
        str,
        FlowStageArtifacts | RatioStageArtifacts | NormalizerStageArtifacts,
    ]


def _paired_train_validation(
    pairs: PairedClassifierDataset,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[PairedClassifierDataset, PairedClassifierDataset]:
    split = group_train_validation_split(
        pairs.group_ids,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    def select(indices: np.ndarray) -> PairedClassifierDataset:
        return PairedClassifierDataset(
            positive=pairs.positive[indices],
            negative=pairs.negative[indices],
            group_ids=pairs.group_ids[indices],
            shared_quantity=pairs.shared_quantity,
        )

    return select(split.training_indices), select(split.validation_indices)


def _paired_train_validation_holdout(
    pairs: PairedClassifierDataset,
    *,
    validation_fraction: float,
    holdout_fraction: float,
    seed: int,
) -> tuple[
    PairedClassifierDataset,
    PairedClassifierDataset,
    PairedClassifierDataset,
]:
    training_pool, holdout = _paired_train_validation(
        pairs,
        validation_fraction=holdout_fraction,
        seed=seed,
    )
    training, validation = _paired_train_validation(
        training_pool,
        validation_fraction=validation_fraction,
        seed=seed + 1,
    )
    return training, validation, holdout


def _select_contexts(
    values: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    contexts = as_2d(values, "normalizer context")
    if not len(contexts):
        raise ValueError("At least one normalizer context is required.")
    if len(contexts) < int(count):
        raise ValueError(
            f"Configured normalizer contexts={int(count)} but only "
            f"{len(contexts)} context rows were supplied."
        )
    indices = rng.choice(len(contexts), size=int(count), replace=False)
    return contexts[indices]


class NativeDualBackend:
    """Config-driven implementation of :class:`BayesianTrainingBackend`."""

    def __init__(
        self,
        *,
        posterior_flow: ConditionalFlowStageConfig,
        posterior_ratio: RatioStageConfig,
        likelihood_flow: ConditionalFlowStageConfig,
        likelihood_ratio: RatioStageConfig,
        normalizer: LogNormalizerStageConfig,
        output_directory: str | Path,
        defensive_epsilon: float = 0.0,
        source_provenance: Mapping[str, Any] | None = None,
        config_provenance: Mapping[str, Any] | None = None,
        ratio_backends: Mapping[str, RatioTrainingBackend] | None = None,
    ) -> None:
        if posterior_flow.target_features != likelihood_flow.context_features:
            raise ValueError("Posterior target and likelihood context must be theta.")
        if posterior_flow.context_features != likelihood_flow.target_features:
            raise ValueError("Posterior context and likelihood target must be x.")
        if posterior_ratio.normalization is not None:
            raise ValueError(
                "posterior_ratio must not declare likelihood normalization."
            )
        if likelihood_ratio.normalization != "conditional_reference_mean":
            raise ValueError(
                "likelihood_ratio requires normalization='conditional_reference_mean'."
            )
        epsilon = float(defensive_epsilon)
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("defensive_epsilon must lie in [0, 1).")
        self.posterior_flow = posterior_flow
        self.posterior_ratio = posterior_ratio
        self.likelihood_flow = likelihood_flow
        self.likelihood_ratio = likelihood_ratio
        self.normalizer = normalizer
        self.output_directory = Path(output_directory)
        self.defensive_epsilon = epsilon
        self.source_provenance = dict(
            source_provenance
            or {"training_data": "caller-provided rho/nu/kappa simulator pairs"}
        )
        self.config_provenance = dict(
            config_provenance or {"backend": "hnsbi.native-dual"}
        )
        self.ratio_backends = dict(ratio_backends or {})
        self.stages: dict[
            str,
            FlowStageArtifacts | RatioStageArtifacts | NormalizerStageArtifacts,
        ] = {}
        self._specifications: dict[str, DualArtifactSpec] = {}
        self._manifest: DualArtifactManifest | None = None
        self._manifest_path = self.output_directory / "dual_model.manifest.json"

    @property
    def theta_features(self) -> tuple[str, ...]:
        return self.posterior_flow.target_features

    @property
    def observation_features(self) -> tuple[str, ...]:
        return self.posterior_flow.context_features

    @property
    def manifest(self) -> DualArtifactManifest:
        if self._manifest is None:
            raise RuntimeError("The dual manifest is available after z_c training.")
        return self._manifest

    @property
    def manifest_path(self) -> Path:
        if self._manifest is None:
            raise RuntimeError("The dual manifest is available after z_c training.")
        return self._manifest_path

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, Any],
        *,
        observation_features: Sequence[str],
        output_directory: str | Path | None = None,
        base_directory: str | Path = ".",
        source_provenance: Mapping[str, Any] | None = None,
        ratio_backends: Mapping[str, RatioTrainingBackend] | None = None,
    ) -> NativeDualBackend:
        """Build the concrete backend from the validated ``bayesian`` section."""

        section = dict(value)
        theta = tuple(section["theta_features"])
        observations = tuple(observation_features)
        if output_directory is None:
            configured = Path(section["output_bundle"])
            output = (
                configured
                if configured.is_absolute()
                else Path(base_directory) / configured
            )
        else:
            output = Path(output_directory)
        provenance = {
            key: value
            for key, value in section.items()
            if key not in {"datasets", "design_distributions"}
        }
        return cls(
            posterior_flow=ConditionalFlowStageConfig.from_mapping(
                section["posterior_flow"],
                target_features=theta,
                context_features=observations,
            ),
            posterior_ratio=RatioStageConfig.from_mapping(section["posterior_ratio"]),
            likelihood_flow=ConditionalFlowStageConfig.from_mapping(
                section["likelihood_flow"],
                target_features=observations,
                context_features=theta,
            ),
            likelihood_ratio=RatioStageConfig.from_mapping(section["likelihood_ratio"]),
            normalizer=LogNormalizerStageConfig.from_mapping(section["normalizer"]),
            output_directory=output,
            defensive_epsilon=float(section.get("defensive_epsilon", 0.0)),
            source_provenance=(
                source_provenance
                or {
                    "datasets": dict(section.get("datasets", {})),
                    "design_distributions": dict(
                        section.get("design_distributions", {})
                    ),
                }
            ),
            config_provenance=provenance,
            ratio_backends=ratio_backends,
        )

    def _flow_stage_config(self, artifact_name: str) -> ConditionalFlowStageConfig:
        if artifact_name == "q_phi":
            return self.posterior_flow
        if artifact_name == "q_eta":
            return self.likelihood_flow
        raise ValueError(f"Unknown conditional density {artifact_name!r}.")

    def train_conditional_density(
        self,
        target: np.ndarray,
        context: np.ndarray,
        *,
        artifact_name: str,
        group_ids: np.ndarray,
        validation: ProposalDataset | None = None,
    ) -> NativeConditionalDensity:
        config = self._flow_stage_config(artifact_name)
        target = as_2d(target, "target")
        context = as_2d(context, "context")
        groups = np.asarray(group_ids).reshape(-1)
        if len(target) != len(context) or len(groups) != len(target):
            raise ValueError("target, context, and group_ids must be row-aligned.")
        if config.max_events is not None and len(target) > config.max_events:
            selected = np.random.default_rng(config.seed).choice(
                len(target), size=config.max_events, replace=False
            )
            target = target[selected]
            context = context[selected]
            groups = groups[selected]
        validation_subset = None
        holdout_subset = None
        if validation is not None:
            if validation.split_values is None:
                validation_subset = validation
            else:
                validation_subset = validation.split_subset("validation")
                holdout_subset = validation.split_subset("holdout")
        if validation_subset is not None:
            if artifact_name == "q_phi":
                validation_target = validation_subset.theta
                validation_context = validation_subset.observation
            else:
                validation_target = validation_subset.observation
                validation_context = validation_subset.theta
        else:
            validation_target = None
            validation_context = None
        training_config = config.training
        if training_config.device == "auto":
            training_config = replace(
                training_config, device=_device(training_config.device)
            )
        result = FlowTrainer(config.flow, training_config).fit(
            target,
            features=config.target_features,
            context=context,
            context_names=config.context_features,
            validation_values=validation_target,
            validation_context=validation_context,
            seed=config.seed,
        )
        directory = self.output_directory / artifact_name
        checkpoint, _ = result.save_checkpoint(directory / f"{artifact_name}.pt")
        parity_dataset = holdout_subset or validation_subset
        parity_split = (
            "holdout"
            if holdout_subset is not None
            else "validation"
            if validation_subset is not None
            else "train"
        )
        if parity_dataset is None:
            parity_target = target[: min(256, len(target))]
            parity_context = context[: len(parity_target)]
        elif artifact_name == "q_phi":
            parity_target = parity_dataset.theta[:256]
            parity_context = parity_dataset.observation[:256]
        else:
            parity_target = parity_dataset.observation[:256]
            parity_context = parity_dataset.theta[:256]
        bundle = result.flow.export_onnx(
            directory,
            prefix=artifact_name,
            example_values=parity_target[: min(32, len(parity_target))],
            example_context=parity_context[: min(32, len(parity_context))],
            opset_version=config.onnx_opset,
        )
        parity = bundle.parity(
            result.flow,
            parity_target,
            context=parity_context,
        )
        for report in parity.values():
            report.assert_close()
        parity_path = directory / "onnx_parity.json"
        parity_path.write_text(
            json.dumps(
                {key: report.to_dict() for key, report in parity.items()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact_manifest = directory / "artifact.manifest.json"
        write_artifact_manifest(
            artifact_manifest,
            artifact_type=f"dual-{artifact_name}-conditional-flow",
            files={
                "native-checkpoint": checkpoint,
                "log-prob-onnx": bundle.log_prob_path,
                "inverse-onnx": bundle.base_to_data_path,
                "forward-onnx": bundle.data_to_base_path,
                "onnx-parity": parity_path,
            },
            metadata={
                "artifact_name": artifact_name,
                "context_features": list(config.context_features),
                "flow": asdict(config.flow),
                "parity_split": parity_split,
                "split_counts": {
                    "train": len(target),
                    "validation": (
                        0 if validation_subset is None else len(validation_subset.theta)
                    ),
                    "holdout": (
                        0 if holdout_subset is None else len(holdout_subset.theta)
                    ),
                },
                "target_features": list(config.target_features),
            },
        )
        root = self.output_directory
        if artifact_name == "q_phi":
            log_inputs = {"theta": "features", "x": "context"}
            inverse_inputs = {"base_noise": "base", "x": "context"}
            inverse_outputs = {"theta": "features"}
        else:
            log_inputs = {"x": "features", "theta": "context"}
            inverse_inputs = {"base_noise": "base", "theta": "context"}
            inverse_outputs = {"x": "features"}
        graphs = (
            OnnxGraphSpec.from_file(
                bundle.log_prob_path,
                root=root,
                role="log_prob",
                inputs=log_inputs,
                outputs={"log_prob": "log_prob"},
                opset=config.onnx_opset,
            ),
            OnnxGraphSpec.from_file(
                bundle.base_to_data_path,
                root=root,
                role="inverse",
                inputs=inverse_inputs,
                outputs=inverse_outputs,
                opset=config.onnx_opset,
            ),
        )
        identity = TransformSpec.identity()
        self._specifications[artifact_name] = DualArtifactSpec(
            name=artifact_name,
            graphs=graphs,
            transforms={"theta": identity, "x": identity},
            ensemble=EnsembleSpec("single", 1),
            base_distribution="standard_normal",
            metadata={
                "artifact_manifest": str(
                    artifact_manifest.relative_to(self.output_directory)
                ),
                "onnx_parity": {
                    key: report.to_dict() for key, report in parity.items()
                },
            },
        )
        self.stages[artifact_name] = FlowStageArtifacts(
            training=result,
            checkpoint_path=checkpoint,
            onnx_bundle=bundle,
            parity=parity,
            artifact_manifest_path=artifact_manifest,
        )
        return NativeConditionalDensity(result.flow)

    def _configured_ratio_backend(
        self,
        artifact_name: str,
        config: RatioStageConfig,
        validation_pairs: PairedClassifierDataset,
    ) -> RatioTrainingBackend:
        if artifact_name in self.ratio_backends:
            return self.ratio_backends[artifact_name]
        if config.backend == "native":
            return NativeTorchRatioBackend(
                theta_features=self.theta_features,
                observation_features=self.observation_features,
                artifact_name=artifact_name,
                validation_pairs=validation_pairs,
                opset_version=config.onnx_opset,
            )
        from ..integrations import NsbiCommonUtilsBackend

        return NsbiCommonUtilsBackend()

    def _ratio_graph(
        self,
        member: RatioBackendResult,
        *,
        artifact_name: str,
        member_index: int,
        config: RatioStageConfig,
        directory: Path,
    ) -> tuple[Path, OnnxParityReport]:
        native = member.files.get("dual-log-ratio-onnx")
        if native is not None:
            graph = Path(native)
        else:
            graph = directory / f"member_{member_index:03d}.log_ratio.onnx"
            _fuse_established_ratio_graph(
                member,
                output_path=graph,
                n_theta=len(self.theta_features),
                n_x=len(self.observation_features),
                use_log_loss=config.training.use_log_loss,
                opset_version=config.onnx_opset,
            )
        rows = np.concatenate(
            [
                self._ratio_parity_pairs.positive,
                self._ratio_parity_pairs.negative,
            ],
            axis=0,
        )
        rows = rows[: min(256, len(rows))]
        portable = (
            OnnxRunner(graph)
            .run(
                {
                    "theta": rows[:, : len(self.theta_features)],
                    "x": rows[:, len(self.theta_features) :],
                }
            )["log_ratio"]
            .reshape(-1)
        )
        if hasattr(member.evaluator, "log_ratio"):
            expected = member.evaluator.log_ratio(rows)
        elif config.training.use_log_loss and hasattr(member.evaluator, "raw_output"):
            expected = member.evaluator.raw_output(rows)
        else:
            expected = np.log(
                np.maximum(
                    np.asarray(member.evaluator(rows)),
                    np.finfo(np.float64).tiny,
                )
            )
        parity = compare_outputs(expected, portable)
        parity.assert_close()
        return graph, parity

    def train_log_ratio(
        self,
        pairs: PairedClassifierDataset,
        *,
        artifact_name: str,
        validation: PairedClassifierDataset | None = None,
    ) -> NativeLogRatio:
        if artifact_name == "r_p":
            config = self.posterior_ratio
        elif artifact_name == "r_c":
            config = self.likelihood_ratio
        else:
            raise ValueError(f"Unknown ratio artifact {artifact_name!r}.")
        if pairs.split_values is not None and np.any(pairs.split_values != "train"):
            raise ValueError(
                f"{artifact_name} fitting data may contain only 'train' rows."
            )
        if validation is None:
            training_pairs, validation_pairs, holdout_pairs = (
                _paired_train_validation_holdout(
                    pairs,
                    validation_fraction=config.validation_fraction,
                    holdout_fraction=config.training.holdout_fraction,
                    seed=config.training.seed,
                )
            )
            split_source = "random_group_split"
        else:
            if validation.shared_quantity != pairs.shared_quantity:
                raise ValueError(
                    f"{artifact_name} validation pairs use the wrong shared quantity."
                )
            if validation.split_values is None:
                validation_pairs = validation
                holdout_pairs = validation
            else:
                validation_pairs = validation.split_subset("validation")
                holdout_pairs = validation.split_subset("holdout")
            if validation_pairs is None:
                training_pairs, validation_pairs = _paired_train_validation(
                    pairs,
                    validation_fraction=config.validation_fraction,
                    seed=config.training.seed,
                )
            else:
                training_pairs = pairs
            if holdout_pairs is None:
                holdout_pairs = validation_pairs
            split_source = "configured"
        assert validation_pairs is not None
        assert holdout_pairs is not None
        holdout_reuses_validation = holdout_pairs is validation_pairs
        self._ratio_parity_pairs = holdout_pairs
        directory = self.output_directory / artifact_name
        backend = self._configured_ratio_backend(
            artifact_name, config, validation_pairs
        )
        result = RatioTrainer(backend, config.training).fit(
            training_pairs.positive,
            training_pairs.negative,
            features=self.theta_features + self.observation_features,
            output_directory=directory / "training",
            numerator_name=f"{artifact_name}_joint",
            denominator_name=f"{artifact_name}_reference",
        )
        graphs: list[Path] = []
        parity: dict[int, OnnxParityReport] = {}
        for index, member in enumerate(result.members):
            graph, report = self._ratio_graph(
                member,
                artifact_name=artifact_name,
                member_index=index,
                config=config,
                directory=directory,
            )
            graphs.append(graph)
            parity[index] = report
        parity_path = directory / "onnx_parity.json"
        parity_path.write_text(
            json.dumps(
                {str(index): report.to_dict() for index, report in parity.items()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        validation_report: dict[str, Any] = {
            "holdout_reuses_validation": holdout_reuses_validation,
            "split_source": split_source,
            "split_counts": {
                "train": len(training_pairs.group_ids),
                "validation": len(validation_pairs.group_ids),
                "holdout": len(holdout_pairs.group_ids),
            },
        }
        for label, selected_pairs in (
            ("validation", validation_pairs),
            ("holdout", holdout_pairs),
        ):
            positive_log_ratio = result.ensemble.log_ratio(selected_pairs.positive)
            negative_log_ratio = result.ensemble.log_ratio(selected_pairs.negative)
            validation_report[label] = {
                "balanced_logistic_loss": float(
                    0.5
                    * (
                        np.mean(np.logaddexp(0.0, -positive_log_ratio))
                        + np.mean(np.logaddexp(0.0, negative_log_ratio))
                    )
                ),
                "classification_accuracy": float(
                    0.5
                    * (
                        np.mean(positive_log_ratio >= 0.0)
                        + np.mean(negative_log_ratio < 0.0)
                    )
                ),
                "negative_mean_log_ratio": float(np.mean(negative_log_ratio)),
                "positive_mean_log_ratio": float(np.mean(positive_log_ratio)),
            }
        validation_path = directory / "validation.json"
        validation_path.write_text(
            json.dumps(validation_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_manifest = directory / "artifact.manifest.json"
        files: dict[str, Path] = {
            f"log-ratio-onnx-{index:03d}": path for index, path in enumerate(graphs)
        }
        files["onnx-parity"] = parity_path
        files["training-manifest"] = result.manifest_path
        files["validation-report"] = validation_path
        write_artifact_manifest(
            artifact_manifest,
            artifact_type=f"dual-{artifact_name}-ratio-ensemble",
            files=files,
            metadata={
                "artifact_name": artifact_name,
                "backend": result.backend,
                "ensemble_reduction": "arithmetic_mean_ratio",
                "normalization": config.normalization,
                "holdout_reuses_validation": holdout_reuses_validation,
                "split_counts": validation_report["split_counts"],
                "split_source": split_source,
                "feature_order": list(self.theta_features + self.observation_features),
            },
        )
        identity = TransformSpec.identity()
        self._specifications[artifact_name] = DualArtifactSpec(
            name=artifact_name,
            graphs=tuple(
                OnnxGraphSpec.from_file(
                    path,
                    root=self.output_directory,
                    role="log_ratio",
                    inputs={"theta": "theta", "x": "x"},
                    outputs={"log_ratio": "log_ratio"},
                    member=index,
                    opset=config.onnx_opset,
                )
                for index, path in enumerate(graphs)
            ),
            transforms={"theta": identity, "x": identity},
            ensemble=EnsembleSpec("arithmetic_mean_ratio", len(graphs)),
            metadata={
                "artifact_manifest": str(
                    artifact_manifest.relative_to(self.output_directory)
                ),
                "feature_order": list(self.theta_features + self.observation_features),
                "normalization": config.normalization,
                "onnx_parity": {
                    str(index): report.to_dict() for index, report in parity.items()
                },
            },
        )
        self.stages[artifact_name] = RatioStageArtifacts(
            training=result,
            parity=parity,
            graph_paths=tuple(graphs),
            artifact_manifest_path=artifact_manifest,
        )
        return NativeLogRatio(result.ensemble, artifact_name)

    def train_log_normalizer(
        self,
        q_eta: NativeConditionalDensity,
        r_c: NativeLogRatio,
        context: np.ndarray,
        *,
        artifact_name: str,
        validation: ProposalDataset | None = None,
    ) -> TorchLogNormalizer:
        if artifact_name != "z_c":
            raise ValueError("The native normalizer artifact must be named 'z_c'.")
        config = self.normalizer
        if config.device == "auto":
            config = replace(config, device=_device(config.device))
        rng = np.random.default_rng(config.seed)
        contexts = _select_contexts(context, count=config.contexts, rng=rng)

        def estimate_targets(selected_contexts: np.ndarray) -> np.ndarray:
            result = np.empty(len(selected_contexts), dtype=np.float64)
            context_batch = max(
                1,
                min(
                    len(selected_contexts),
                    max(1, 65_536 // config.reference_draws_per_context),
                ),
            )
            for start in range(0, len(selected_contexts), context_batch):
                selected = selected_contexts[start : start + context_batch]
                draws = q_eta.sample(
                    config.reference_draws_per_context,
                    context=selected,
                    rng=rng,
                )
                flat_draws = draws.reshape(-1, draws.shape[-1])
                flat_contexts = np.repeat(
                    selected, config.reference_draws_per_context, axis=0
                )
                log_ratios = r_c.log_ratio(flat_draws, context=flat_contexts).reshape(
                    len(selected), config.reference_draws_per_context
                )
                result[start : start + len(selected)] = logmeanexp(log_ratios, axis=1)
            return result

        log_targets = estimate_targets(contexts)
        validation_subset = None
        holdout_subset = None
        if validation is not None:
            if validation.split_values is None:
                validation_subset = validation
            else:
                validation_subset = validation.split_subset("validation")
                holdout_subset = validation.split_subset("holdout")
        validation_contexts = (
            None
            if validation_subset is None
            else _select_contexts(
                validation_subset.theta,
                count=min(config.contexts, len(validation_subset.theta)),
                rng=rng,
            )
        )
        validation_targets = (
            None
            if validation_contexts is None
            else estimate_targets(validation_contexts)
        )
        holdout_contexts = (
            None
            if holdout_subset is None
            else _select_contexts(
                holdout_subset.theta,
                count=min(config.contexts, len(holdout_subset.theta)),
                rng=rng,
            )
        )
        holdout_targets = (
            None if holdout_contexts is None else estimate_targets(holdout_contexts)
        )
        model, history = _fit_log_normalizer_mlp(
            contexts,
            log_targets,
            config=config,
            validation_contexts=validation_contexts,
            validation_log_targets=validation_targets,
        )
        evaluator = TorchLogNormalizer(model, config.device)
        directory = self.output_directory / artifact_name
        directory.mkdir(parents=True, exist_ok=True)
        history_path = directory / "training_history.json"
        history_path.write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validation_report: dict[str, Any] = {
            "split_counts": {
                "train": len(contexts),
                "validation": (
                    0 if validation_contexts is None else len(validation_contexts)
                ),
                "holdout": 0 if holdout_contexts is None else len(holdout_contexts),
            },
            "split_source": (
                "configured" if validation_contexts is not None else "random"
            ),
        }
        for label, selected_contexts, selected_targets in (
            ("validation", validation_contexts, validation_targets),
            ("holdout", holdout_contexts, holdout_targets),
        ):
            if selected_contexts is None or selected_targets is None:
                continue
            residual = evaluator.log_normalization(selected_contexts) - selected_targets
            validation_report[label] = {
                "bias": float(np.mean(residual)),
                "max_absolute_error": float(np.max(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            }
        validation_path = directory / "validation.json"
        validation_path.write_text(
            json.dumps(validation_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        torch = require_optional("torch", extra="bayes", purpose="exporting log Z_C")
        parity_contexts = (
            holdout_contexts
            if holdout_contexts is not None
            else validation_contexts
            if validation_contexts is not None
            else contexts
        )
        parity_split = (
            "holdout"
            if holdout_contexts is not None
            else "validation"
            if validation_contexts is not None
            else "train"
        )
        rows = min(64, len(parity_contexts))
        graph_path = directory / "log_normalization.onnx"
        export_torch_onnx(
            model,
            torch.as_tensor(
                parity_contexts[:rows], dtype=torch.float32, device=config.device
            ),
            graph_path,
            input_names=["theta"],
            output_names=["log_normalization"],
            artifact_type="dual-log-normalization-onnx",
            metadata={
                "contexts": config.contexts,
                "parity_split": parity_split,
                "reference_draws_per_context": (config.reference_draws_per_context),
                "theta_features": list(self.theta_features),
            },
            opset_version=config.onnx_opset,
        )
        portable = (
            OnnxRunner(graph_path)
            .run({"theta": parity_contexts[:rows]})["log_normalization"]
            .reshape(-1)
        )
        parity = compare_outputs(
            evaluator.log_normalization(parity_contexts[:rows]), portable
        )
        parity.assert_close()
        parity_path = directory / "onnx_parity.json"
        parity_path.write_text(
            json.dumps(parity.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        targets_path = directory / "mc_targets.npz"
        np.savez_compressed(
            targets_path,
            theta=contexts,
            log_z_c=log_targets,
            validation_theta=(
                np.empty((0, contexts.shape[1]))
                if validation_contexts is None
                else validation_contexts
            ),
            validation_log_z_c=(
                np.empty(0) if validation_targets is None else validation_targets
            ),
            holdout_theta=(
                np.empty((0, contexts.shape[1]))
                if holdout_contexts is None
                else holdout_contexts
            ),
            holdout_log_z_c=(
                np.empty(0) if holdout_targets is None else holdout_targets
            ),
        )
        artifact_manifest = directory / "artifact.manifest.json"
        write_artifact_manifest(
            artifact_manifest,
            artifact_type="dual-z-c-log-normalizer",
            files={
                "log-normalization-onnx": graph_path,
                "mc-targets": targets_path,
                "onnx-parity": parity_path,
                "training-history": history_path,
                "validation-report": validation_path,
            },
            metadata={
                "contexts": config.contexts,
                "parity_split": parity_split,
                "reference_draws_per_context": (config.reference_draws_per_context),
                "split_counts": validation_report["split_counts"],
                "split_source": validation_report["split_source"],
                "theta_features": list(self.theta_features),
            },
        )
        self._specifications[artifact_name] = DualArtifactSpec(
            name=artifact_name,
            graphs=(
                OnnxGraphSpec.from_file(
                    graph_path,
                    root=self.output_directory,
                    role="log_normalization",
                    inputs={"theta": "theta"},
                    outputs={"log_normalization": "log_normalization"},
                    opset=config.onnx_opset,
                ),
            ),
            transforms={"theta": TransformSpec.identity()},
            ensemble=EnsembleSpec("single", 1),
            metadata={
                "artifact_manifest": str(
                    artifact_manifest.relative_to(self.output_directory)
                ),
                "mc_contexts": config.contexts,
                "mc_draws_per_context": config.reference_draws_per_context,
                "onnx_parity": parity.to_dict(),
            },
        )
        self.stages[artifact_name] = NormalizerStageArtifacts(
            model=evaluator,
            contexts=contexts,
            log_targets=log_targets,
            graph_path=graph_path,
            parity=parity,
            artifact_manifest_path=artifact_manifest,
        )
        self._finalize_manifest()
        return evaluator

    def _finalize_manifest(self) -> None:
        expected = {"q_phi", "r_p", "q_eta", "r_c", "z_c"}
        if set(self._specifications) != expected:
            missing = sorted(expected.difference(self._specifications))
            raise RuntimeError(
                f"Cannot finalize dual manifest; missing artifacts {missing}."
            )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self._manifest = create_dual_artifact_manifest(
            self._manifest_path,
            x_signature=FeatureSignature(self.observation_features),
            theta_signature=FeatureSignature(self.theta_features),
            artifacts=self._specifications,
            source_provenance=self.source_provenance,
            config_provenance=self.config_provenance,
            posterior_ratio_reference=(
                "defensive" if self.defensive_epsilon > 0.0 else "flow"
            ),
            defensive_epsilon=self.defensive_epsilon,
            metadata={
                "artifact_manifests": {
                    name: str(
                        stage.artifact_manifest_path.relative_to(self.output_directory)
                    )
                    for name, stage in self.stages.items()
                },
                "normalizer_mc": {
                    "contexts": self.normalizer.contexts,
                    "draws_per_context": (self.normalizer.reference_draws_per_context),
                },
                "ratio_normalization": {
                    "r_p": self.posterior_ratio.normalization,
                    "r_c": self.likelihood_ratio.normalization,
                },
            },
        )


def _fit_log_normalizer_mlp(
    contexts: np.ndarray,
    log_targets: np.ndarray,
    *,
    config: LogNormalizerStageConfig,
    validation_contexts: np.ndarray | None = None,
    validation_log_targets: np.ndarray | None = None,
) -> tuple[Any, list[dict[str, float | int]]]:
    torch = require_optional("torch", extra="bayes", purpose="training log Z_C")
    contexts = as_2d(contexts, "normalizer contexts").astype(np.float32)
    targets = np.asarray(log_targets, dtype=np.float32).reshape(-1)
    if len(targets) != len(contexts):
        raise ValueError("Normalizer targets must align with their contexts.")
    rng = np.random.default_rng(config.seed)
    if validation_contexts is None:
        if validation_log_targets is not None:
            raise ValueError("validation_log_targets requires validation_contexts.")
        order = rng.permutation(len(contexts))
        validation_count = min(
            len(contexts) - 1,
            max(1, int(round(config.validation_fraction * len(contexts)))),
        )
        validation_indices = order[:validation_count]
        training_indices = order[validation_count:]
    else:
        external_contexts = as_2d(
            validation_contexts, "normalizer validation contexts"
        ).astype(np.float32)
        if external_contexts.shape[1] != contexts.shape[1]:
            raise ValueError("Normalizer validation contexts have the wrong dimension.")
        if validation_log_targets is None:
            raise ValueError("validation_contexts requires validation_log_targets.")
        external_targets = np.asarray(validation_log_targets, dtype=np.float32).reshape(
            -1
        )
        if len(external_targets) != len(external_contexts) or not len(
            external_contexts
        ):
            raise ValueError(
                "Normalizer validation contexts/targets must be non-empty "
                "and row-aligned."
            )
        training_rows = len(contexts)
        contexts = np.concatenate([contexts, external_contexts], axis=0)
        targets = np.concatenate([targets, external_targets])
        training_indices = np.arange(training_rows, dtype=np.int64)
        validation_indices = np.arange(training_rows, len(contexts), dtype=np.int64)
    if not len(training_indices):
        raise ValueError("Normalizer validation split leaves no training contexts.")
    mean = np.mean(contexts[training_indices], axis=0, dtype=np.float64).astype(
        np.float32
    )
    scale = np.std(contexts[training_indices], axis=0, dtype=np.float64).astype(
        np.float32
    )
    scale = np.maximum(scale, np.float32(1.0e-6))
    target_mean = np.float32(np.mean(targets[training_indices], dtype=np.float64))
    target_scale = np.float32(
        max(float(np.std(targets[training_indices], dtype=np.float64)), 1.0e-6)
    )

    class EmbeddedNormalizer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "input_mean", torch.as_tensor(mean, dtype=torch.float32)
            )
            self.register_buffer(
                "input_scale", torch.as_tensor(scale, dtype=torch.float32)
            )
            self.register_buffer(
                "target_mean",
                torch.as_tensor(target_mean, dtype=torch.float32),
            )
            self.register_buffer(
                "target_scale",
                torch.as_tensor(target_scale, dtype=torch.float32),
            )
            layers: list[Any] = []
            width = contexts.shape[1]
            for _ in range(config.hidden_layers):
                layers.extend(
                    [
                        torch.nn.Linear(width, config.hidden_features),
                        torch.nn.SiLU(),
                    ]
                )
                width = config.hidden_features
            layers.append(torch.nn.Linear(width, 1))
            self.network = torch.nn.Sequential(*layers)

        def forward(self, theta: Any) -> Any:
            normalized = (theta - self.input_mean) / self.input_scale
            prediction = self.network(normalized).reshape(-1)
            return prediction * self.target_scale + self.target_mean

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = EmbeddedNormalizer().to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1.0e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.learning_rate_factor,
        patience=max(1, config.patience // 3),
    )
    x = torch.as_tensor(contexts, dtype=torch.float32, device=config.device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=config.device)
    training_index_tensor = torch.as_tensor(
        training_indices, dtype=torch.long, device=config.device
    )
    validation_index_tensor = torch.as_tensor(
        validation_indices, dtype=torch.long, device=config.device
    )
    best_state: dict[str, Any] | None = None
    best_loss = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        shuffled = rng.permutation(training_indices)
        for start in range(0, len(shuffled), config.batch_size):
            indices = torch.as_tensor(
                shuffled[start : start + config.batch_size],
                dtype=torch.long,
                device=config.device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x[indices]), y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            training_loss = float(
                torch.nn.functional.mse_loss(
                    model(x[training_index_tensor]), y[training_index_tensor]
                )
            )
            validation_loss = float(
                torch.nn.functional.mse_loss(
                    model(x[validation_index_tensor]),
                    y[validation_index_tensor],
                )
            )
        scheduler.step(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1.0e-7:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is None:  # pragma: no cover - at least one epoch is required
        raise RuntimeError("Normalizer training produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def _fuse_established_ratio_graph(
    member: RatioBackendResult,
    *,
    output_path: Path,
    n_theta: int,
    n_x: int,
    use_log_loss: bool,
    opset_version: int,
) -> None:
    """Fuse upstream scaler/classifier graphs into the dual graph contract."""

    onnx = require_optional(
        "onnx",
        extra="bayes",
        purpose="fusing established ratio ONNX graphs",
    )
    scaler_path = member.files.get("scaler-onnx")
    classifier_path = member.files.get("classifier-onnx")
    if scaler_path is None or classifier_path is None:
        raise ValueError(
            "An established ratio member must expose scaler-onnx and "
            "classifier-onnx files for portable dual inference."
        )
    scaler = onnx.load(str(scaler_path))
    classifier = onnx.compose.add_prefix(onnx.load(str(classifier_path)), "classifier/")
    scaler_input = scaler.graph.input[0].name
    scaler_output = scaler.graph.output[0].name
    classifier_input = classifier.graph.input[0].name
    merged = onnx.compose.merge_models(
        scaler,
        classifier,
        io_map=[(scaler_output, classifier_input)],
    )
    raw_output = merged.graph.output[0].name
    del merged.graph.input[:]
    merged.graph.input.extend(
        [
            onnx.helper.make_tensor_value_info(
                "theta", onnx.TensorProto.FLOAT, ["batch", n_theta]
            ),
            onnx.helper.make_tensor_value_info(
                "x", onnx.TensorProto.FLOAT, ["batch", n_x]
            ),
        ]
    )
    merged.graph.node.insert(
        0,
        onnx.helper.make_node("Concat", ["theta", "x"], [scaler_input], axis=1),
    )
    del merged.graph.output[:]
    if use_log_loss:
        merged.graph.node.append(
            onnx.helper.make_node("Identity", [raw_output], ["log_ratio"])
        )
    else:
        minimum = onnx.numpy_helper.from_array(
            np.asarray(1.0e-9, dtype=np.float32), name="ratio_score_minimum"
        )
        maximum = onnx.numpy_helper.from_array(
            np.asarray(
                np.nextafter(np.float32(1.0), np.float32(0.0)),
                dtype=np.float32,
            ),
            name="ratio_score_maximum",
        )
        one = onnx.numpy_helper.from_array(
            np.asarray(1.0, dtype=np.float32), name="ratio_score_one"
        )
        merged.graph.initializer.extend([minimum, maximum, one])
        merged.graph.node.extend(
            [
                onnx.helper.make_node(
                    "Clip",
                    [raw_output, minimum.name, maximum.name],
                    ["clipped_score"],
                ),
                onnx.helper.make_node(
                    "Sub", [one.name, "clipped_score"], ["one_minus_score"]
                ),
                onnx.helper.make_node("Log", ["clipped_score"], ["log_score"]),
                onnx.helper.make_node(
                    "Log", ["one_minus_score"], ["log_one_minus_score"]
                ),
                onnx.helper.make_node(
                    "Sub",
                    ["log_score", "log_one_minus_score"],
                    ["log_ratio"],
                ),
            ]
        )
    merged.graph.output.extend(
        [onnx.helper.make_tensor_value_info("log_ratio", onnx.TensorProto.FLOAT, None)]
    )
    for opset in merged.opset_import:
        if not opset.domain:
            opset.version = max(int(opset.version), int(opset_version))
    onnx.checker.check_model(merged)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(merged, str(output_path))
    write_artifact_manifest(
        output_path.with_suffix(".onnx.manifest.json"),
        artifact_type="dual-fused-established-log-ratio-onnx",
        files={"onnx-model": output_path},
        metadata={
            "n_theta": n_theta,
            "n_x": n_x,
            "opset_version": opset_version,
            "source_backend": "nsbi_common_utils",
        },
    )


def train_native_dual(
    data: DualTrainingData,
    *,
    rho: Any,
    bayesian_config: Mapping[str, Any],
    observation_features: Sequence[str],
    output_directory: str | Path | None = None,
    base_directory: str | Path = ".",
    seed: int | None = None,
    source_provenance: Mapping[str, Any] | None = None,
    ratio_backends: Mapping[str, RatioTrainingBackend] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> NativeDualTrainingArtifacts:
    """Train and package the complete five-artifact dual model."""

    backend = NativeDualBackend.from_config(
        bayesian_config,
        observation_features=observation_features,
        output_directory=output_directory,
        base_directory=base_directory,
        source_provenance=source_provenance,
        ratio_backends=ratio_backends,
    )
    default_seed = int(bayesian_config["posterior_flow"]["training"].get("seed", 0))
    model = DualTrainer(
        backend=backend,
        seed=default_seed if seed is None else int(seed),
    ).fit(
        data,
        rho=rho,
        defensive_epsilon=float(bayesian_config.get("defensive_epsilon", 0.0)),
        metadata=dict(metadata or {}),
    )
    return NativeDualTrainingArtifacts(
        model=model,
        manifest=backend.manifest,
        manifest_path=backend.manifest_path,
        stages=dict(backend.stages),
    )


__all__ = [
    "ConditionalFlowStageConfig",
    "FlowStageArtifacts",
    "LogNormalizerStageConfig",
    "NativeConditionalDensity",
    "NativeDualBackend",
    "NativeDualTrainingArtifacts",
    "NativeLogRatio",
    "NativeTorchRatioBackend",
    "NormalizerStageArtifacts",
    "RatioStageArtifacts",
    "RatioStageConfig",
    "TorchLogNormalizer",
    "train_native_dual",
]
