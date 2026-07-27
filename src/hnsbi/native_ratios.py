"""Self-contained Torch density-ratio training and portable ONNX inference."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactManifest
from .diagnostics import json_safe
from .onnx import (
    OnnxRunner,
    compare_outputs,
    export_torch_onnx,
    require_optional,
)
from .ratio_diagnostics import RatioDiagnosticReport, diagnose_ratio
from .ratios import RatioBackendResult, RatioTrainingConfig, normalize_class_weights


def _activation(torch: Any, name: str) -> Any:
    normalized = str(name).lower()
    if normalized == "relu":
        return torch.nn.ReLU
    if normalized == "swish":
        return torch.nn.SiLU
    if normalized == "tanh":
        return torch.nn.Tanh
    if normalized == "gelu":
        return torch.nn.GELU
    raise ValueError(f"Unsupported native ratio activation {name!r}.")


def _build_log_ratio_module(
    torch: Any,
    *,
    mean: Any,
    scale: Any,
    hidden_layers: int,
    neurons: int,
    activation: str,
) -> Any:
    """Build the checkpoint-stable embedded-standardization classifier."""

    mean_array = np.asarray(mean, dtype=np.float32).reshape(-1)
    scale_array = np.asarray(scale, dtype=np.float32).reshape(-1)
    if (
        not len(mean_array)
        or len(mean_array) != len(scale_array)
        or not np.isfinite(mean_array).all()
        or not np.isfinite(scale_array).all()
        or np.any(scale_array <= 0)
    ):
        raise ValueError("Native ratio standardization must be finite and positive.")
    activation_type = _activation(torch, activation)

    class EmbeddedLogRatio(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mean", torch.as_tensor(mean_array))
            self.register_buffer("scale", torch.as_tensor(scale_array))
            layers: list[Any] = []
            width = len(mean_array)
            for _ in range(int(hidden_layers)):
                layers.extend(
                    [
                        torch.nn.Linear(width, int(neurons)),
                        activation_type(),
                    ]
                )
                width = int(neurons)
            layers.append(torch.nn.Linear(width, 1))
            self.network = torch.nn.Sequential(*layers)

        def forward(self, rows: Any) -> Any:
            return self.network((rows - self.mean) / self.scale).reshape(-1)

    return EmbeddedLogRatio()


@dataclass(frozen=True)
class _ClassSplit:
    training: np.ndarray
    validation: np.ndarray
    holdout: np.ndarray


_SCIENTIFIC_SPLIT_LABELS = ("train", "validation", "holdout")


def _split_indices(
    length: int,
    *,
    validation_fraction: float,
    holdout_fraction: float,
    rng: np.random.Generator,
) -> _ClassSplit:
    if length < 5:
        raise ValueError(
            "Native ratio training requires at least five events in each class."
        )
    n_holdout = max(1, int(round(length * holdout_fraction)))
    n_validation = max(1, int(round(length * validation_fraction)))
    if n_holdout + n_validation > length - 2:
        excess = n_holdout + n_validation - (length - 2)
        reduce_holdout = min(excess, n_holdout - 1)
        n_holdout -= reduce_holdout
        n_validation -= excess - reduce_holdout
    permutation = rng.permutation(length)
    return _ClassSplit(
        training=np.sort(permutation[n_holdout + n_validation :]),
        validation=np.sort(permutation[n_holdout : n_holdout + n_validation]),
        holdout=np.sort(permutation[:n_holdout]),
    )


def _aligned_vector(
    values: Any | None,
    *,
    length: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values).reshape(-1)
    if len(array) != length:
        raise ValueError(f"{name} must contain one value per event.")
    return array


def _split_labels(
    values: Any | None,
    *,
    length: int,
    name: str,
) -> np.ndarray | None:
    array = _aligned_vector(values, length=length, name=name)
    if array is None:
        return None
    labels = np.asarray(
        [(value.item() if isinstance(value, np.generic) else value) for value in array],
        dtype=object,
    )
    labels = np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in labels
        ],
        dtype=str,
    )
    invalid = sorted(set(labels).difference(_SCIENTIFIC_SPLIT_LABELS))
    if invalid:
        raise ValueError(
            f"{name} accepts exactly 'train', 'validation', or 'holdout'; "
            f"received {invalid}."
        )
    return labels


def _stable_group_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except TypeError:  # pragma: no cover - default=str handles normal objects
        return repr(value)


def _group_keys(
    values: Any | None,
    *,
    length: int,
    name: str,
) -> tuple[str, ...] | None:
    array = _aligned_vector(values, length=length, name=name)
    if array is None:
        return None
    return tuple(_stable_group_key(value) for value in array)


def _group_partition_counts(
    length: int,
    *,
    validation_fraction: float,
    holdout_fraction: float,
) -> dict[str, int]:
    if length < 3:
        raise ValueError(
            "Grouped native ratio training requires at least three distinct "
            "groups for train, validation, and holdout."
        )
    n_holdout = max(1, int(round(length * holdout_fraction)))
    n_validation = max(1, int(round(length * validation_fraction)))
    while n_holdout + n_validation >= length:
        if n_holdout > 1:
            n_holdout -= 1
        elif n_validation > 1:
            n_validation -= 1
        else:  # pragma: no cover - length >= 3 makes this unreachable
            raise RuntimeError("Could not allocate grouped ratio partitions.")
    return {
        "train": length - n_holdout - n_validation,
        "validation": n_validation,
        "holdout": n_holdout,
    }


def _allocate_group_partitions(
    keys: set[str],
    assignments: dict[str, str],
    *,
    validation_fraction: float,
    holdout_fraction: float,
    seed: int,
) -> None:
    """Fill deterministic group assignments while preserving explicit labels."""

    targets = _group_partition_counts(
        len(keys),
        validation_fraction=validation_fraction,
        holdout_fraction=holdout_fraction,
    )
    current = {
        label: sum(assignments.get(key) == label for key in keys)
        for label in _SCIENTIFIC_SPLIT_LABELS
    }
    unassigned = sorted(
        (key for key in keys if key not in assignments),
        key=lambda key: hashlib.sha256(f"{seed}\0{key}".encode()).digest(),
    )
    # Prefer filling the largest target deficit. The fixed tie order keeps the
    # assignment deterministic even when target counts are equal.
    priority = {"holdout": 2, "validation": 1, "train": 0}
    for key in unassigned:
        label = max(
            _SCIENTIFIC_SPLIT_LABELS,
            key=lambda candidate: (
                targets[candidate] - current[candidate],
                priority[candidate],
            ),
        )
        assignments[key] = label
        current[label] += 1


def _indices_from_labels(labels: np.ndarray) -> _ClassSplit:
    return _ClassSplit(
        training=np.flatnonzero(labels == "train"),
        validation=np.flatnonzero(labels == "validation"),
        holdout=np.flatnonzero(labels == "holdout"),
    )


def _validate_class_split(
    split: _ClassSplit,
    *,
    length: int,
    name: str,
) -> _ClassSplit:
    if length < 5:
        raise ValueError(
            "Native ratio training requires at least five events in each class."
        )
    missing = [
        label
        for label, indices in (
            ("train", split.training),
            ("validation", split.validation),
            ("holdout", split.holdout),
        )
        if not len(indices)
    ]
    if missing:
        raise ValueError(
            f"{name} has no rows in required partitions {missing}. "
            "Provide more representative groups or explicit split labels."
        )
    combined = np.concatenate([split.training, split.validation, split.holdout])
    if len(combined) != length or len(np.unique(combined)) != length:
        raise RuntimeError(
            f"{name} partitioning did not assign every row exactly once."
        )
    return split


def _resolve_class_splits(
    *,
    numerator_length: int,
    denominator_length: int,
    validation_fraction: float,
    holdout_fraction: float,
    rng: np.random.Generator,
    seed: int,
    numerator_split: Any | None = None,
    denominator_split: Any | None = None,
    numerator_groups: Any | None = None,
    denominator_groups: Any | None = None,
) -> tuple[_ClassSplit, _ClassSplit]:
    """Resolve explicit, grouped, or random splits for both classifier classes."""

    labels = {
        "numerator": _split_labels(
            numerator_split,
            length=numerator_length,
            name="numerator_split",
        ),
        "denominator": _split_labels(
            denominator_split,
            length=denominator_length,
            name="denominator_split",
        ),
    }
    group_keys = {
        "numerator": _group_keys(
            numerator_groups,
            length=numerator_length,
            name="numerator_groups",
        ),
        "denominator": _group_keys(
            denominator_groups,
            length=denominator_length,
            name="denominator_groups",
        ),
    }
    assignments: dict[str, str] = {}
    for class_name in ("numerator", "denominator"):
        class_labels = labels[class_name]
        class_groups = group_keys[class_name]
        if class_labels is None or class_groups is None:
            continue
        for group, label in zip(class_groups, class_labels, strict=True):
            previous = assignments.get(group)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Group {group!r} appears in both {previous!r} and "
                    f"{label!r} partitions."
                )
            assignments[group] = str(label)

    numerator_keys = (
        set() if group_keys["numerator"] is None else set(group_keys["numerator"])
    )
    denominator_keys = (
        set() if group_keys["denominator"] is None else set(group_keys["denominator"])
    )
    generated_classes = [
        name
        for name in ("numerator", "denominator")
        if labels[name] is None and group_keys[name] is not None
    ]
    # Allocate the smaller group set first. If one class is a subset of the
    # other, it therefore receives all required partitions before the larger
    # class fills any remaining deficits, while shared groups keep one label.
    generated_classes.sort(
        key=lambda name: len(
            numerator_keys if name == "numerator" else denominator_keys
        )
    )
    for class_name in generated_classes:
        _allocate_group_partitions(
            numerator_keys if class_name == "numerator" else denominator_keys,
            assignments,
            validation_fraction=validation_fraction,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )

    resolved: dict[str, _ClassSplit] = {}
    lengths = {
        "numerator": numerator_length,
        "denominator": denominator_length,
    }
    for class_name in ("numerator", "denominator"):
        class_labels = labels[class_name]
        class_groups = group_keys[class_name]
        if class_labels is not None:
            split = _indices_from_labels(class_labels)
        elif class_groups is not None:
            split = _indices_from_labels(
                np.asarray([assignments[group] for group in class_groups])
            )
        else:
            split = _split_indices(
                lengths[class_name],
                validation_fraction=validation_fraction,
                holdout_fraction=holdout_fraction,
                rng=rng,
            )
        resolved[class_name] = _validate_class_split(
            split,
            length=lengths[class_name],
            name=class_name,
        )
    return resolved["numerator"], resolved["denominator"]


def _configured_diagnostic_checks(
    config: RatioTrainingConfig,
) -> tuple[str, ...] | None:
    diagnostics = config.backend_options.get("diagnostics") or {}
    methods = diagnostics.get("methods")
    return None if methods is None else tuple(str(method) for method in methods)


@dataclass(frozen=True)
class PiecewiseLinearCalibrator:
    """Portable monotonic score calibration represented by JSON arrays."""

    x: np.ndarray
    y: np.ndarray
    score_clip: float = 1.0e-8

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=np.float64).reshape(-1)
        y = np.asarray(self.y, dtype=np.float64).reshape(-1)
        if (
            len(x) < 2
            or len(x) != len(y)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any(np.diff(x) < 0)
            or np.any((y < 0) | (y > 1))
        ):
            raise ValueError("Calibrator knots must be aligned, finite, and monotonic.")
        if not 0 < self.score_clip < 0.5:
            raise ValueError("score_clip must lie in (0, 0.5).")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __call__(self, scores: Any) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64)
        calibrated = np.interp(values, self.x, self.y)
        return np.clip(calibrated, self.score_clip, 1.0 - self.score_clip)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(
            json.dumps(
                {
                    "kind": "piecewise_linear_probability",
                    "score_clip": self.score_clip,
                    "x": self.x.tolist(),
                    "y": self.y.tolist(),
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> PiecewiseLinearCalibrator:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") != "piecewise_linear_probability":
            raise ValueError("Unsupported native ratio calibrator.")
        return cls(
            x=np.asarray(payload["x"]),
            y=np.asarray(payload["y"]),
            score_clip=float(payload.get("score_clip", 1.0e-8)),
        )


def _fit_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    kind: str,
    bins: int,
) -> PiecewiseLinearCalibrator:
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    if kind == "isotonic":
        sklearn = require_optional(
            "sklearn.isotonic",
            extra="lhc",
            purpose="isotonic ratio calibration",
        )
        fitted = sklearn.IsotonicRegression(
            y_min=1.0e-8,
            y_max=1.0 - 1.0e-8,
            out_of_bounds="clip",
        ).fit(scores, labels, sample_weight=weights)
        return PiecewiseLinearCalibrator(
            x=np.asarray(fitted.X_thresholds_, dtype=np.float64),
            y=np.asarray(fitted.y_thresholds_, dtype=np.float64),
        )
    if kind != "histogram":
        raise ValueError("calibration_type must be 'isotonic' or 'histogram'.")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    indices = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    estimates = np.empty(int(bins), dtype=np.float64)
    global_fraction = float(np.sum(weights * labels) / np.sum(weights))
    for index in range(int(bins)):
        selected = indices == index
        mass = float(np.sum(weights[selected]))
        estimates[index] = (
            float(np.sum(weights[selected] * labels[selected]) / mass)
            if mass > 0
            else global_fraction
        )
    # A cumulative maximum prevents finite-sample fluctuations from making
    # the probability map non-monotonic.
    estimates = np.maximum.accumulate(estimates)
    return PiecewiseLinearCalibrator(
        x=np.concatenate([[0.0], centers, [1.0]]),
        y=np.concatenate([[estimates[0]], estimates, [estimates[-1]]]),
    )


@dataclass
class NativeRatioEvaluator:
    """Evaluate a fitted Torch log-ratio network in physical feature space."""

    module: Any
    device: str = "cpu"
    calibrator: PiecewiseLinearCalibrator | None = None

    def log_ratio(self, values: Any) -> np.ndarray:
        torch = require_optional(
            "torch", extra="lhc", purpose="native density-ratio inference"
        )
        array = np.asarray(values, dtype=np.float32)
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
        self.module.eval()
        with torch.no_grad():
            result = self.module(tensor).detach().cpu().numpy().reshape(-1)
        if self.calibrator is None:
            return np.asarray(result, dtype=np.float64)
        score = 1.0 / (1.0 + np.exp(-np.clip(result, -80.0, 80.0)))
        calibrated = self.calibrator(score)
        return np.log(calibrated) - np.log1p(-calibrated)

    def torch_log_ratio(self, values: Any) -> Any:
        """Return a differentiable native log ratio in physical coordinates."""

        if self.calibrator is not None:
            raise RuntimeError(
                "Calibrated ratios are not differentiable through their JSON "
                "piecewise map; train the nominal FNF base with calibration=false."
            )
        torch = require_optional(
            "torch",
            extra="lhc",
            purpose="differentiable native density-ratio inference",
        )
        tensor = (
            values.to(device=self.device, dtype=torch.float32)
            if torch.is_tensor(values)
            else torch.as_tensor(values, dtype=torch.float32, device=self.device)
        )
        self.module.eval()
        return self.module(tensor).reshape(-1)

    def __call__(self, values: Any) -> np.ndarray:
        return np.exp(np.clip(self.log_ratio(values), -80.0, 80.0))


@dataclass
class OnnxNativeRatioMember:
    """Portable embedded-scaler ONNX ratio with optional JSON calibration."""

    model_path: Path
    calibrator_path: Path | None = None
    providers: tuple[str, ...] | None = None
    _runner: OnnxRunner = field(init=False, repr=False)
    _calibrator: PiecewiseLinearCalibrator | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.calibrator_path = (
            None if self.calibrator_path is None else Path(self.calibrator_path)
        )
        self._runner = OnnxRunner(self.model_path, providers=self.providers)
        self._calibrator = (
            None
            if self.calibrator_path is None
            else PiecewiseLinearCalibrator.load(self.calibrator_path)
        )

    def raw_log_ratio(self, values: Any) -> np.ndarray:
        outputs = self._runner.run(np.asarray(values, dtype=np.float32))
        result = np.asarray(next(iter(outputs.values())), dtype=np.float64).reshape(-1)
        if not np.isfinite(result).all():
            raise ValueError("Native ONNX ratio returned non-finite logits.")
        return result

    def log_ratio(self, values: Any) -> np.ndarray:
        result = self.raw_log_ratio(values)
        if self._calibrator is None:
            return result
        score = 1.0 / (1.0 + np.exp(-np.clip(result, -80.0, 80.0)))
        calibrated = self._calibrator(score)
        return np.log(calibrated) - np.log1p(-calibrated)

    def __call__(self, values: Any) -> np.ndarray:
        return np.exp(np.clip(self.log_ratio(values), -80.0, 80.0))


@dataclass
class NativeRatioBackend:
    """Weighted classifier backend replacing ``nsbi-lhc-toolkit`` training."""

    device: str = "cpu"
    opset_version: int = 17
    name: str = field(default="native", init=False)

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
        numerator_split: np.ndarray | None = None,
        denominator_split: np.ndarray | None = None,
        numerator_groups: np.ndarray | None = None,
        denominator_groups: np.ndarray | None = None,
    ) -> RatioBackendResult:
        torch = require_optional(
            "torch", extra="lhc", purpose="native density-ratio training"
        )
        rng = np.random.default_rng(config.seed + member_index)
        numerator_partition, denominator_partition = _resolve_class_splits(
            numerator_length=len(numerator_values),
            denominator_length=len(denominator_values),
            validation_fraction=config.validation_fraction,
            holdout_fraction=config.holdout_fraction,
            rng=rng,
            seed=config.seed + member_index,
            numerator_split=numerator_split,
            denominator_split=denominator_split,
            numerator_groups=numerator_groups,
            denominator_groups=denominator_groups,
        )

        def stacked(
            numerator_indices: np.ndarray,
            denominator_indices: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            values = np.concatenate(
                [
                    numerator_values[numerator_indices],
                    denominator_values[denominator_indices],
                ],
                axis=0,
            ).astype(np.float32)
            labels = np.concatenate(
                [
                    np.ones(len(numerator_indices), dtype=np.float32),
                    np.zeros(len(denominator_indices), dtype=np.float32),
                ]
            )
            weights = np.concatenate(
                [
                    0.5
                    * normalize_class_weights(
                        numerator_weights[numerator_indices],
                    ),
                    0.5
                    * normalize_class_weights(
                        denominator_weights[denominator_indices],
                    ),
                ]
            ).astype(np.float32)
            return values, labels, weights

        train_values, train_labels, train_weights = stacked(
            numerator_partition.training,
            denominator_partition.training,
        )
        validation_values, validation_labels, validation_weights = stacked(
            numerator_partition.validation,
            denominator_partition.validation,
        )
        mean = np.average(train_values, axis=0, weights=train_weights).astype(
            np.float32
        )
        variance = np.average(
            (train_values - mean) ** 2,
            axis=0,
            weights=train_weights,
        )
        scale = np.sqrt(np.maximum(variance, 1.0e-12)).astype(np.float32)

        torch.manual_seed(config.seed + member_index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed + member_index)
        model = _build_log_ratio_module(
            torch,
            mean=mean,
            scale=scale,
            hidden_layers=config.hidden_layers,
            neurons=config.neurons,
            activation=config.activation,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=float(config.backend_options.get("weight_decay", 1.0e-5)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=min(float(config.learning_rate_factor), 0.99),
            patience=max(1, config.patience // 3),
        )
        train_tensor = torch.as_tensor(train_values, device=self.device)
        train_label_tensor = torch.as_tensor(train_labels, device=self.device)
        train_weight_tensor = torch.as_tensor(train_weights, device=self.device)
        validation_tensor = torch.as_tensor(validation_values, device=self.device)
        validation_label_tensor = torch.as_tensor(validation_labels, device=self.device)
        validation_weight_tensor = torch.as_tensor(
            validation_weights, device=self.device
        )
        best_state: dict[str, Any] | None = None
        best_loss = math.inf
        without_improvement = 0
        history: list[dict[str, float | int]] = []
        for epoch in range(1, config.epochs + 1):
            model.train()
            permutation = rng.permutation(len(train_values))
            for start in range(0, len(train_values), config.batch_size):
                indices = torch.as_tensor(
                    permutation[start : start + config.batch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                optimizer.zero_grad(set_to_none=True)
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(train_tensor[indices]),
                    train_label_tensor[indices],
                    reduction="none",
                )
                selected_weights = train_weight_tensor[indices]
                loss = (losses * selected_weights).sum() / selected_weights.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                training_losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(train_tensor),
                    train_label_tensor,
                    reduction="none",
                )
                validation_losses = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        model(validation_tensor),
                        validation_label_tensor,
                        reduction="none",
                    )
                )
                training_loss = float(
                    (training_losses * train_weight_tensor).sum()
                    / train_weight_tensor.sum()
                )
                validation_loss = float(
                    (validation_losses * validation_weight_tensor).sum()
                    / validation_weight_tensor.sum()
                )
            scheduler.step(validation_loss)
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1.0e-6:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                without_improvement = 0
            else:
                without_improvement += 1
                if config.early_stopping and without_improvement >= config.patience:
                    break
        if best_state is None:  # pragma: no cover - epochs is validated positive
            raise RuntimeError("Native ratio training produced no checkpoint.")
        model.load_state_dict(best_state)
        model.eval()

        def logits(values: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                result = model(
                    torch.as_tensor(
                        np.asarray(values, dtype=np.float32),
                        device=self.device,
                    )
                )
            return result.detach().cpu().numpy().reshape(-1)

        calibrator: PiecewiseLinearCalibrator | None = None
        if config.calibration:
            calibrator = _fit_calibrator(
                logits(validation_values),
                validation_labels,
                validation_weights,
                kind=config.calibration_type,
                bins=config.calibration_bins,
            )
        evaluator = NativeRatioEvaluator(
            module=model,
            device=self.device,
            calibrator=calibrator,
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        history_path = output_directory / "training_history.json"
        history_path.write_text(
            json.dumps(history, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        split_path = output_directory / "split_provenance.json"
        split_path.write_text(
            json.dumps(
                {
                    "denominator": {
                        key: getattr(denominator_partition, key).tolist()
                        for key in ("training", "validation", "holdout")
                    },
                    "numerator": {
                        key: getattr(numerator_partition, key).tolist()
                        for key in ("training", "validation", "holdout")
                    },
                    "partitioning": {
                        "denominator_groups": denominator_groups is not None,
                        "denominator_split": denominator_split is not None,
                        "numerator_groups": numerator_groups is not None,
                        "numerator_split": numerator_split is not None,
                    },
                    "seed": config.seed + member_index,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint_path = output_directory / "ratio_state.pt"
        torch.save(
            {
                "config": asdict(config),
                "feature_order": features,
                "mean": mean.tolist(),
                "scale": scale.tolist(),
                "state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        calibration_path: Path | None = None
        if calibrator is not None:
            calibration_path = calibrator.write(output_directory / "calibration.json")
        rows = min(64, len(train_values))
        graph_path, graph_manifest = export_torch_onnx(
            model,
            torch.as_tensor(train_values[:rows], device=self.device),
            output_directory / "log_ratio.onnx",
            input_names=["features"],
            output_names=["log_ratio"],
            artifact_type="native-density-log-ratio-onnx",
            metadata={
                "calibration": (
                    None if calibration_path is None else calibration_path.name
                ),
                "denominator": denominator_name,
                "feature_order": list(features),
                "member": member_index,
                "numerator": numerator_name,
                "scaler_embedded": True,
            },
            opset_version=self.opset_version,
        )
        onnx_logits = (
            OnnxRunner(graph_path).run(train_values[:rows])["log_ratio"].reshape(-1)
        )
        parity = compare_outputs(logits(train_values[:rows]), onnx_logits)
        parity.assert_close()
        parity_path = output_directory / "onnx_parity.json"
        parity_path.write_text(
            json.dumps(parity.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        diagnostic_report: RatioDiagnosticReport | None = None
        diagnostic_path: Path | None = None
        diagnostic_manifest: Path | None = None
        if config.run_diagnostics:
            diagnostic_report = diagnose_ratio(
                numerator_train=numerator_values[numerator_partition.training],
                denominator_train=denominator_values[denominator_partition.training],
                numerator_holdout=numerator_values[numerator_partition.holdout],
                denominator_holdout=denominator_values[denominator_partition.holdout],
                numerator_train_weights=numerator_weights[numerator_partition.training],
                denominator_train_weights=denominator_weights[
                    denominator_partition.training
                ],
                numerator_holdout_weights=numerator_weights[
                    numerator_partition.holdout
                ],
                denominator_holdout_weights=denominator_weights[
                    denominator_partition.holdout
                ],
                train_ratios=(
                    evaluator(numerator_values[numerator_partition.training]),
                    evaluator(denominator_values[denominator_partition.training]),
                ),
                holdout_ratios=(
                    evaluator(numerator_values[numerator_partition.holdout]),
                    evaluator(denominator_values[denominator_partition.holdout]),
                ),
                features=features,
                history=history,
                bins=config.diagnostic_bins,
                output_directory=output_directory / "diagnostics",
                checks=_configured_diagnostic_checks(config),
            )
            diagnostic_path, diagnostic_manifest = diagnostic_report.write(
                output_directory / "diagnostics"
            )
        files = {
            "checkpoint": checkpoint_path,
            "log-ratio-onnx": graph_path,
            "log-ratio-onnx-manifest": graph_manifest,
            "onnx-parity": parity_path,
            "split-provenance": split_path,
            "training-history": history_path,
        }
        if calibration_path is not None:
            files["calibration"] = calibration_path
        if diagnostic_path is not None and diagnostic_manifest is not None:
            files["diagnostics"] = diagnostic_path
            files["diagnostics-manifest"] = diagnostic_manifest
        return RatioBackendResult(
            evaluator=evaluator,
            files=files,
            metadata={
                "best_validation_loss": best_loss,
                "calibration": config.calibration,
                "diagnostics": (
                    None
                    if diagnostic_report is None
                    else json_safe(diagnostic_report.metrics)
                ),
                "epochs": len(history),
                "onnx_parity": parity.to_dict(),
            },
        )


def _native_ratio_manifest(
    manifest_path: str | Path,
    *,
    expected_features: tuple[str, ...] | None,
) -> tuple[ArtifactManifest, tuple[str, ...], dict[int, dict[str, Path]]]:
    path = Path(manifest_path)
    manifest = ArtifactManifest.load(path)
    if manifest.artifact_type != "density-ratio-ensemble":
        raise ValueError(
            f"Expected a density-ratio-ensemble, found {manifest.artifact_type!r}."
        )
    manifest.verify(path.parent)
    if manifest.metadata.get("backend") != "native":
        raise ValueError("FNF nominal densities require native ratio artifacts.")
    features = tuple(manifest.metadata.get("features", ()))
    if not features:
        raise ValueError("Native ratio manifest has no feature signature.")
    if expected_features is not None and tuple(expected_features) != features:
        raise ValueError(
            f"Ratio feature order {features} does not match {tuple(expected_features)}."
        )
    member_files: dict[int, dict[str, Path]] = {}
    for record in manifest.files:
        if not record.kind.startswith("member-"):
            continue
        parts = record.kind.split("-", 2)
        if len(parts) != 3 or not parts[1].isdigit():
            raise ValueError(f"Malformed ratio member role {record.kind!r}.")
        member_files.setdefault(int(parts[1]), {})[parts[2]] = path.parent / record.path
    if not member_files or sorted(member_files) != list(range(len(member_files))):
        raise ValueError("Ratio ensemble member indices must be contiguous from zero.")
    return manifest, features, member_files


def load_native_ratio_ensemble(
    manifest_path: str | Path,
    *,
    providers: tuple[str, ...] | None = None,
    expected_features: tuple[str, ...] | None = None,
) -> Any:
    """Load a verified arithmetic ensemble from native ONNX member artifacts."""

    from .ratios import RatioEnsemble

    _, _, member_files = _native_ratio_manifest(
        manifest_path,
        expected_features=expected_features,
    )
    members = []
    for index in range(len(member_files)):
        files = member_files[index]
        if "log-ratio-onnx" not in files:
            raise ValueError(f"Ratio member {index} has no ONNX log-ratio graph.")
        members.append(
            OnnxNativeRatioMember(
                model_path=files["log-ratio-onnx"],
                calibrator_path=files.get("calibration"),
                providers=providers,
            )
        )
    return RatioEnsemble(members)


def load_native_ratio_torch_ensemble(
    manifest_path: str | Path,
    *,
    device: str = "cpu",
    expected_features: tuple[str, ...] | None = None,
    live_ensemble: Any | None = None,
) -> Any:
    """Load hash-verified Torch members and optionally authenticate a live ensemble.

    Supplying ``live_ensemble`` is an exact scientific-identity check, not an
    inference shortcut. The returned ensemble is always reconstructed from the
    checksummed checkpoints, preventing a residual flow from being trained on
    live ratio objects that disagree with its recorded artifact provenance.
    """

    from .ratios import RatioEnsemble, RatioTrainingConfig

    torch = require_optional(
        "torch",
        extra="lhc",
        purpose="loading differentiable native ratio artifacts",
    )
    manifest, features, member_files = _native_ratio_manifest(
        manifest_path,
        expected_features=expected_features,
    )
    manifest_config = manifest.metadata.get("config")
    loaded_members: list[NativeRatioEvaluator] = []
    for index in range(len(member_files)):
        files = member_files[index]
        checkpoint = files.get("checkpoint")
        if checkpoint is None:
            raise ValueError(f"Ratio member {index} has no native checkpoint.")
        try:
            payload = torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not safely decode native ratio member {index}."
            ) from exc
        required = {"config", "feature_order", "mean", "scale", "state_dict"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise ValueError(
                f"Native ratio member {index} checkpoint has an invalid schema."
            )
        if tuple(payload["feature_order"]) != features:
            raise ValueError(
                f"Native ratio member {index} feature order conflicts with "
                "the ensemble manifest."
            )
        checkpoint_config = RatioTrainingConfig(**dict(payload["config"]))
        if json_safe(payload["config"]) != manifest_config:
            raise ValueError(
                f"Native ratio member {index} training config conflicts with "
                "the ensemble manifest."
            )
        module = _build_log_ratio_module(
            torch,
            mean=payload["mean"],
            scale=payload["scale"],
            hidden_layers=checkpoint_config.hidden_layers,
            neurons=checkpoint_config.neurons,
            activation=checkpoint_config.activation,
        ).to(device)
        try:
            module.load_state_dict(payload["state_dict"], strict=True)
        except Exception as exc:
            raise ValueError(
                f"Native ratio member {index} state does not match its config."
            ) from exc
        module.eval()
        calibrator = (
            None
            if "calibration" not in files
            else PiecewiseLinearCalibrator.load(files["calibration"])
        )
        loaded_members.append(
            NativeRatioEvaluator(
                module=module,
                device=str(device),
                calibrator=calibrator,
            )
        )
    loaded = RatioEnsemble(loaded_members)
    if live_ensemble is None:
        return loaded
    if type(live_ensemble) is not RatioEnsemble:
        raise ValueError(
            "The live ratio ensemble cannot be authenticated as a native artifact."
        )
    if live_ensemble.minimum_ratio != loaded.minimum_ratio:
        raise ValueError(
            "The live ratio ensemble disagrees with its native ratio manifest."
        )
    if len(live_ensemble.members) != len(loaded.members):
        raise ValueError(
            "The live ratio ensemble disagrees with its native ratio manifest."
        )
    for index, (live, recorded) in enumerate(
        zip(live_ensemble.members, loaded.members, strict=True)
    ):
        if type(live) is not NativeRatioEvaluator:
            raise ValueError(
                f"Live ratio member {index} cannot be authenticated as native."
            )
        live_state = live.module.state_dict()
        recorded_state = recorded.module.state_dict()
        if live_state.keys() != recorded_state.keys() or any(
            not torch.equal(
                live_state[name].detach().cpu(),
                recorded_state[name].detach().cpu(),
            )
            for name in live_state
        ):
            raise ValueError(
                "The live ratio ensemble disagrees with its native ratio manifest."
            )
        live_calibrator = live.calibrator
        recorded_calibrator = recorded.calibrator
        if (live_calibrator is None) != (recorded_calibrator is None):
            raise ValueError(
                "The live ratio ensemble disagrees with its native ratio manifest."
            )
        if live_calibrator is not None and (
            live_calibrator.score_clip != recorded_calibrator.score_clip
            or not np.array_equal(live_calibrator.x, recorded_calibrator.x)
            or not np.array_equal(live_calibrator.y, recorded_calibrator.y)
        ):
            raise ValueError(
                "The live ratio ensemble disagrees with its native ratio manifest."
            )
    return loaded


__all__ = [
    "NativeRatioBackend",
    "NativeRatioEvaluator",
    "OnnxNativeRatioMember",
    "PiecewiseLinearCalibrator",
    "load_native_ratio_ensemble",
    "load_native_ratio_torch_ensemble",
]
