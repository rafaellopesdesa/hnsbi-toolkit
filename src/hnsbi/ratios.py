"""Backend-independent density-ratio ensembles and training orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .artifacts import write_artifact_manifest
from .onnx import OnnxRunner

RatioCallable = Callable[[np.ndarray], np.ndarray]


def _matrix(values: Any, n_features: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != n_features:
        raise ValueError(f"{name} must have shape (n, {n_features}).")
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be non-empty and finite.")
    return array


def normalize_class_weights(
    weights: Any,
    *,
    target_sum: float = 1.0,
) -> np.ndarray:
    """Normalize one class independently, as required for ratio training."""

    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("weights cannot be empty.")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite and non-negative.")
    if target_sum <= 0 or not np.isfinite(target_sum):
        raise ValueError("target_sum must be finite and positive.")
    total = float(np.sum(values))
    if not total > 0:
        raise ValueError("weights must have positive sum.")
    return values * (target_sum / total)


@dataclass
class RatioEnsemble:
    """An arithmetic mean of independently trained density ratios.

    The averaging happens in ratio space, not score or log-ratio space.  This
    preserves the ensemble convention used by ``nsbi-lhc-toolkit``.
    """

    members: Sequence[RatioCallable]
    minimum_ratio: float = 0.0

    def __post_init__(self) -> None:
        self.members = tuple(self.members)
        if not self.members:
            raise ValueError("A ratio ensemble requires at least one member.")
        if self.minimum_ratio < 0 or not np.isfinite(self.minimum_ratio):
            raise ValueError("minimum_ratio must be finite and non-negative.")

    def member_ratios(self, values: Any) -> np.ndarray:
        """Return an array with shape ``(n_members, n_events)``."""

        rows = len(values)
        predictions: list[np.ndarray] = []
        for index, member in enumerate(self.members):
            ratio = np.asarray(member(values), dtype=np.float64).reshape(-1)
            if len(ratio) != rows:
                raise ValueError(
                    f"Ratio member {index} returned {len(ratio)} predictions "
                    f"for {rows} events."
                )
            if not np.isfinite(ratio).all() or np.any(ratio < 0):
                raise ValueError(
                    f"Ratio member {index} returned a non-finite or negative ratio."
                )
            predictions.append(np.maximum(ratio, self.minimum_ratio))
        return np.stack(predictions, axis=0)

    def __call__(self, values: Any) -> np.ndarray:
        """Return the arithmetic mean ratio for every event."""

        return np.mean(self.member_ratios(values), axis=0)

    def standard_deviation(self, values: Any) -> np.ndarray:
        """Return the population spread across ensemble members."""

        return np.std(self.member_ratios(values), axis=0, ddof=0)

    def log_ratio(self, values: Any, *, floor: float = 1e-12) -> np.ndarray:
        """Return the log of the arithmetic-mean ratio."""

        if floor <= 0 or not np.isfinite(floor):
            raise ValueError("floor must be finite and positive.")
        return np.log(np.maximum(self(values), floor))


@dataclass
class OnnxRatioMember:
    """A portable upstream scaler-plus-classifier ratio evaluator."""

    scaler_path: Path
    model_path: Path
    use_log_loss: bool = False
    score_clip: float = 1e-9
    calibrator: Callable[[np.ndarray], np.ndarray] | None = None
    providers: Sequence[str] | None = None
    _scaler: OnnxRunner = field(init=False, repr=False)
    _model: OnnxRunner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.scaler_path = Path(self.scaler_path)
        self.model_path = Path(self.model_path)
        if not 0 < self.score_clip < 0.5:
            raise ValueError("score_clip must lie strictly between zero and 0.5.")
        self._scaler = OnnxRunner(self.scaler_path, providers=self.providers)
        self._model = OnnxRunner(self.model_path, providers=self.providers)

    def raw_output(self, values: Any) -> np.ndarray:
        """Evaluate scaler and classifier ONNX graphs in sequence."""

        scaled_outputs = self._scaler.run(values)
        scaled = np.asarray(next(iter(scaled_outputs.values())), dtype=np.float32)
        model_outputs = self._model.run(scaled)
        raw = np.asarray(next(iter(model_outputs.values()))).reshape(-1)
        if not np.isfinite(raw).all():
            raise ValueError("Ratio classifier returned a non-finite raw output.")
        return raw

    def score(self, values: Any) -> np.ndarray:
        """Return the binary-classifier score before odds conversion."""

        raw = self.raw_output(values)
        if self.use_log_loss:
            clipped = np.clip(raw, -80.0, 80.0)
            score = 1.0 / (1.0 + np.exp(-clipped))
        else:
            score = raw
            if np.any((score < 0.0) | (score > 1.0)):
                raise ValueError(
                    "Ratio classifier probability output must lie in [0, 1]."
                )
        if self.calibrator is not None:
            score = np.asarray(self.calibrator(score), dtype=np.float64)
        score = np.asarray(score, dtype=np.float64).reshape(-1)
        if not np.isfinite(score).all():
            raise ValueError("Ratio classifier probability output must be finite.")
        if np.any((score < 0.0) | (score > 1.0)):
            raise ValueError("Ratio classifier probability output must lie in [0, 1].")
        return np.clip(
            score,
            self.score_clip,
            1.0 - self.score_clip,
        )

    def __call__(self, values: Any) -> np.ndarray:
        if self.use_log_loss and self.calibrator is None:
            return np.exp(np.clip(self.raw_output(values), -80.0, 80.0))
        score = self.score(values)
        return score / (1.0 - score)


@dataclass(frozen=True)
class RatioTrainingConfig:
    """Shared settings passed to a ratio-training backend."""

    ensemble_size: int = 5
    hidden_layers: int = 3
    neurons: int = 256
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 1e-3
    scaler_type: str = "MinMax"
    scaling_features: tuple[str, ...] | None = None
    use_log_loss: bool = False
    calibration: bool = False
    calibration_type: str = "isotonic"
    calibration_bins: int = 40
    validation_fraction: float = 0.1
    holdout_fraction: float = 0.3
    early_stopping: bool = True
    patience: int = 30
    learning_rate_factor: float = 0.01
    activation: str = "swish"
    num_workers: int = 0
    seed: int = 0
    run_diagnostics: bool = True
    diagnostic_bins: int = 30
    load_existing: bool = False
    backend_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "ensemble_size",
            "hidden_layers",
            "neurons",
            "epochs",
            "batch_size",
            "calibration_bins",
            "patience",
            "diagnostic_bins",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie in (0, 1).")
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must lie in (0, 1).")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if self.learning_rate_factor <= 0:
            raise ValueError("learning_rate_factor must be positive.")
        if self.calibration_type not in {"isotonic", "histogram"}:
            raise ValueError("calibration_type must be 'isotonic' or 'histogram'.")


@dataclass
class RatioBackendResult:
    """One trained member returned by a backend adapter."""

    evaluator: RatioCallable
    files: dict[str, Path]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RatioTrainingBackend(Protocol):
    """Structural interface implemented by ratio-training integrations."""

    name: str

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
        """Train one member and return its portable evaluator and files."""


@dataclass
class RatioTrainingResult:
    """A fitted arithmetic-mean ensemble and its portable artifacts."""

    ensemble: RatioEnsemble
    members: tuple[RatioBackendResult, ...]
    manifest_path: Path
    features: tuple[str, ...]
    backend: str


class RatioTrainer:
    """Train an ensemble through a delegated backend."""

    def __init__(
        self,
        backend: RatioTrainingBackend,
        config: RatioTrainingConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or RatioTrainingConfig()

    def fit(
        self,
        numerator_values: Any,
        denominator_values: Any,
        *,
        features: Sequence[str],
        output_directory: str | Path,
        numerator_weights: Any | None = None,
        denominator_weights: Any | None = None,
        numerator_name: str = "numerator",
        denominator_name: str = "reference",
    ) -> RatioTrainingResult:
        """Fit every member and checksum the complete ratio bundle."""

        feature_names = tuple(features)
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("features must be non-empty and unique.")
        numerator = _matrix(numerator_values, len(feature_names), "numerator_values")
        denominator = _matrix(
            denominator_values, len(feature_names), "denominator_values"
        )
        numerator_raw_weights = (
            np.ones(len(numerator), dtype=np.float64)
            if numerator_weights is None
            else np.asarray(numerator_weights, dtype=np.float64).reshape(-1)
        )
        denominator_raw_weights = (
            np.ones(len(denominator), dtype=np.float64)
            if denominator_weights is None
            else np.asarray(denominator_weights, dtype=np.float64).reshape(-1)
        )
        if len(numerator_raw_weights) != len(numerator):
            raise ValueError(
                "numerator_weights must contain one value per numerator event."
            )
        if len(denominator_raw_weights) != len(denominator):
            raise ValueError(
                "denominator_weights must contain one value per denominator event."
            )
        numerator_normalized = normalize_class_weights(numerator_raw_weights)
        denominator_normalized = normalize_class_weights(denominator_raw_weights)
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        results: list[RatioBackendResult] = []
        for member_index in range(self.config.ensemble_size):
            member_directory = directory / f"member_{member_index:03d}"
            member_directory.mkdir(parents=True, exist_ok=True)
            result = self.backend.train_member(
                numerator_values=numerator,
                denominator_values=denominator,
                numerator_weights=numerator_normalized,
                denominator_weights=denominator_normalized,
                features=feature_names,
                output_directory=member_directory,
                member_index=member_index,
                numerator_name=numerator_name,
                denominator_name=denominator_name,
                config=self.config,
            )
            if not result.files:
                raise RuntimeError(
                    f"Backend member {member_index} returned no artifact files."
                )
            results.append(result)
        manifest_files: dict[str, Path] = {}
        for member_index, result in enumerate(results):
            for kind, path in result.files.items():
                manifest_files[f"member-{member_index:03d}-{kind}"] = Path(path)
        manifest_path = directory / "ratio_ensemble.manifest.json"
        write_artifact_manifest(
            manifest_path,
            artifact_type="density-ratio-ensemble",
            files=manifest_files,
            metadata={
                "backend": self.backend.name,
                "config": asdict(self.config),
                "denominator_name": denominator_name,
                "ensemble_reduction": "arithmetic-mean-of-ratios",
                "features": list(feature_names),
                "numerator_name": numerator_name,
                "weight_normalization": "independent-unit-sum",
            },
        )
        ensemble = RatioEnsemble([result.evaluator for result in results])
        return RatioTrainingResult(
            ensemble=ensemble,
            members=tuple(results),
            manifest_path=manifest_path,
            features=feature_names,
            backend=self.backend.name,
        )
