"""Density-ratio delegation to ``nsbi_common_utils``.

This adapter intentionally wraps only training and diagnostics.  Workspace
construction, model serialization, fitting, and scans remain owned by the
upstream LHC toolkit.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..artifacts import ArtifactManifest, write_artifact_manifest
from ..onnx import (
    compare_outputs,
    convert_joblib_scaler_to_onnx,
    manifest_path_for,
    require_optional,
)
from ..ratios import (
    OnnxRatioMember,
    RatioBackendResult,
    RatioEnsemble,
    RatioTrainingConfig,
)


class NsbiCommonUtilsBackend:
    """Adapter for the established LHC density-ratio trainer."""

    name = "nsbi_common_utils"

    def __init__(
        self,
        *,
        trainer_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._trainer_factory = trainer_factory

    @classmethod
    def available(cls) -> bool:
        """Return whether the upstream top-level package can be discovered."""

        return importlib.util.find_spec("nsbi_common_utils") is not None

    def _factory(self) -> Callable[..., Any]:
        if self._trainer_factory is not None:
            return self._trainer_factory
        training = require_optional(
            "nsbi_common_utils.training",
            extra="lhc",
            purpose="LHC density-ratio training and diagnostics",
        )
        return training.density_ratio_trainer

    @staticmethod
    def _paths(directory: Path, member_index: int) -> dict[str, Path]:
        suffix = str(member_index)
        return {
            "classifier-onnx": directory / f"model{suffix}.onnx",
            "native-scaler": directory / f"model_scaler{suffix}.bin",
            "scaler-onnx": directory / f"model_scaler{suffix}.onnx",
            "split-state": directory
            / f"num_events_random_state_train_holdout_split{suffix}.npy",
            "calibrator": directory / f"model_calibrated_hist{suffix}.obj",
        }

    @staticmethod
    def _backend_options(config: RatioTrainingConfig, section: str) -> dict[str, Any]:
        options = config.backend_options.get(section, {})
        if not isinstance(options, Mapping):
            raise ValueError(f"backend_options[{section!r}] must be a mapping.")
        return dict(options)

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
        """Train one member and convert its joblib scaler to ONNX."""

        pandas = require_optional(
            "pandas", extra="lhc", purpose="LHC density-ratio training"
        )
        require_optional(
            "onnxruntime",
            extra="lhc",
            purpose="validating LHC ratio ONNX artifacts",
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        figure_directory = output_directory / "diagnostics"
        figure_directory.mkdir(parents=True, exist_ok=True)
        values = np.concatenate([numerator_values, denominator_values], axis=0)
        weights = np.concatenate([numerator_weights, denominator_weights], axis=0)
        labels = np.concatenate(
            [
                np.ones(len(numerator_values), dtype=np.int64),
                np.zeros(len(denominator_values), dtype=np.int64),
            ]
        )
        frame = pandas.DataFrame(values, columns=features)
        scaling_features = (
            tuple(features)
            if config.scaling_features is None
            else tuple(config.scaling_features)
        )
        unknown_scaling = set(scaling_features).difference(features)
        if unknown_scaling:
            raise ValueError(
                f"scaling_features contains unknown features {sorted(unknown_scaling)}."
            )
        initializer_options = self._backend_options(config, "initializer")
        if initializer_options.get("delete_existing_models"):
            raise ValueError(
                "delete_existing_models=True is not supported by this adapter; "
                "choose a new output directory instead."
            )
        initializer_options["delete_existing_models"] = False
        directory_string = str(output_directory) + os.sep
        figure_string = str(figure_directory) + os.sep
        trainer = self._factory()(
            dataset=frame,
            weights=weights,
            training_labels=labels,
            features=list(features),
            features_scaling=list(scaling_features),
            sample_name=[numerator_name, denominator_name],
            output_name=f"{numerator_name}vs{denominator_name}",
            path_to_figures=figure_string,
            path_to_models=directory_string,
            use_log_loss=config.use_log_loss,
            **initializer_options,
        )
        controlled_train_options = {
            "activation": config.activation,
            "batch_size": config.batch_size,
            "callback": config.early_stopping,
            "callback_factor": config.learning_rate_factor,
            "callback_patience": config.patience,
            "calibration": config.calibration,
            "ensemble_index": member_index,
            "hidden_layers": config.hidden_layers,
            "holdout_split": config.holdout_fraction,
            "learning_rate": config.learning_rate,
            "load_trained_models": config.load_existing,
            "neurons": config.neurons,
            "num_bins_cal": config.calibration_bins,
            "num_workers": config.num_workers,
            "number_of_epochs": config.epochs,
            "rnd_seed": config.seed + member_index,
            "scalerType": config.scaler_type,
            "type_of_calibration": config.calibration_type,
            "validation_split": config.validation_fraction,
        }
        train_options = self._backend_options(config, "train")
        overlap = set(controlled_train_options).intersection(train_options)
        if overlap:
            raise ValueError(
                "Use RatioTrainingConfig fields instead of overriding "
                f"controlled upstream options {sorted(overlap)}."
            )
        controlled_train_options.update(train_options)
        trainer.train(**controlled_train_options)
        if config.run_diagnostics:
            self.run_diagnostics(
                trainer,
                features=features,
                member_index=member_index,
                bins=config.diagnostic_bins,
                options=self._backend_options(config, "diagnostics"),
            )

        paths = self._paths(output_directory, member_index)
        for kind in ("classifier-onnx", "native-scaler", "split-state"):
            if not paths[kind].is_file():
                raise FileNotFoundError(
                    f"nsbi_common_utils did not produce expected {kind} "
                    f"artifact {paths[kind]}."
                )
        convert_joblib_scaler_to_onnx(
            paths["native-scaler"],
            paths["scaler-onnx"],
            n_features=len(features),
            feature_names=features,
            allow_unsafe_pickle=True,
            metadata={
                "backend": self.name,
                "features": list(features),
                "member_index": member_index,
            },
        )
        model_manifest = manifest_path_for(paths["classifier-onnx"])
        write_artifact_manifest(
            model_manifest,
            artifact_type="nsbi-common-utils-ratio-classifier",
            files={"classifier-onnx": paths["classifier-onnx"]},
            metadata={
                "features": list(features),
                "member_index": member_index,
                "output_semantics": (
                    "log-ratio" if config.use_log_loss else "binary-score"
                ),
            },
        )
        calibrator = (
            getattr(trainer, "histogram_calibrator", None)
            if config.calibration
            else None
        )
        calibrator_function = None if calibrator is None else calibrator.cali_pred
        evaluator = OnnxRatioMember(
            scaler_path=paths["scaler-onnx"],
            model_path=paths["classifier-onnx"],
            use_log_loss=config.use_log_loss,
            calibrator=calibrator_function,
        )
        parity_rows = min(512, len(frame))
        native_scores = np.asarray(trainer.full_data_prediction[:parity_rows]).reshape(
            -1
        )
        portable_scores = evaluator.score(values[:parity_rows])
        parity = compare_outputs(
            native_scores,
            portable_scores,
            atol=2e-5,
            rtol=2e-4,
        )
        parity.assert_close()
        parity_path = output_directory / "onnx_parity.json"
        parity_path.write_text(
            json.dumps(parity.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        artifact_files: dict[str, Path] = {
            "classifier-onnx": paths["classifier-onnx"],
            "classifier-onnx-manifest": model_manifest,
            "native-scaler": paths["native-scaler"],
            "scaler-onnx": paths["scaler-onnx"],
            "scaler-onnx-manifest": manifest_path_for(paths["scaler-onnx"]),
            "split-state": paths["split-state"],
            "onnx-parity": parity_path,
        }
        if paths["calibrator"].is_file():
            artifact_files["calibrator"] = paths["calibrator"]
        for figure in sorted(figure_directory.rglob("*")):
            if figure.is_file():
                relative = figure.relative_to(figure_directory).as_posix()
                key = "diagnostic-" + relative.replace("/", "-")
                artifact_files[key] = figure
        member_manifest = output_directory / "member.manifest.json"
        write_artifact_manifest(
            member_manifest,
            artifact_type="nsbi-common-utils-ratio-member",
            files=artifact_files,
            metadata={
                "backend": self.name,
                "config": asdict(config),
                "denominator_name": denominator_name,
                "features": list(features),
                "member_index": member_index,
                "numerator_name": numerator_name,
                "portable_inference": ("scaler ONNX -> classifier ONNX -> ratio"),
            },
        )
        artifact_files["member-manifest"] = member_manifest
        return RatioBackendResult(
            evaluator=evaluator,
            files=artifact_files,
            metadata={
                "member_index": member_index,
                "onnx_parity": parity.to_dict(),
            },
        )

    @staticmethod
    def run_diagnostics(
        trainer: Any,
        *,
        features: Sequence[str],
        member_index: int,
        bins: int,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """Delegate the established overfit, closure, and normalization checks."""

        settings = dict(options or {})
        methods = tuple(
            settings.pop(
                "methods",
                (
                    "overfit",
                    "calibration",
                    "reweighting",
                    "normalization",
                ),
            )
        )
        if settings:
            raise ValueError(f"Unknown diagnostic options {sorted(settings)}.")
        unknown = set(methods).difference(
            {"overfit", "calibration", "reweighting", "normalization"}
        )
        if unknown:
            raise ValueError(f"Unknown diagnostic methods {sorted(unknown)}.")
        if "overfit" in methods:
            trainer.make_overfit_plots(ensemble_index=member_index)
        if "calibration" in methods:
            trainer.make_calib_plots(
                observable="score",
                nbins=bins,
                ensemble_index=member_index,
            )
        if "reweighting" in methods:
            trainer.make_reweighted_plots(
                list(features),
                "linear",
                bins,
                ensemble_index=member_index,
            )
        if "normalization" in methods:
            trainer.test_normalization()

    @classmethod
    def load_member(
        cls,
        directory: str | Path,
        member_index: int,
        *,
        use_log_loss: bool | None = None,
        calibration: bool | None = None,
        allow_unsafe_pickle: bool = False,
        verify: bool = True,
        expected_features: Sequence[str] | None = None,
        expected_config: Mapping[str, Any] | None = None,
        expected_numerator_name: str | None = None,
        expected_denominator_name: str | None = None,
        providers: Sequence[str] | None = None,
    ) -> OnnxRatioMember:
        """Load a portable member produced by this adapter."""

        require_optional(
            "onnxruntime", extra="lhc", purpose="loading an LHC ratio member"
        )
        member_directory = Path(directory)
        manifest = member_directory / "member.manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing ratio member manifest {manifest}.")
        return cls._load_member_manifest(
            manifest,
            member_index,
            use_log_loss=use_log_loss,
            calibration=calibration,
            allow_unsafe_pickle=allow_unsafe_pickle,
            verify=verify,
            expected_features=expected_features,
            expected_config=expected_config,
            expected_numerator_name=expected_numerator_name,
            expected_denominator_name=expected_denominator_name,
            providers=providers,
        )

    @staticmethod
    def _validated_config(
        metadata: Mapping[str, Any],
        *,
        context: str,
    ) -> dict[str, Any]:
        raw_config = metadata.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"{context} has no valid ratio training config.")
        config = dict(raw_config)
        expected_keys = {field.name for field in fields(RatioTrainingConfig)}
        missing = expected_keys.difference(config)
        unknown = set(config).difference(expected_keys)
        if missing or unknown:
            raise ValueError(
                f"{context} ratio config fields do not match "
                f"RatioTrainingConfig; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}."
            )
        for name in ("use_log_loss", "calibration"):
            if not isinstance(config[name], bool):
                raise ValueError(f"{context} config field {name!r} must be boolean.")
        try:
            RatioTrainingConfig(**config)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context} contains an invalid ratio training config."
            ) from exc
        return config

    @staticmethod
    def _validated_features(
        metadata: Mapping[str, Any],
        *,
        context: str,
    ) -> tuple[str, ...]:
        raw_features = metadata.get("features")
        if not isinstance(raw_features, list):
            raise ValueError(f"{context} feature signature must be a JSON array.")
        features = tuple(raw_features)
        if (
            not features
            or any(not isinstance(name, str) or not name for name in features)
            or len(set(features)) != len(features)
        ):
            raise ValueError(
                f"{context} feature signature must be non-empty and unique."
            )
        return features

    @staticmethod
    def _validated_name(
        metadata: Mapping[str, Any],
        key: str,
        *,
        context: str,
    ) -> str:
        value = metadata.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context} has no valid {key}.")
        return value

    @staticmethod
    def _path_from_record(
        record: Any,
        *,
        root: Path,
        context: str,
    ) -> Path:
        root_resolved = root.resolve()
        candidate = root.joinpath(*PurePosixPath(record.path).parts)
        try:
            candidate.resolve().relative_to(root_resolved)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{context} path escapes its artifact root: {record.path!r}."
            ) from exc
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    @classmethod
    def _recorded_path(
        cls,
        manifest: ArtifactManifest,
        *,
        root: Path,
        kind: str,
        required: bool = True,
    ) -> Path | None:
        records = [record for record in manifest.files if record.kind == kind]
        if len(records) > 1:
            raise ValueError(f"Ratio manifest repeats artifact role {kind!r}.")
        if not records:
            if required:
                raise ValueError(f"Ratio manifest is missing artifact role {kind!r}.")
            return None
        return cls._path_from_record(
            records[0],
            root=root,
            context=f"Ratio artifact role {kind!r}",
        )

    @classmethod
    def _load_member_manifest(
        cls,
        manifest: Path,
        member_index: int,
        *,
        use_log_loss: bool | None,
        calibration: bool | None,
        allow_unsafe_pickle: bool,
        verify: bool,
        expected_features: Sequence[str] | None,
        expected_config: Mapping[str, Any] | None,
        expected_numerator_name: str | None,
        expected_denominator_name: str | None,
        providers: Sequence[str] | None,
    ) -> OnnxRatioMember:
        if type(member_index) is not int or member_index < 0:
            raise ValueError("member_index must be a non-negative integer.")
        member_directory = manifest.parent
        loaded_manifest = ArtifactManifest.load(manifest)
        if loaded_manifest.artifact_type != "nsbi-common-utils-ratio-member":
            raise ValueError(
                "Unexpected ratio member artifact type "
                f"{loaded_manifest.artifact_type!r}."
            )
        if verify:
            loaded_manifest.verify(member_directory)
        metadata = loaded_manifest.metadata
        if metadata.get("backend") != cls.name:
            raise ValueError(
                "Ratio member manifest is not bound to the nsbi_common_utils backend."
            )
        recorded_index = metadata.get("member_index")
        if type(recorded_index) is not int or recorded_index != member_index:
            raise ValueError(
                "Ratio member index mismatch: expected "
                f"{member_index}, found {recorded_index!r}."
            )
        recorded_config = cls._validated_config(
            metadata, context="Ratio member manifest"
        )
        if expected_config is not None and dict(expected_config) != recorded_config:
            raise ValueError(
                "Ratio member config conflicts with its ensemble manifest."
            )
        recorded_log_loss = recorded_config["use_log_loss"]
        recorded_calibration = recorded_config["calibration"]
        if use_log_loss is None:
            use_log_loss = recorded_log_loss
        elif not isinstance(use_log_loss, bool):
            raise ValueError("use_log_loss must be boolean when supplied.")
        elif use_log_loss != recorded_log_loss:
            raise ValueError(
                "Requested use_log_loss conflicts with the ratio manifest."
            )
        if calibration is None:
            calibration = recorded_calibration
        elif not isinstance(calibration, bool):
            raise ValueError("calibration must be boolean when supplied.")
        elif calibration != recorded_calibration:
            raise ValueError("Requested calibration conflicts with the ratio manifest.")
        features = cls._validated_features(metadata, context="Ratio member manifest")
        if expected_features is not None and tuple(expected_features) != features:
            raise ValueError(
                "Ratio member feature order mismatch: expected "
                f"{tuple(expected_features)}, found {features}."
            )
        numerator_name = cls._validated_name(
            metadata,
            "numerator_name",
            context="Ratio member manifest",
        )
        denominator_name = cls._validated_name(
            metadata,
            "denominator_name",
            context="Ratio member manifest",
        )
        if (
            expected_numerator_name is not None
            and numerator_name != expected_numerator_name
        ):
            raise ValueError(
                "Ratio member numerator name conflicts with its ensemble manifest."
            )
        if (
            expected_denominator_name is not None
            and denominator_name != expected_denominator_name
        ):
            raise ValueError(
                "Ratio member denominator name conflicts with its ensemble manifest."
            )
        scaler_path = cls._recorded_path(
            loaded_manifest,
            root=member_directory,
            kind="scaler-onnx",
        )
        model_path = cls._recorded_path(
            loaded_manifest,
            root=member_directory,
            kind="classifier-onnx",
        )
        calibrator_path = cls._recorded_path(
            loaded_manifest,
            root=member_directory,
            kind="calibrator",
            required=recorded_calibration,
        )
        calibrator_function = None
        if calibration:
            if not allow_unsafe_pickle:
                raise ValueError(
                    "This calibrated legacy member requires a pickle artifact, "
                    "which can execute code. Pass allow_unsafe_pickle=True only "
                    "for a bundle you trust, or retrain without calibration."
                )
            assert calibrator_path is not None
            require_optional(
                "nsbi_common_utils",
                extra="lhc",
                purpose="loading an upstream calibration object",
            )
            with calibrator_path.open("rb") as stream:
                calibrator = pickle.load(stream)
            calibrator_function = getattr(calibrator, "cali_pred", None)
            if not callable(calibrator_function):
                raise ValueError(
                    "Calibration artifact has no callable cali_pred method."
                )
        assert scaler_path is not None
        assert model_path is not None
        return OnnxRatioMember(
            scaler_path=scaler_path,
            model_path=model_path,
            use_log_loss=use_log_loss,
            calibrator=calibrator_function,
            providers=providers,
        )

    @classmethod
    def load_ensemble(
        cls,
        directory: str | Path,
        ensemble_size: int | None = None,
        *,
        use_log_loss: bool | None = None,
        calibration: bool | None = None,
        allow_unsafe_pickle: bool = False,
        verify: bool = True,
        expected_features: Sequence[str] | None = None,
        providers: Sequence[str] | None = None,
    ) -> RatioEnsemble:
        """Load the exact ordered members declared by the ensemble manifest."""

        root = Path(directory)
        manifest_path = root / "ratio_ensemble.manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing ratio ensemble manifest {manifest_path}.")
        manifest = ArtifactManifest.load(manifest_path)
        if manifest.artifact_type != "density-ratio-ensemble":
            raise ValueError(
                f"Unexpected ratio ensemble artifact type {manifest.artifact_type!r}."
            )
        if verify:
            manifest.verify(root)
        metadata = manifest.metadata
        if metadata.get("backend") != cls.name:
            raise ValueError(
                "Ratio ensemble manifest is not bound to the nsbi_common_utils backend."
            )
        if metadata.get("ensemble_reduction") != "arithmetic-mean-of-ratios":
            raise ValueError(
                "Unsupported ratio ensemble reduction; expected "
                "'arithmetic-mean-of-ratios'."
            )
        config = cls._validated_config(metadata, context="Ratio ensemble manifest")
        configured_size = config["ensemble_size"]
        if type(configured_size) is not int or configured_size < 1:
            raise ValueError("Ratio ensemble config has an invalid ensemble_size.")
        if ensemble_size is not None:
            if type(ensemble_size) is not int or ensemble_size < 1:
                raise ValueError("ensemble_size must be a positive integer.")
            if ensemble_size != configured_size:
                raise ValueError(
                    "Requested ensemble_size conflicts with the ratio "
                    "ensemble manifest."
                )
        features = cls._validated_features(metadata, context="Ratio ensemble manifest")
        if expected_features is not None and tuple(expected_features) != features:
            raise ValueError(
                "Ratio ensemble feature order mismatch: expected "
                f"{tuple(expected_features)}, found {features}."
            )
        numerator_name = cls._validated_name(
            metadata,
            "numerator_name",
            context="Ratio ensemble manifest",
        )
        denominator_name = cls._validated_name(
            metadata,
            "denominator_name",
            context="Ratio ensemble manifest",
        )
        recorded_log_loss = config["use_log_loss"]
        if use_log_loss is not None and (
            not isinstance(use_log_loss, bool) or use_log_loss != recorded_log_loss
        ):
            raise ValueError(
                "Requested use_log_loss conflicts with the ratio manifest."
            )
        recorded_calibration = config["calibration"]
        if calibration is not None and (
            not isinstance(calibration, bool) or calibration != recorded_calibration
        ):
            raise ValueError("Requested calibration conflicts with the ratio manifest.")

        member_records: dict[int, Any] = {}
        pattern = re.compile(r"member-(\d+)-member-manifest\Z")
        for record in manifest.files:
            match = pattern.fullmatch(record.kind)
            if match is None:
                continue
            index = int(match.group(1))
            if record.kind != f"member-{index:03d}-member-manifest":
                raise ValueError(
                    "Ratio ensemble contains a non-canonical member manifest "
                    f"role {record.kind!r}."
                )
            if index in member_records:
                raise ValueError(
                    f"Ratio ensemble repeats member manifest index {index}."
                )
            member_records[index] = record
        expected_indices = list(range(configured_size))
        if sorted(member_records) != expected_indices:
            raise ValueError(
                "Ratio ensemble member manifest order is incomplete or "
                f"non-contiguous: expected {expected_indices}, found "
                f"{sorted(member_records)}."
            )

        members = []
        for index in expected_indices:
            member_manifest = cls._path_from_record(
                member_records[index],
                root=root,
                context=f"Ratio ensemble member {index}",
            )
            members.append(
                cls._load_member_manifest(
                    member_manifest,
                    index,
                    use_log_loss=recorded_log_loss,
                    calibration=recorded_calibration,
                    allow_unsafe_pickle=allow_unsafe_pickle,
                    verify=verify,
                    expected_features=features,
                    expected_config=config,
                    expected_numerator_name=numerator_name,
                    expected_denominator_name=denominator_name,
                    providers=providers,
                )
            )
        return RatioEnsemble(members)
