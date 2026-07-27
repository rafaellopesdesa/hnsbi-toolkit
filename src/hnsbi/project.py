"""Configuration-first orchestration for frequentist hNSBI projects."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .asimov import AsimovBuilder
from .config import ToolkitConfig
from .data import DataSource
from .intensity import IntensityModel, RatioNormalizer
from .protocols import RatioEvaluator, Sampler
from .ratio_diagnostics import RatioValidationReport
from .toys import ToyGenerator


@dataclass(frozen=True)
class ReferenceTrainingArtifacts:
    """Native and portable outputs from one reference-flow training."""

    training: Any
    checkpoint_path: Path
    checkpoint_manifest: Path
    onnx_bundle: Any
    onnx_parity: Mapping[str, Any]
    diagnostics: Any
    diagnostics_report: Path
    diagnostics_manifest: Path


@dataclass(frozen=True)
class RatioSetTrainingArtifacts:
    """Nominal ratio ensembles and independent reference normalizers."""

    training: Mapping[str, Any]
    normalizer: RatioNormalizer
    normalization_events: int
    normalizer_path: Path
    normalizer_manifest: Path
    diagnostics: Mapping[str, RatioValidationReport] = field(default_factory=dict)

    @property
    def evaluators(self) -> dict[str, RatioEvaluator]:
        return {name: result.ensemble for name, result in self.training.items()}


@dataclass(frozen=True)
class NISWorkflowArtifacts:
    """Trained proposal, diagnostics, ONNX bundle, and efficient Asimov."""

    design: Any
    flow_training: Any
    checkpoint_path: Path
    checkpoint_manifest: Path
    onnx_bundle: Any
    validation: Any
    validation_report: Path
    validation_manifest: Path
    validation_provenance: Mapping[str, Any]
    defensive_proposal: Any
    asimov: Any
    asimov_path: Path
    asimov_array_paths: Mapping[str, Path]
    onnx_parity: Mapping[str, Any]


@dataclass(frozen=True)
class FNFModelArtifacts:
    """One fitted or loaded sample-specific factorizable flow bundle."""

    name: str
    sample: str
    residual: Any
    artifact: Any
    yield_morph: Any | None = None
    training: Any | None = None

    @property
    def manifest_path(self) -> Path:
        return self.artifact.manifest_path


class Project:
    """A validated YAML/JSON/dictionary project with explicit runtime objects.

    Configuration controls reproducible defaults and artifact locations.
    In-memory PyArrow/Awkward/pandas objects are supplied through ``registry``
    and referenced by ``registry_key`` in the same schema used for files.
    Learned models remain ordinary Python objects, so advanced analyses can
    replace any training or execution backend without subclassing ``Project``.
    """

    def __init__(
        self,
        config: ToolkitConfig,
        *,
        base_directory: str | Path = ".",
        registry: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.base_directory = Path(base_directory)
        self.registry = dict(registry or {})

    @classmethod
    def load(
        cls,
        source: str | Path | Mapping[str, Any],
        *,
        registry: Mapping[str, Any] | None = None,
        validate_schema: bool = True,
    ) -> Project:
        if isinstance(source, Mapping):
            base = Path.cwd()
        else:
            base = Path(source).resolve().parent
        return cls(
            ToolkitConfig.load(source, validate_schema=validate_schema),
            base_directory=base,
            registry=registry,
        )

    @property
    def output_directory(self) -> Path:
        value = self.config.output_dir
        return value if value.is_absolute() else self.base_directory / value

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve an analysis-relative path from the project configuration."""

        path = Path(value)
        return path if path.is_absolute() else self.base_directory / path

    def _resolve_path(self, value: str | Path) -> Path:
        """Backward-compatible private alias for :meth:`resolve_path`."""

        return self.resolve_path(value)

    def _frequentist(self) -> dict[str, Any]:
        value = self.config.frequentist
        if value is None:
            raise ValueError("This project has no frequentist configuration.")
        return value

    def _bayesian(self) -> dict[str, Any]:
        value = self.config.bayesian
        if value is None:
            raise ValueError("This project has no Bayesian configuration.")
        return value

    def data_source(
        self,
        specification: Mapping[str, Any],
        *,
        features: tuple[str, ...] | None = None,
    ) -> DataSource:
        """Resolve one schema data source without materializing it."""

        spec = dict(specification)
        if spec.get("selection") is not None:
            raise NotImplementedError(
                "Data-source selections are intentionally not interpreted as "
                "Python. Apply selections in Arrow/Awkward before registration."
            )
        kind = spec.get("kind")
        if "registry_key" in spec:
            key = spec["registry_key"]
            if key not in self.registry:
                raise KeyError(f"No in-memory dataset is registered as {key!r}.")
            source = self.registry[key]
            data_format = None
        elif "path" in spec:
            path = Path(spec["path"])
            source = path if path.is_absolute() else self.base_directory / path
            data_format = "arrow_ipc" if kind == "arrow_ipc" else "parquet"
        elif "paths" in spec:
            paths = [Path(value) for value in spec["paths"]]
            source = [
                value if value.is_absolute() else self.base_directory / value
                for value in paths
            ]
            data_format = "arrow_ipc" if kind == "arrow_ipc" else "parquet"
        else:
            raise ValueError("Data source needs path, paths, or registry_key.")
        auxiliary = tuple(
            dict.fromkeys(
                value
                for value in (
                    spec.get("split_column"),
                    spec.get("group_column"),
                    spec.get("log_density_column"),
                )
                if value is not None
            )
        )
        return DataSource(
            source,
            features=self.config.features if features is None else features,
            weight=spec.get("weight_column"),
            row_id=spec.get("event_id_column"),
            auxiliary=auxiliary,
            format=data_format,
            batch_size=int(spec.get("batch_size", 65_536)),
        )

    def reference_source(self) -> DataSource:
        return self.data_source(self._frequentist()["reference"])

    def sample_sources(self) -> dict[str, DataSource]:
        return {
            sample["name"]: self.data_source(sample["source"])
            for sample in self._frequentist()["samples"]
        }

    @staticmethod
    def _ratio_partition_metadata(
        specification: Mapping[str, Any],
        batch: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return configured scientific splits and correlated-event groups."""

        split_column = specification.get("split_column")
        split = (
            None if split_column is None else np.asarray(batch.columns[split_column])
        )
        group_column = specification.get("group_column")
        if group_column is not None:
            groups = np.asarray(batch.columns[group_column])
        elif specification.get("event_id_column") is not None:
            groups = np.asarray(batch.row_ids)
        else:
            # DataSource generates positional row IDs when none were declared.
            # Those IDs must not accidentally correlate unrelated samples.
            groups = None
        return split, groups

    def fnf_model_specifications(self) -> tuple[dict[str, Any], ...]:
        """Return the validated sample-specific FNF model specifications."""

        section = self._frequentist().get("fnf")
        if section is None:
            return ()
        return tuple(dict(model) for model in section["models"])

    def _fnf_model_specification(self, name: str) -> dict[str, Any]:
        matches = [
            model for model in self.fnf_model_specifications() if model["name"] == name
        ]
        if not matches:
            raise KeyError(f"Unknown FNF model {name!r}.")
        return matches[0]

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def fnf_configs(self, name: str) -> tuple[Any, Any]:
        """Translate one YAML FNF model to residual and training configs."""

        from .fnf import FNFResidualConfig, FNFTrainingConfig

        model = self._fnf_model_specification(name)
        nuisance_names = tuple(model["nuisances"])
        parameters = {
            parameter["name"]: parameter
            for parameter in self._frequentist()["parameters"]
        }
        centers = {
            nuisance: float(parameters[nuisance]["nominal"])
            for nuisance in nuisance_names
        }
        centers.update(
            {
                nuisance: float(value)
                for nuisance, value in model.get("centers", {}).items()
            }
        )
        scales = {}
        for nuisance in nuisance_names:
            constraint = parameters[nuisance].get("constraint")
            scales[nuisance] = (
                float(constraint["sigma"]) if isinstance(constraint, Mapping) else 1.0
            )
        scales.update(
            {
                nuisance: float(value)
                for nuisance, value in model.get("scales", {}).items()
            }
        )
        architecture = model.get("architecture", {})
        training = model.get("training", {})
        residual_config = FNFResidualConfig(
            n_features=len(self.config.features),
            nuisance_names=nuisance_names,
            num_layers=int(architecture.get("num_layers", 2)),
            hidden_features=tuple(architecture.get("hidden_features", (64, 64))),
            interactions=tuple(tuple(pair) for pair in model.get("interactions", ())),
            nuisance_centers=tuple(centers[nuisance] for nuisance in nuisance_names),
            nuisance_scales=tuple(scales[nuisance] for nuisance in nuisance_names),
            quadratic_scale=float(architecture.get("quadratic_scale", 1.0)),
            interaction_scale=float(architecture.get("interaction_scale", 1.0)),
            log_scale_clip=float(architecture.get("log_scale_clip", 2.0)),
            shift_clip=(
                None
                if architecture.get("shift_clip") is None
                else float(architecture["shift_clip"])
            ),
        )
        training_config = FNFTrainingConfig(
            epochs=int(training.get("epochs", 100)),
            batch_size=int(training.get("batch_size", 1024)),
            learning_rate=float(training.get("learning_rate", 1.0e-4)),
            weight_decay=float(training.get("weight_decay", 1.0e-5)),
            validation_fraction=float(training.get("validation_fraction", 0.2)),
            holdout_fraction=float(training.get("holdout_fraction", 0.0)),
            patience=int(training.get("early_stopping_patience", 20)),
            min_delta=float(training.get("min_delta", 1.0e-5)),
            gradient_clip_norm=(
                None
                if training.get("gradient_clip_norm", 5.0) is None
                else float(training.get("gradient_clip_norm", 5.0))
            ),
            steps_per_epoch=(
                None
                if training.get("steps_per_epoch") is None
                else int(training["steps_per_epoch"])
            ),
            device=self._resolve_device(str(training.get("device", "cpu"))),
        )
        return residual_config, training_config

    def fnf_anchors(
        self,
        name: str,
        *,
        max_events_per_anchor: int | None = None,
    ) -> tuple[Any, ...]:
        """Materialize the YAML anchor samples for one FNF model."""

        from .fnf import FNFAnchor

        model = self._fnf_model_specification(name)
        configured_limit = model.get("training", {}).get("max_events_per_anchor")
        limit = (
            configured_limit if max_events_per_anchor is None else max_events_per_anchor
        )
        result = []
        for anchor in model["anchors"]:
            source_specification = anchor["source"]
            batch = self.data_source(source_specification).materialize(
                batch_size=int(source_specification.get("batch_size", 65_536)),
                max_events=None if limit is None else int(limit),
            )
            if not len(batch.values):
                raise ValueError(f"FNF anchor {name!r}/{anchor['name']!r} is empty.")
            group_column = source_specification.get("group_column")
            if group_column is not None:
                groups = batch.columns[group_column]
            elif source_specification.get("event_id_column") is not None:
                groups = batch.row_ids
            else:
                groups = None
            result.append(
                FNFAnchor(
                    values=batch.values,
                    point=dict(anchor["point"]),
                    weights=batch.weights,
                    groups=groups,
                    name=anchor["name"],
                )
            )
        return tuple(result)

    def _fnf_yield_morph(self, model: Mapping[str, Any], config: Any) -> Any | None:
        anchors = model.get("yield_anchors", {})
        if not anchors:
            return None
        from .fnf import LogQuadraticYieldMorph

        samples = {sample["name"]: sample for sample in self._frequentist()["samples"]}
        nominal_yield = float(samples[model["sample"]]["nominal_yield"])
        absolute = {
            nuisance: tuple(nominal_yield * float(factor) for factor in factors)
            for nuisance, factors in anchors.items()
        }
        centers = dict(
            zip(
                config.nuisance_names,
                config.nuisance_centers,
                strict=True,
            )
        )
        scales = dict(
            zip(
                config.nuisance_names,
                config.nuisance_scales,
                strict=True,
            )
        )
        return LogQuadraticYieldMorph.from_anchors(
            nominal_yield,
            absolute,
            centers=centers,
            scales=scales,
        )

    @staticmethod
    def _fnf_artifact_target(path: Path) -> tuple[Path, str]:
        suffix = ".manifest.json"
        if not path.name.endswith(suffix) or path.name == suffix:
            raise ValueError("An FNF artifact path must end in '.manifest.json'.")
        return path.parent, path.name[: -len(suffix)]

    def train_fnf_systematics(
        self,
        *,
        base_torch_log_probs: Mapping[str, Any] | None = None,
        reference: Any | None = None,
        ratios: RatioSetTrainingArtifacts | None = None,
        reference_manifest: str | Path | None = None,
        max_events_per_anchor: int | None = None,
        seed: int | None = None,
    ) -> dict[str, FNFModelArtifacts]:
        """Train and save every YAML-configured sample-specific FNF model.

        Callers may provide explicit ``base_torch_log_probs``. The normal
        project workflow instead supplies the trained reference flow and
        :class:`RatioSetTrainingArtifacts`; the toolkit composes each
        normalized nominal process density automatically and binds the FNF
        artifact to the exact reference and ratio manifests.
        """

        from .artifacts import sha256_file
        from .fnf import FNFTrainer
        from .fnf_runtime import (
            NativeProcessDensity,
            load_verified_reference_density,
        )
        from .native_ratios import load_native_ratio_torch_ensemble

        specifications = self.fnf_model_specifications()
        base_densities: dict[str, NativeProcessDensity] | None = None
        reference_manifest_path: Path | None = None
        authenticated_normalizer: RatioNormalizer | None = None
        if base_torch_log_probs is None:
            if reference is None or ratios is None:
                raise ValueError(
                    "FNF training requires either base_torch_log_probs or both "
                    "the trained reference flow and ratio artifacts."
                )
            if reference_manifest is None:
                raise ValueError(
                    "Project FNF training with a native nominal density requires "
                    "the reference-flow manifest so the residual can be bound "
                    "to a portable scientific model."
                )
            reference_manifest_path = Path(reference_manifest)
            if not reference_manifest_path.is_file():
                raise FileNotFoundError(reference_manifest_path)
            expected_normalizer_manifest = ratios.normalizer_path.with_suffix(
                ratios.normalizer_path.suffix + ".manifest.json"
            )
            if (
                expected_normalizer_manifest.resolve()
                != Path(ratios.normalizer_manifest).resolve()
            ):
                raise ValueError(
                    "Ratio artifacts do not bind their declared normalizer manifest."
                )
            authenticated_normalizer = RatioNormalizer.load(ratios.normalizer_path)
            if (
                authenticated_normalizer.means != ratios.normalizer.means
                or authenticated_normalizer.standard_errors
                != ratios.normalizer.standard_errors
                or authenticated_normalizer.metadata != ratios.normalizer.metadata
            ):
                raise ValueError(
                    "The live ratio normalizer disagrees with its checked artifact."
                )
            authenticated_reference = load_verified_reference_density(
                reference_manifest_path,
                features=tuple(self.config.features),
                live_density=reference,
            )
            required_samples = {model["sample"] for model in specifications}
            missing_ratios = required_samples.difference(ratios.training)
            if missing_ratios:
                raise KeyError(
                    "Missing nominal ratio artifacts for FNF samples "
                    f"{sorted(missing_ratios)}."
                )
            authenticated_ratios = {
                sample: load_native_ratio_torch_ensemble(
                    ratios.training[sample].manifest_path,
                    expected_features=tuple(self.config.features),
                    live_ensemble=ratios.evaluators[sample],
                )
                for sample in required_samples
            }
            base_densities = {
                model["sample"]: NativeProcessDensity(
                    reference_density=authenticated_reference,
                    ratio=authenticated_ratios[model["sample"]],
                    ratio_normalization=authenticated_normalizer.means[model["sample"]],
                )
                for model in specifications
            }
            base_torch_log_probs = {
                sample: density.torch_log_prob
                for sample, density in base_densities.items()
            }
        missing = {
            model["sample"]
            for model in specifications
            if model["sample"] not in base_torch_log_probs
        }
        if missing:
            raise KeyError(
                "Missing nominal Torch log densities for FNF samples "
                f"{sorted(missing)}."
            )
        result: dict[str, FNFModelArtifacts] = {}
        for model in specifications:
            residual_config, training_config = self.fnf_configs(model["name"])
            if base_densities is not None:
                base_densities[model["sample"]].to(training_config.device)
            training_seed = int(
                model.get("training", {}).get("seed", 0) if seed is None else seed
            )
            training = FNFTrainer(
                residual_config,
                training_config,
            ).fit(
                self.fnf_anchors(
                    model["name"],
                    max_events_per_anchor=max_events_per_anchor,
                ),
                base_torch_log_prob=base_torch_log_probs[model["sample"]],
                features=self.config.features,
                seed=training_seed,
            )
            yield_morph = self._fnf_yield_morph(model, residual_config)
            base_dependencies = None
            if base_densities is not None:
                assert reference_manifest_path is not None
                assert authenticated_normalizer is not None
                ratio_manifest = Path(ratios.training[model["sample"]].manifest_path)
                base_dependencies = {
                    "reference_manifest_sha256": sha256_file(reference_manifest_path),
                    "ratio_manifest_sha256": sha256_file(ratio_manifest),
                    "ratio_normalization": float(
                        authenticated_normalizer.means[model["sample"]]
                    ),
                }
            target = self._resolve_path(model["output_path"])
            directory, prefix = self._fnf_artifact_target(target)
            artifact = training.save(
                directory,
                prefix=prefix,
                metadata={
                    "component": model["sample"],
                    "features": list(self.config.features),
                    "model_name": model["name"],
                    "nuisance_names": list(residual_config.nuisance_names),
                    **(
                        {"base_density_dependencies": base_dependencies}
                        if base_dependencies is not None
                        else {}
                    ),
                    **(
                        {"yield_morph": yield_morph.to_dict()}
                        if yield_morph is not None
                        else {}
                    ),
                },
            )
            if artifact.manifest_path.resolve() != target.resolve():
                raise RuntimeError(
                    "The saved FNF artifact did not match its configured path."
                )
            result[model["sample"]] = FNFModelArtifacts(
                name=model["name"],
                sample=model["sample"],
                residual=training.residual,
                artifact=artifact,
                yield_morph=yield_morph,
                training=training,
            )
        return result

    def configured_fnf_manifests(
        self,
        *,
        require_existing: bool = True,
    ) -> dict[str, Path]:
        """Return sample-to-manifest paths declared by the YAML project."""

        result = {
            model["sample"]: self._resolve_path(model["output_path"])
            for model in self.fnf_model_specifications()
        }
        if require_existing:
            missing = [path for path in result.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(missing[0])
        return result

    def load_fnf_systematics(
        self,
        *,
        device: str | None = None,
        verify: bool = True,
    ) -> dict[str, FNFModelArtifacts]:
        """Load and verify all sample-specific FNF artifacts from YAML paths."""

        from .artifacts import ArtifactManifest
        from .fnf import (
            FactorizableResidualStack,
            FNFArtifact,
            LogQuadraticYieldMorph,
        )

        result: dict[str, FNFModelArtifacts] = {}
        for model in self.fnf_model_specifications():
            configured, training_config = self.fnf_configs(model["name"])
            artifact = FNFArtifact.load(
                self._resolve_path(model["output_path"]),
                verify=verify,
            )
            payload = json.loads(artifact.model_path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            manifest_metadata = ArtifactManifest.load(artifact.manifest_path).metadata
            expected_metadata = {
                "component": model["sample"],
                "features": list(self.config.features),
                "model_name": model["name"],
                "nuisance_names": list(configured.nuisance_names),
            }
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    raise ValueError(
                        f"FNF artifact metadata {key!r} does not match "
                        f"configured model {model['name']!r}."
                    )
            if metadata.get("base_density_dependencies") != manifest_metadata.get(
                "base_density_dependencies"
            ):
                raise ValueError(
                    f"FNF artifact base-density dependencies do not match "
                    f"configured model {model['name']!r}."
                )
            residual = FactorizableResidualStack.load(
                artifact.manifest_path,
                device=(
                    training_config.device
                    if device is None
                    else self._resolve_device(device)
                ),
                verify=verify,
            )
            yield_payload = metadata.get("yield_morph")
            yield_morph = (
                None
                if yield_payload is None
                else LogQuadraticYieldMorph.from_dict(yield_payload)
            )
            result[model["sample"]] = FNFModelArtifacts(
                name=model["name"],
                sample=model["sample"],
                residual=residual,
                artifact=artifact,
                yield_morph=yield_morph,
            )
        return result

    def build_fnf_runtime_systematics(
        self,
        *,
        reference_density: Any,
        ratios: RatioSetTrainingArtifacts,
        reference_manifest: str | Path,
        device: str | None = None,
    ) -> dict[str, Any]:
        """Bind configured residuals to their normalized nominal densities."""

        from .artifacts import ArtifactManifest, sha256_file
        from .fnf import FactorizableDensity
        from .fnf_runtime import (
            FNFSystematic,
            NativeProcessDensity,
            load_verified_reference_density,
        )
        from .native_ratios import load_native_ratio_torch_ensemble

        loaded = self.load_fnf_systematics(device=device)
        missing = set(loaded).difference(ratios.training)
        if missing:
            raise KeyError(
                f"Missing nominal ratio artifacts for FNF samples {sorted(missing)}."
            )
        reference_manifest_path = Path(reference_manifest)
        if not reference_manifest_path.is_file():
            raise FileNotFoundError(reference_manifest_path)
        expected_normalizer_manifest = ratios.normalizer_path.with_suffix(
            ratios.normalizer_path.suffix + ".manifest.json"
        )
        if (
            expected_normalizer_manifest.resolve()
            != Path(ratios.normalizer_manifest).resolve()
        ):
            raise ValueError(
                "Ratio artifacts do not bind their declared normalizer manifest."
            )
        authenticated_normalizer = RatioNormalizer.load(ratios.normalizer_path)
        if (
            authenticated_normalizer.means != ratios.normalizer.means
            or authenticated_normalizer.standard_errors
            != ratios.normalizer.standard_errors
            or authenticated_normalizer.metadata != ratios.normalizer.metadata
        ):
            raise ValueError(
                "The live ratio normalizer disagrees with its checked artifact."
            )
        runtime_device = "cpu" if device is None else self._resolve_device(device)
        authenticated_reference = load_verified_reference_density(
            reference_manifest_path,
            features=tuple(self.config.features),
            device=runtime_device,
            live_density=reference_density,
        )
        authenticated_ratios = {}
        for sample in loaded:
            manifest_path = ratios.training[sample].manifest_path
            authenticated_ratios[sample] = load_native_ratio_torch_ensemble(
                manifest_path,
                device=runtime_device,
                expected_features=tuple(self.config.features),
                live_ensemble=ratios.evaluators[sample],
            )
        reference_digest = sha256_file(reference_manifest_path)
        result = {}
        for sample, artifact in loaded.items():
            manifest = ArtifactManifest.load(artifact.manifest_path)
            dependencies = manifest.metadata.get("base_density_dependencies")
            ratio_manifest = Path(ratios.training[sample].manifest_path)
            if not isinstance(dependencies, Mapping):
                raise ValueError(
                    f"FNF artifact for {sample!r} has no authenticated "
                    "base-density dependencies."
                )
            if dependencies.get("reference_manifest_sha256") != reference_digest:
                raise ValueError(
                    f"FNF artifact for {sample!r} was trained with a different "
                    "reference-flow manifest."
                )
            if dependencies.get("ratio_manifest_sha256") != sha256_file(ratio_manifest):
                raise ValueError(
                    f"FNF artifact for {sample!r} was trained with a different "
                    "native ratio manifest."
                )
            if not np.isclose(
                float(dependencies.get("ratio_normalization", np.nan)),
                float(authenticated_normalizer.means[sample]),
                rtol=0.0,
                atol=0.0,
            ):
                raise ValueError(
                    f"FNF artifact for {sample!r} was trained with a different "
                    "ratio normalization."
                )
            base = NativeProcessDensity(
                reference_density=authenticated_reference,
                ratio=authenticated_ratios[sample],
                ratio_normalization=authenticated_normalizer.means[sample],
            )
            result[sample] = FNFSystematic(
                density=FactorizableDensity(
                    residual=artifact.residual,
                    base_density=base,
                    base_torch_log_prob=base.torch_log_prob,
                ),
                yield_morph=artifact.yield_morph,
            )
        return result

    def _validate_configured_fnf_runtime(
        self,
        supplied: Mapping[str, Any] | None,
        *,
        operation: str,
    ) -> dict[str, Any]:
        expected = {model["sample"] for model in self.fnf_model_specifications()}
        if supplied is None:
            if expected:
                raise ValueError(
                    f"{operation} would ignore configured FNF models for "
                    f"{sorted(expected)}; supply fnf_systematics explicitly."
                )
            return {}
        parsed = dict(supplied)
        if expected and set(parsed) != expected:
            raise ValueError(
                f"{operation} FNF components must match the YAML project: "
                f"expected {sorted(expected)}, found {sorted(parsed)}."
            )
        return parsed

    def design_distribution(self, name: str) -> Any:
        """Construct a normalized rho/nu/kappa design from configuration."""

        from .bayes import BoxUniform, IndependentNormal

        designs = self._bayesian()["design_distributions"]
        if name not in designs:
            raise KeyError(f"Unknown Bayesian design {name!r}.")
        specification = designs[name]
        kind = specification["kind"]
        if kind == "independent_normal":
            return IndependentNormal(
                mean=np.asarray(specification["mean"], dtype=np.float64),
                scale=np.asarray(specification["scale"], dtype=np.float64),
            )
        if kind == "box_uniform":
            return BoxUniform(
                low=np.asarray(specification["low"], dtype=np.float64),
                high=np.asarray(specification["high"], dtype=np.float64),
            )
        if kind == "registry":
            key = specification["registry_key"]
            if key not in self.registry:
                raise KeyError(f"No distribution is registered as {key!r}.")
            distribution = self.registry[key]
            if not hasattr(distribution, "sample") or not hasattr(
                distribution, "log_prob"
            ):
                raise TypeError(
                    "A registered design distribution must provide sample() "
                    "and normalized log_prob()."
                )
            return distribution
        raise ValueError(f"Unsupported design distribution kind {kind!r}.")

    def proposal_dataset(self, name: str) -> Any:
        """Materialize one proposal-sampled theta/observation dataset."""

        from .bayes import ProposalDataset

        bayesian = self._bayesian()
        datasets = bayesian["datasets"]
        if name not in datasets:
            raise KeyError(f"Unknown Bayesian dataset {name!r}.")
        specification = datasets[name]
        theta_features = tuple(bayesian["theta_features"])
        columns = theta_features + self.config.features
        source = self.data_source(specification, features=columns)
        batch = source.materialize(
            batch_size=int(specification.get("batch_size", 65_536))
        )
        theta_columns = len(theta_features)
        return ProposalDataset(
            theta=batch.values[:, :theta_columns],
            observation=batch.values[:, theta_columns:],
            simulation_ids=batch.row_ids,
            design=name,
            parameter_names=theta_features,
            observation_names=self.config.features,
            split_values=(
                None
                if specification.get("split_column") is None
                else batch.columns[specification["split_column"]]
            ),
            log_density=(
                None
                if specification.get("log_density_column") is None
                else batch.columns[specification["log_density_column"]]
            ),
            metadata={
                "source": {
                    key: value
                    for key, value in specification.items()
                    if key not in {"path", "paths"}
                }
            },
        )

    def dual_training_data(self) -> Any:
        """Load the rho/nu/kappa proposal samples for :class:`DualTrainer`."""

        from .bayes import DualTrainingData

        datasets = self._bayesian()["datasets"]
        rho_flow = self.proposal_dataset("rho")
        if "rho_residual" in datasets:
            rho_ratio = self.proposal_dataset("rho_residual")
        else:
            from .bayes import group_train_validation_split

            original = rho_flow
            if original.split_values is None:
                training_indices = np.arange(len(original.theta), dtype=np.int64)
                evaluation_indices = np.empty(0, dtype=np.int64)
            else:
                training_indices = np.flatnonzero(original.split_values == "train")
                evaluation_indices = np.flatnonzero(original.split_values != "train")
                if len(training_indices) < 2:
                    raise ValueError(
                        "rho requires at least two 'train' rows when "
                        "rho_residual is omitted."
                    )
            split = group_train_validation_split(
                original.simulation_ids[training_indices],
                validation_fraction=0.5,
                seed=int(
                    self._bayesian()["posterior_ratio"]["training"].get("seed", 0)
                ),
            )
            rho_flow = original.subset(
                np.concatenate(
                    [
                        training_indices[split.training_indices],
                        evaluation_indices,
                    ]
                )
            )
            rho_ratio = original.subset(
                np.concatenate(
                    [
                        training_indices[split.validation_indices],
                        evaluation_indices,
                    ]
                )
            )
        return DualTrainingData(
            rho_flow=rho_flow,
            rho_ratio=rho_ratio,
            nu_flow=self.proposal_dataset("nu"),
            kappa_ratio=self.proposal_dataset("kappa"),
            validation=(
                self.proposal_dataset("validation")
                if "validation" in datasets
                else None
            ),
        )

    def train_dual(
        self,
        *,
        backend: Any | None = None,
        rho: Any | None = None,
        seed: int | None = None,
        ratio_backends: Mapping[str, Any] | None = None,
    ) -> Any:
        """Train and package the five dual artifacts from configuration."""

        from .bayes import DualTrainer, train_native_dual

        bayesian = self._bayesian()
        training_seed = (
            int(bayesian["posterior_flow"]["training"].get("seed", 0))
            if seed is None
            else int(seed)
        )
        training_data = self.dual_training_data()
        design = self.design_distribution("rho") if rho is None else rho
        metadata = {
            "features": list(self.config.features),
            "theta_features": list(bayesian["theta_features"]),
            "output_bundle": bayesian["output_bundle"],
        }
        if backend is None:
            return train_native_dual(
                training_data,
                rho=design,
                bayesian_config=bayesian,
                observation_features=self.config.features,
                base_directory=self.base_directory,
                seed=training_seed,
                ratio_backends=ratio_backends,
                metadata=metadata,
            )
        if ratio_backends:
            raise ValueError(
                "ratio_backends are configured through NativeDualBackend; "
                "do not combine them with an injected backend."
            )
        return DualTrainer(backend=backend, seed=training_seed).fit(
            training_data,
            rho=design,
            defensive_epsilon=float(bayesian.get("defensive_epsilon", 0.0)),
            metadata=metadata,
        )

    def intensity_model(self) -> IntensityModel:
        return IntensityModel.from_config(self._frequentist())

    def _translate_flow(self, flow: Mapping[str, Any]) -> tuple[Any, Any]:
        from .flows import FlowConfig, FlowTrainingConfig

        training = flow["training"]
        device = training.get("device", "cpu")
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        return (
            FlowConfig(
                n_features=len(self.config.features),
                flow_type=flow["architecture"],
                num_transforms=int(flow["n_coupling_layers"]),
                hidden_features=int(flow["hidden_features"]),
                num_blocks=int(flow["hidden_layers"]),
                num_bins=int(flow.get("spline_num_bins", 8)),
                tail_bound=float(flow.get("spline_tail_bound", 4.0)),
                dropout_probability=float(flow.get("dropout_probability", 0.0)),
                max_log_scale=float(flow.get("scale_clip", 2.0)),
            ),
            FlowTrainingConfig(
                epochs=int(training["epochs"]),
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                validation_fraction=float(training.get("validation_fraction", 0.2)),
                patience=int(training.get("early_stopping_patience", 20)),
                learning_rate_factor=float(training.get("lr_reduction_factor", 0.5)),
                device=device,
            ),
        )

    def flow_configs(self) -> tuple[Any, Any]:
        """Translate the reference-flow schema to native config objects."""

        return self._translate_flow(self._frequentist()["flow"])

    def train_reference(
        self,
        *,
        max_events: int | None = None,
        seed: int | None = None,
        validation_source: DataSource | None = None,
    ) -> ReferenceTrainingArtifacts:
        """Train, checkpoint, diagnose, and export the reference flow.

        Diagnostics use ``validation_source`` when supplied. Otherwise they
        use the trainer's deterministic internal validation holdout, or the
        training rows when validation splitting is disabled. The internal
        holdout participates in early stopping and is not independent
        paper-level validation.
        """

        from .flows import FlowTrainer

        section = self._frequentist()
        configured_training = section["flow"]["training"]
        limit = (
            max_events
            if max_events is not None
            else configured_training.get("max_events")
        )
        batch = self.reference_source().materialize(
            batch_size=int(section["reference"].get("batch_size", 65_536)),
            max_events=limit,
        )
        flow_config, training_config = self.flow_configs()
        training_seed = int(
            configured_training.get("seed", 0) if seed is None else seed
        )
        result = FlowTrainer(flow_config, training_config).fit(
            batch.values,
            features=self.config.features,
            weights=batch.weights,
            seed=training_seed,
        )
        if validation_source is None and training_config.validation_fraction > 0:
            _, diagnostic_indices = FlowTrainer.random_split_indices(
                len(batch.values),
                training_config.validation_fraction,
                rng=np.random.default_rng(training_seed),
            )
            diagnostic_values = batch.values[diagnostic_indices]
            diagnostic_weights = batch.weights[diagnostic_indices]
            diagnostic_source = "internal_training_holdout"
        elif validation_source is None:
            diagnostic_values = batch.values
            diagnostic_weights = batch.weights
            diagnostic_source = "training_sample_no_validation_split"
        else:
            validation_batch = validation_source.materialize(max_events=limit)
            diagnostic_values = validation_batch.values
            diagnostic_weights = validation_batch.weights
            diagnostic_source = "external_validation_source"
        if not len(diagnostic_values):
            raise ValueError("Reference-flow diagnostics require validation events.")
        directory = self.output_directory / "reference"
        checkpoint, checkpoint_manifest = result.save_checkpoint(
            directory / "reference_flow.pt"
        )
        bundle = result.flow.export_onnx(
            directory,
            prefix="reference_flow",
            example_values=diagnostic_values[: min(32, len(diagnostic_values))],
            opset_version=int(section["flow"].get("onnx_opset", 17)),
        )
        parity = bundle.parity(
            result.flow,
            diagnostic_values[: min(512, len(diagnostic_values))],
        )
        for report in parity.values():
            report.assert_close()
        parity_payload = {name: report.to_dict() for name, report in parity.items()}
        parity_path = directory / "reference_flow.onnx_parity.json"
        parity_path.write_text(
            json.dumps(parity_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        from .artifacts import ArtifactManifest

        manifest = ArtifactManifest.load(bundle.manifest_path)
        manifest.add_file(
            parity_path,
            root=bundle.manifest_path.parent,
            kind="onnx-parity",
        )
        manifest.metadata["onnx_parity"] = parity_payload
        manifest.metadata["validation"] = {
            "rows": int(len(diagnostic_values)),
            "source": diagnostic_source,
        }
        manifest.write(bundle.manifest_path)
        from .flow_diagnostics import diagnose_flow

        diagnostics = diagnose_flow(
            result.flow,
            diagnostic_values,
            weights=diagnostic_weights,
            n_generated=min(50_000, len(diagnostic_values)),
            rng=np.random.default_rng(training_seed + 1),
        )
        diagnostics_report, diagnostics_manifest = diagnostics.save(
            directory / "diagnostics"
        )
        return ReferenceTrainingArtifacts(
            training=result,
            checkpoint_path=checkpoint,
            checkpoint_manifest=checkpoint_manifest,
            onnx_bundle=bundle,
            onnx_parity=parity_payload,
            diagnostics=diagnostics,
            diagnostics_report=diagnostics_report,
            diagnostics_manifest=diagnostics_manifest,
        )

    def ratio_config(self) -> Any:
        from .ratios import RatioTrainingConfig

        ratio = self._frequentist()["ratios"]
        training = ratio["training"]
        diagnostics = ratio.get("diagnostics", {})
        enabled_diagnostics = []
        if diagnostics.get("overtraining", True):
            enabled_diagnostics.append("overfit")
        if diagnostics.get("calibration", True):
            enabled_diagnostics.append("calibration")
        if diagnostics.get("reweighting", True):
            enabled_diagnostics.append("reweighting")
        if diagnostics.get("normalization", True):
            enabled_diagnostics.append("normalization")
        return RatioTrainingConfig(
            ensemble_size=int(ratio.get("ensemble_size", 1)),
            hidden_layers=int(training["hidden_layers"]),
            neurons=int(training["neurons"]),
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            learning_rate_factor=float(training.get("lr_reduction_factor", 0.5)),
            calibration=bool(ratio.get("calibration", False)),
            calibration_type=str(ratio.get("calibration_type", "isotonic")),
            calibration_bins=int(ratio.get("calibration_bins", 40)),
            validation_fraction=float(training.get("validation_fraction", 0.1)),
            holdout_fraction=float(training.get("holdout_fraction", 0.3)),
            patience=int(training.get("early_stopping_patience", 30)),
            activation=str(training.get("activation", "swish")),
            seed=int(training.get("seed", 0)),
            run_diagnostics=bool(enabled_diagnostics),
            backend_options={"diagnostics": {"methods": enabled_diagnostics}},
        )

    def train_ratios(
        self,
        reference: Sampler,
        *,
        backend: Any | None = None,
        max_events_per_sample: int | None = None,
        denominator_events: int | None = None,
        normalization_events: int = 100_000,
        seed: int = 0,
    ) -> RatioSetTrainingArtifacts:
        """Train all nominal ratios against fresh frozen-flow samples."""

        from .native_ratios import NativeRatioBackend
        from .ratios import RatioTrainer

        configured_backend = self._frequentist()["ratios"].get("backend", "native")
        if backend is None:
            if configured_backend != "native":
                raise ValueError(
                    "Frequentist ratio training supports only the native backend."
                )
            backend = NativeRatioBackend(
                device=self._resolve_device(
                    str(
                        self._frequentist()["ratios"]["training"].get(
                            "device",
                            "cpu",
                        )
                    )
                ),
                opset_version=int(self._frequentist()["ratios"].get("onnx_opset", 17)),
            )
        normalization_mode = self._frequentist()["ratios"].get(
            "normalization", "independent_reference_mean"
        )
        if normalization_mode != "independent_reference_mean":
            raise NotImplementedError(
                "Conditional ratio normalization is a Bayesian workflow; "
                "frequentist ratios require independent_reference_mean."
            )
        training: dict[str, Any] = {}
        independent_target_holdouts: dict[
            str,
            tuple[np.ndarray, np.ndarray],
        ] = {}
        rng = np.random.default_rng(seed)
        ratio_config = self.ratio_config()
        sample_specifications = {
            sample["name"]: sample["source"]
            for sample in self._frequentist()["samples"]
        }
        for name, source in self.sample_sources().items():
            target = source.materialize(max_events=max_events_per_sample)
            target_split, target_groups = self._ratio_partition_metadata(
                sample_specifications[name],
                target,
            )
            if (
                sample_specifications[name].get("split_column") is not None
                and target_split is not None
            ):
                holdout = np.asarray(target_split).reshape(-1) == "holdout"
                if np.any(holdout):
                    independent_target_holdouts[name] = (
                        np.asarray(target.values[holdout], dtype=np.float32).copy(),
                        np.asarray(target.weights[holdout], dtype=np.float64).copy(),
                    )
            count = (
                len(target.values)
                if denominator_events is None
                else int(denominator_events)
            )
            denominator = np.asarray(reference.sample(count, rng=rng), dtype=np.float32)
            training[name] = RatioTrainer(backend, ratio_config).fit(
                target.values,
                denominator,
                features=self.config.features,
                output_directory=self.output_directory / "ratios" / name,
                numerator_weights=target.weights,
                numerator_name=name,
                denominator_name="reference_flow",
                numerator_split=target_split,
                numerator_groups=target_groups,
            )
        normalization_values = np.asarray(
            reference.sample(int(normalization_events), rng=rng),
            dtype=np.float32,
        )
        normalizer = RatioNormalizer.fit(
            {
                name: result.ensemble(normalization_values)
                for name, result in training.items()
            },
            metadata={
                "mode": "independent_reference_flow",
                "seed": int(seed),
            },
        )
        normalizer_path, normalizer_manifest = normalizer.write(
            self.output_directory / "ratios" / "ratio_normalization.json"
        )
        diagnostics: dict[str, RatioValidationReport] = {}
        if ratio_config.run_diagnostics:
            from .ratio_diagnostics import diagnose_ratio_validation

            for name, (
                target_values,
                target_weights,
            ) in independent_target_holdouts.items():
                reference_values = np.asarray(
                    reference.sample(len(target_values), rng=rng),
                    dtype=np.float32,
                )
                diagnostics[name] = diagnose_ratio_validation(
                    ensemble=training[name].ensemble,
                    target_values=target_values,
                    reference_values=reference_values,
                    target_weights=target_weights,
                    reference_weights=None,
                    features=self.config.features,
                    normalization=normalizer.means[name],
                    bins=ratio_config.diagnostic_bins,
                    output_directory=(
                        self.output_directory / "ratios" / name / "ensemble_validation"
                    ),
                )
        return RatioSetTrainingArtifacts(
            training=training,
            normalizer=normalizer,
            normalization_events=len(normalization_values),
            normalizer_path=normalizer_path,
            normalizer_manifest=normalizer_manifest,
            diagnostics=diagnostics,
        )

    def build_configured_asimov(
        self,
        *,
        reference: Sampler,
        ratios: Mapping[str, RatioEvaluator],
        normalizer: RatioNormalizer | None = None,
        point: Mapping[str, float] | None = None,
        n_events: int | None = None,
        seed: int = 0,
        write: bool = True,
        systematics: Mapping[str, list[Any] | tuple[Any, ...]] | None = None,
        fnf_systematics: Mapping[str, Any] | None = None,
    ) -> Any:
        """Build the configured direct Asimov and optionally write Parquet."""

        section = self._frequentist()
        configuration = section.get("asimov")
        if configuration is None:
            raise ValueError("This project has no Asimov configuration.")
        if normalizer is None and "normalization_source" in configuration:
            source_spec = configuration["normalization_source"]
            batch = self.data_source(source_spec).materialize(
                batch_size=int(source_spec.get("batch_size", 65_536))
            )
            normalizer = RatioNormalizer.fit(
                {
                    name: np.asarray(evaluator(batch.values))
                    for name, evaluator in ratios.items()
                },
                batch.weights,
                metadata={
                    "mode": "configured_independent_reference",
                    "rows": len(batch.values),
                },
            )
        result = self.asimov_builder(
            reference=reference,
            ratios=ratios,
            normalizer=normalizer,
            systematics=systematics,
            fnf_systematics=fnf_systematics,
        ).build(
            (configuration["parameter_point"] if point is None else point),
            n_events=int(configuration["n_events"] if n_events is None else n_events),
            seed=int(seed),
            normalization="fixed" if normalizer is not None else "sample",
        )
        if write:
            result.events.write_parquet(
                self._resolve_path(configuration["output_path"])
            )
        return result

    def train_systematics(
        self,
        *,
        backend: Any | None = None,
        max_events: int | None = None,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        """Train every configured up/nominal and down/nominal ratio."""

        from .native_ratios import NativeRatioBackend
        from .ratios import RatioTrainer
        from .systematics import SystematicSpecification, SystematicsTrainer

        section = self._frequentist()
        configurations = section.get("systematics", ())
        if not configurations:
            return {}
        if backend is None:
            backend = NativeRatioBackend(
                device=self._resolve_device(
                    str(section["ratios"]["training"].get("device", "cpu"))
                ),
                opset_version=int(section["ratios"].get("onnx_opset", 17)),
            )
        nominal_sources = self.sample_sources()
        nominal_specifications = {
            sample["name"]: sample["source"] for sample in section["samples"]
        }
        output: dict[str, dict[str, dict[str, Any]]] = {}
        for systematic in configurations:
            systematic_output: dict[str, dict[str, Any]] = {}
            for variation in systematic["variations"]:
                sample = variation["sample"]
                nominal = nominal_sources[sample].materialize(max_events=max_events)
                up = self.data_source(variation["up"]).materialize(
                    max_events=max_events
                )
                down = self.data_source(variation["down"]).materialize(
                    max_events=max_events
                )
                nominal_split, nominal_groups = self._ratio_partition_metadata(
                    nominal_specifications[sample],
                    nominal,
                )
                up_split, up_groups = self._ratio_partition_metadata(
                    variation["up"],
                    up,
                )
                down_split, down_groups = self._ratio_partition_metadata(
                    variation["down"],
                    down,
                )
                nominal_sum = float(np.sum(nominal.weights))
                if nominal_sum <= 0:
                    raise ValueError(
                        f"Nominal systematic sample {sample!r} has "
                        "non-positive total weight."
                    )
                inferred_yield_up = float(np.sum(up.weights)) / nominal_sum
                inferred_yield_down = float(np.sum(down.weights)) / nominal_sum
                rate_specification = SystematicSpecification(
                    parameter=systematic["parameter"],
                    component=sample,
                    yield_up=variation.get("yield_up", inferred_yield_up),
                    yield_down=variation.get("yield_down", inferred_yield_down),
                    interpolation=systematic.get("interpolation", "nsbi_code4p"),
                )
                trained = SystematicsTrainer(
                    RatioTrainer(backend, self.ratio_config())
                ).fit_variation(
                    nominal=nominal.values,
                    up=up.values,
                    down=down.values,
                    parameter=systematic["parameter"],
                    component=sample,
                    output_dir=(
                        self.output_directory
                        / "systematics"
                        / systematic["name"]
                        / sample
                    ),
                    features=self.config.features,
                    nominal_weights=nominal.weights,
                    up_weights=up.weights,
                    down_weights=down.weights,
                    nominal_split=nominal_split,
                    up_split=up_split,
                    down_split=down_split,
                    nominal_groups=nominal_groups,
                    up_groups=up_groups,
                    down_groups=down_groups,
                )
                systematic_output[sample] = {
                    **trained,
                    "yield_up": rate_specification.yield_up,
                    "yield_down": rate_specification.yield_down,
                    "yield_source": {
                        "up": (
                            "configured"
                            if "yield_up" in variation
                            else "integrated_mc_weights"
                        ),
                        "down": (
                            "configured"
                            if "yield_down" in variation
                            else "integrated_mc_weights"
                        ),
                    },
                    "parameter": rate_specification.parameter,
                    "interpolation": rate_specification.interpolation,
                }
            output[systematic["name"]] = systematic_output
        return output

    def build_systematic_modifiers(
        self,
        asimov: Any,
        training: Mapping[str, Mapping[str, Mapping[str, Any]]],
        *,
        directory: str | Path | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Evaluate trained variation ratios on one workspace quadrature."""

        from .systematics import SystematicAnchor

        root = (
            self.output_directory / "workspace" / "systematics"
            if directory is None
            else Path(directory)
        )
        modifiers: dict[str, list[dict[str, Any]]] = {}
        for systematic_name, samples in training.items():
            for sample, trained in samples.items():
                up = np.asarray(
                    trained["up"].ensemble(asimov.events.values),
                    dtype=np.float64,
                )
                down = np.asarray(
                    trained["down"].ensemble(asimov.events.values),
                    dtype=np.float64,
                )
                nominal_measure = (
                    asimov.reference_weights * asimov.normalized_ratios[sample]
                )
                for label, values in (("up", up), ("down", down)):
                    partition = float(np.sum(nominal_measure * values))
                    if not np.isfinite(partition) or partition <= 0:
                        raise ValueError(
                            f"Systematic {systematic_name}/{sample} {label} "
                            "has a non-positive shape partition."
                        )
                    values /= partition
                anchor = SystematicAnchor(
                    parameter=trained["parameter"],
                    component=sample,
                    ratio_up=up,
                    ratio_down=down,
                    yield_up=float(trained["yield_up"]),
                    yield_down=float(trained["yield_down"]),
                    interpolation=trained["interpolation"],
                )
                modifiers.setdefault(sample, []).append(
                    anchor.write_workspace_modifier(root / systematic_name / sample)
                )
        return modifiers

    def build_runtime_systematics(
        self,
        training: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> dict[str, list[Any]]:
        """Bind trained up/down ensembles for Asimov and toy generation."""

        from .systematics import RuntimeSystematic

        runtime: dict[str, list[Any]] = {}
        for samples in training.values():
            for sample, trained in samples.items():
                runtime.setdefault(sample, []).append(
                    RuntimeSystematic(
                        parameter=trained["parameter"],
                        component=sample,
                        ratio_up=trained["up"].ensemble,
                        ratio_down=trained["down"].ensemble,
                        yield_up=float(trained["yield_up"]),
                        yield_down=float(trained["yield_down"]),
                        interpolation=trained["interpolation"],
                    )
                )
        return runtime

    def write_configured_workspace(
        self,
        result: Any,
        *,
        systematic_modifiers: Mapping[str, list[Mapping[str, Any]]] | None = None,
        reference_manifest: str | Path | None = None,
        ratio_manifests: Mapping[str, str | Path] | None = None,
        fnf_manifests: Mapping[str, str | Path] | None = None,
    ) -> Any:
        """Write the native hNSBI JSON workspace configured by the YAML project."""

        from .workspace import write_workspace

        workspace_config = self._frequentist()["workspace"]
        target = self._resolve_path(workspace_config["output_path"])
        template_workspace = None
        if "template_path" in workspace_config:
            template_path = self._resolve_path(workspace_config["template_path"])
            try:
                template_workspace = json.loads(
                    template_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Could not load workspace template {template_path}."
                ) from exc
            if not isinstance(template_workspace, dict):
                raise ValueError("Workspace template must be a JSON object.")
        pois = [
            parameter["name"]
            for parameter in self._frequentist()["parameters"]
            if parameter["role"] == "poi"
        ]
        if fnf_manifests is None and self.fnf_model_specifications():
            fnf_manifests = self.configured_fnf_manifests()
        return write_workspace(
            result=result,
            intensity=self.intensity_model(),
            output_dir=target.parent,
            workspace_filename=target.name,
            measurement=workspace_config["measurement"],
            poi=pois[0],
            channel=workspace_config["channel"],
            systematic_modifiers=systematic_modifiers,
            reference_manifest=reference_manifest,
            ratio_manifests=ratio_manifests,
            fnf_manifests=fnf_manifests,
            template_workspace=template_workspace,
        )

    def generate_configured_toys(
        self,
        *,
        generator: ToyGenerator | None = None,
        reference: Sampler | None = None,
        ratios: Mapping[str, RatioEvaluator] | None = None,
        normalizer: RatioNormalizer | None = None,
        component_samplers: Mapping[str, Sampler] | None = None,
        systematics: Mapping[str, list[Any] | tuple[Any, ...]] | None = None,
        fnf_systematics: Mapping[str, Any] | None = None,
        seed: int = 0,
        write: bool = True,
    ) -> list[dict[str, Any]]:
        """Run and optionally persist the configured pseudo-experiment campaign."""

        configuration = self._frequentist().get("toys")
        if configuration is None:
            raise ValueError("This project has no toy configuration.")
        if configuration["method"] != "importance_resampling":
            raise NotImplementedError(
                "reference_bootstrap is not an exact component-wise Poisson "
                "toy method; use importance_resampling."
            )
        if generator is None:
            generator = self.toy_generator(
                reference=reference,
                ratios=ratios,
                normalizer=normalizer,
                component_samplers=component_samplers,
                systematics=systematics,
                fnf_systematics=fnf_systematics,
            )
        else:
            expected_fnf = {
                model["sample"] for model in self.fnf_model_specifications()
            }
            observed_fnf = set(getattr(generator, "fnf_systematics", {}))
            if expected_fnf and observed_fnf != expected_fnf:
                raise ValueError(
                    "Configured toy generation would ignore FNF models: "
                    f"expected {sorted(expected_fnf)}, found "
                    f"{sorted(observed_fnf)}."
                )
        output_root = self._resolve_path(
            configuration.get("output_path", self.output_directory / "toys")
        )
        sequence = np.random.SeedSequence(int(seed))
        total = len(configuration["parameter_points"]) * int(
            configuration["toys_per_point"]
        )
        children = iter(sequence.spawn(total))
        records: list[dict[str, Any]] = []
        for point_index, point in enumerate(configuration["parameter_points"]):
            for toy_index in range(int(configuration["toys_per_point"])):
                child = next(children)
                toy_seed = int(child.generate_state(1, dtype=np.uint32)[0])
                result = generator.generate(
                    point,
                    seed=toy_seed,
                    min_pool=int(configuration.get("proposal_events", 4096)),
                )
                path = (
                    output_root
                    / f"point_{point_index:03d}"
                    / f"toy_{toy_index:06d}.parquet"
                )
                if write:
                    result.events.write_parquet(path)
                records.append(
                    {
                        "point_index": point_index,
                        "toy_index": toy_index,
                        "seed": toy_seed,
                        "path": path if write else None,
                        "result": result,
                    }
                )
        return records

    def workspace_runtime(
        self,
        path: str | Path | None = None,
        *,
        fnf_systematics: Mapping[str, Any] | None = None,
        fnf_device: str = "cpu",
        onnx_providers: tuple[str, ...] | None = None,
    ) -> Any:
        """Load a native serialized workspace as an extended likelihood."""

        from .likelihood import ExtendedUnbinnedLikelihood

        workspace_path = (
            self._resolve_path(self._frequentist()["workspace"]["output_path"])
            if path is None
            else Path(path)
        )
        return ExtendedUnbinnedLikelihood.from_workspace(
            workspace_path,
            fnf_systematics=fnf_systematics,
            fnf_device=self._resolve_device(fnf_device),
            onnx_providers=onnx_providers,
        )

    def combined_workspace_runtime(
        self,
        paths: Mapping[str, str | Path],
        *,
        use_configured_paths: bool = True,
    ) -> Any:
        """Load independent channels with shared parameters and constraints."""

        from .multi_workspace import CombinedLikelihood

        likelihoods = {
            name: self.workspace_runtime(
                self.resolve_path(path) if use_configured_paths else Path(path)
            )
            for name, path in paths.items()
        }
        return CombinedLikelihood(likelihoods)

    def asimov_builder(
        self,
        *,
        reference: Sampler,
        ratios: Mapping[str, RatioEvaluator],
        normalizer: RatioNormalizer | None = None,
        systematics: Mapping[str, list[Any] | tuple[Any, ...]] | None = None,
        fnf_systematics: Mapping[str, Any] | None = None,
    ) -> AsimovBuilder:
        fnf = self._validate_configured_fnf_runtime(
            fnf_systematics,
            operation="Asimov construction",
        )
        return AsimovBuilder(
            reference=reference,
            ratios=ratios,
            intensity=self.intensity_model(),
            features=self.config.features,
            normalizer=normalizer,
            systematics=systematics,
            fnf_systematics=fnf,
        )

    def train_nis_asimov(
        self,
        *,
        reference: Sampler,
        ratios: Mapping[str, RatioEvaluator],
        truth_point: Mapping[str, float] | None = None,
        asimov_point: Mapping[str, float] | None = None,
        seed: int | None = None,
        systematics: Mapping[str, list[Any] | tuple[Any, ...]] | None = None,
        fnf_systematics: Mapping[str, Any] | None = None,
    ) -> NISWorkflowArtifacts:
        """Run configured pilot design, proposal training, diagnostics, and NIS.

        ``reference`` must provide both ``sample`` and normalized ``log_prob``.
        The configured design points and defensive epsilon are used verbatim.
        The automatic flow report is evaluated on the weighted pilot target
        used for training; independent validation remains an explicit
        downstream operation.
        """

        from .flow_diagnostics import diagnose_flow
        from .flows import FlowTrainer
        from .nis import (
            DefensiveMixture,
            NISAsimovBuilder,
            NISProposalTrainer,
        )

        section = self._frequentist()
        if "nis" not in section:
            raise ValueError("This project has no NIS configuration.")
        nis = section["nis"]
        fnf = self._validate_configured_fnf_runtime(
            fnf_systematics,
            operation="NIS Asimov construction",
        )
        default_point = section.get("asimov", {}).get("parameter_point")
        if truth_point is None:
            truth_point = default_point
        if asimov_point is None:
            asimov_point = default_point
        if truth_point is None or asimov_point is None:
            raise ValueError(
                "truth_point and asimov_point are required when no configured "
                "Asimov parameter point is available."
            )
        if not hasattr(reference, "log_prob"):
            raise TypeError("The NIS reference must provide normalized log_prob().")
        flow_config, training_config = self._translate_flow(nis["flow"])
        if training_config.validation_fraction <= 0:
            raise ValueError(
                "NIS flow validation_fraction must be positive so ONNX parity "
                "and flow diagnostics use rows excluded from optimization."
            )
        trainer = FlowTrainer(flow_config, training_config)

        class WeightedFlowAdapter:
            def __init__(self) -> None:
                self.training: Any | None = None

            def fit(
                self,
                values: np.ndarray,
                *,
                sample_weights: np.ndarray,
                **_: Any,
            ) -> Any:
                self.training = trainer.fit(
                    values,
                    features=self_features,
                    weights=sample_weights,
                    seed=training_seed,
                )
                return self.training.flow

        self_features = self.config.features
        training_seed = int(
            nis["flow"]["training"].get("seed", 0) if seed is None else seed
        )
        adapter = WeightedFlowAdapter()
        design = NISProposalTrainer(
            reference=reference,
            ratios=ratios,
            intensity=self.intensity_model(),
            trainer=adapter,
            systematics=systematics,
            fnf_systematics=fnf,
        ).fit(
            truth_point=truth_point,
            design_points=nis["design_points"],
            pilot_events=int(nis["pilot_events"]),
            seed=training_seed,
        )
        if adapter.training is None:
            raise RuntimeError("The NIS flow backend returned no training result.")
        training_indices, validation_indices = FlowTrainer.random_split_indices(
            len(design.pilot_values),
            training_config.validation_fraction,
            rng=np.random.default_rng(training_seed),
        )
        validation_values = design.pilot_values[validation_indices]
        validation_weights = design.training_weights[validation_indices]
        validation_provenance = {
            "source": "internal_training_holdout",
            "seed": training_seed,
            "validation_fraction": training_config.validation_fraction,
            "training_rows": int(len(training_indices)),
            "validation_rows": int(len(validation_indices)),
        }
        directory = self._resolve_path(nis["output_path"])
        checkpoint, checkpoint_manifest = adapter.training.save_checkpoint(
            directory / "nis_flow.pt"
        )
        bundle = adapter.training.flow.export_onnx(
            directory,
            prefix="nis_flow",
            example_values=validation_values[: min(32, len(validation_values))],
            opset_version=int(nis["flow"].get("onnx_opset", 17)),
        )
        parity = bundle.parity(
            adapter.training.flow,
            validation_values[: min(512, len(validation_values))],
        )
        for report in parity.values():
            report.assert_close()
        parity_payload = {name: report.to_dict() for name, report in parity.items()}
        parity_path = directory / "nis_flow.onnx_parity.json"
        parity_path.write_text(
            json.dumps(parity_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        from .artifacts import ArtifactManifest

        manifest = ArtifactManifest.load(bundle.manifest_path)
        manifest.add_file(
            parity_path,
            root=bundle.manifest_path.parent,
            kind="onnx-parity",
        )
        manifest.metadata["onnx_parity"] = parity_payload
        manifest.metadata["validation"] = validation_provenance
        manifest.write(bundle.manifest_path)
        diagnostic_rows = min(
            len(validation_values), max(2_000, int(nis["target_events"]))
        )
        validation = diagnose_flow(
            adapter.training.flow,
            validation_values,
            weights=validation_weights,
            n_generated=diagnostic_rows,
            rng=np.random.default_rng(training_seed + 1),
        )
        validation_report, validation_manifest = validation.save(
            directory / "validation", prefix="nis_validation"
        )
        validation_artifact = ArtifactManifest.load(validation_manifest)
        validation_artifact.metadata["validation"] = validation_provenance
        validation_artifact.write(validation_manifest)
        defensive = DefensiveMixture(
            reference=reference,
            reference_density=reference,
            proposal=adapter.training.flow,
            proposal_density=adapter.training.flow,
            epsilon=float(nis["epsilon"]),
        )
        asimov = NISAsimovBuilder(
            proposal=defensive,
            ratios=ratios,
            intensity=self.intensity_model(),
            features=self.config.features,
            systematics=systematics,
            fnf_systematics=fnf,
        ).build(
            asimov_point,
            n_events=int(nis["target_events"]),
            seed=training_seed + 2,
        )
        asimov_path = asimov.events.write_parquet(
            directory / "efficient_asimov.parquet"
        )
        asimov_array_paths = asimov.write_nsbi_arrays(
            directory / "efficient_asimov_arrays"
        )
        return NISWorkflowArtifacts(
            design=design,
            flow_training=adapter.training,
            checkpoint_path=checkpoint,
            checkpoint_manifest=checkpoint_manifest,
            onnx_bundle=bundle,
            validation=validation,
            validation_report=validation_report,
            validation_manifest=validation_manifest,
            validation_provenance=validation_provenance,
            defensive_proposal=defensive,
            asimov=asimov,
            asimov_path=asimov_path,
            asimov_array_paths=asimov_array_paths,
            onnx_parity=parity_payload,
        )

    def toy_generator(
        self,
        *,
        reference: Sampler | None = None,
        ratios: Mapping[str, RatioEvaluator] | None = None,
        normalizer: RatioNormalizer | None = None,
        component_samplers: Mapping[str, Sampler] | None = None,
        systematics: Mapping[str, list[Any] | tuple[Any, ...]] | None = None,
        fnf_systematics: Mapping[str, Any] | None = None,
    ) -> ToyGenerator:
        fnf = self._validate_configured_fnf_runtime(
            fnf_systematics,
            operation="Toy generation",
        )
        return ToyGenerator(
            intensity=self.intensity_model(),
            features=self.config.features,
            reference=reference,
            ratios=ratios,
            normalizer=normalizer,
            component_samplers=component_samplers,
            systematics=systematics,
            fnf_systematics=fnf,
        )
