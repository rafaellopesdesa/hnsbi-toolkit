"""Bind factorizable flows to native likelihoods and workspace artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactManifest
from .flows import FlowOnnxBundle, ReferenceFlow
from .fnf import (
    FactorizableDensity,
    FactorizableResidualStack,
    LogQuadraticYieldMorph,
)
from .native_ratios import load_native_ratio_ensemble


@dataclass
class NativeProcessDensity:
    r"""Normalized process density $p_k=q\,r_k/E_q[r_k]$.

    The object composes the trained reference flow and arithmetic ratio
    ensemble. Its Torch path remains differentiable with respect to input
    features and is therefore suitable as the frozen nominal density used
    while training an FNF residual.
    """

    reference_density: Any
    ratio: Any
    ratio_normalization: float

    def __post_init__(self) -> None:
        value = float(self.ratio_normalization)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("ratio_normalization must be finite and positive.")
        self.ratio_normalization = value

    def log_prob(self, values: Any) -> np.ndarray:
        reference = np.asarray(
            self.reference_density.log_prob(values),
            dtype=np.float64,
        ).reshape(-1)
        ratio = np.asarray(self.ratio(values), dtype=np.float64).reshape(-1)
        if (
            len(reference) != len(ratio)
            or not np.isfinite(reference).all()
            or not np.isfinite(ratio).all()
            or np.any(ratio <= 0)
        ):
            raise ValueError("The reference/ratio density returned invalid values.")
        return reference + np.log(ratio) - math.log(self.ratio_normalization)

    def to(self, device: str) -> NativeProcessDensity:
        """Move every native Torch constituent to one FNF training device."""

        mover = getattr(self.reference_density, "to", None)
        if callable(mover):
            mover(device)
        for member in getattr(self.ratio, "members", ()):
            module = getattr(member, "module", None)
            if module is None:
                continue
            module.to(device)
            member.device = str(device)
        return self

    def torch_log_prob(self, values: Any) -> Any:
        """Return differentiable log density for native Torch artifacts."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Differentiable process densities require hnsbi-toolkit[lhc]."
            ) from exc
        reference_evaluator = getattr(
            self.reference_density,
            "torch_log_prob",
            None,
        )
        if not callable(reference_evaluator):
            raise TypeError(
                "The reference density has no differentiable torch_log_prob()."
            )
        member_evaluators = [
            getattr(member, "torch_log_ratio", None) for member in self.ratio.members
        ]
        if not member_evaluators or any(
            not callable(evaluator) for evaluator in member_evaluators
        ):
            raise TypeError(
                "Every ratio member must expose differentiable torch_log_ratio()."
            )
        member_logs = torch.stack(
            [evaluator(values) for evaluator in member_evaluators],
            dim=0,
        )
        ensemble_log_ratio = torch.logsumexp(member_logs, dim=0) - math.log(
            len(member_evaluators)
        )
        return (
            reference_evaluator(values)
            + ensemble_log_ratio
            - math.log(self.ratio_normalization)
        )


@dataclass
class FNFSystematic:
    """A fitted normalized FNF shape and optional positive yield morph."""

    density: FactorizableDensity
    yield_morph: LogQuadraticYieldMorph | None = None

    def __post_init__(self) -> None:
        if self.yield_morph is None:
            return
        residual = self.density.residual.config
        positions = {name: index for index, name in enumerate(residual.nuisance_names)}
        unknown = set(self.yield_morph.nuisance_names).difference(positions)
        if unknown:
            raise ValueError(
                "FNF yield morph references parameters outside the residual: "
                f"{sorted(unknown)}."
            )
        for name, center, scale in zip(
            self.yield_morph.nuisance_names,
            self.yield_morph.centers,
            self.yield_morph.scales,
            strict=True,
        ):
            index = positions[name]
            center_matches = np.isclose(center, residual.nuisance_centers[index])
            scale_matches = np.isclose(scale, residual.nuisance_scales[index])
            if not center_matches or not scale_matches:
                raise ValueError(
                    f"FNF yield morph coordinates for {name!r} do not match "
                    "the residual model."
                )

    @property
    def parameters(self) -> tuple[str, ...]:
        return self.density.parameters

    def shape_factor(
        self,
        values: Any,
        point: Mapping[str, float],
    ) -> np.ndarray:
        nuisance_point = {name: float(point[name]) for name in self.parameters}
        factor = np.exp(self.density.log_ratio(values, nuisance_point))
        result = np.asarray(factor, dtype=np.float64).reshape(-1)
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError("FNF shape factors must be finite and positive.")
        return result

    def yield_factor(self, point: Mapping[str, float]) -> float:
        if self.yield_morph is None:
            value = 1.0
        else:
            yield_point = {
                name: float(point[name]) for name in self.yield_morph.nuisance_names
            }
            value = self.yield_morph.factor(yield_point)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("FNF yield factors must be finite and positive.")
        return float(value)


@dataclass(frozen=True)
class FNFSupportEvaluation:
    """Finite-support FNF shape normalization and yield multiplier."""

    shape_ratio: np.ndarray
    shape_partition: float
    yield_factor: float


def validate_fnf_systematics(
    intensity: Any,
    fnf_systematics: Mapping[str, Any] | None,
    *,
    anchor_parameters: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Validate component, parameter, API, and modifier-overlap contracts."""

    parsed = dict(fnf_systematics or {})
    components = set(intensity.component_names)
    unknown_components = set(parsed).difference(components)
    if unknown_components:
        raise ValueError(
            "FNF systematics reference unknown components "
            f"{sorted(unknown_components)}."
        )
    known_parameters = {parameter.name for parameter in intensity.parameters}
    anchors = dict(anchor_parameters or {})
    for component, fnf in parsed.items():
        parameters = tuple(getattr(fnf, "parameters", ()))
        unknown_parameters = set(parameters).difference(known_parameters)
        if (
            not parameters
            or len(parameters) != len(set(parameters))
            or unknown_parameters
        ):
            raise ValueError(
                f"FNF for {component!r} has invalid nuisance parameters {parameters}."
            )
        overlap = set(parameters).intersection(anchors.get(component, ()))
        if overlap:
            raise ValueError(
                f"Component {component!r} configures both FNF and anchor "
                f"systematics for {sorted(overlap)}."
            )
        if not callable(getattr(fnf, "shape_factor", None)) or not callable(
            getattr(fnf, "yield_factor", None)
        ):
            raise TypeError(
                "An FNF systematic must provide shape_factor() and yield_factor()."
            )
    return parsed


def evaluate_fnf_on_support(
    fnf: Any,
    *,
    values: Any,
    point: Mapping[str, float],
    nominal_shape: Any,
    integration_weights: Any,
) -> FNFSupportEvaluation:
    """Apply an FNF and normalize it on the supplied nominal-process measure."""

    shape = np.asarray(nominal_shape, dtype=np.float64).reshape(-1)
    weights = np.asarray(integration_weights, dtype=np.float64).reshape(-1)
    event_values = np.asarray(values)
    if (
        event_values.ndim != 2
        or len(event_values) != len(shape)
        or len(weights) != len(shape)
    ):
        raise ValueError("FNF values, shapes, and integration weights must align.")
    if (
        not np.isfinite(shape).all()
        or np.any(shape < 0)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not float(np.sum(weights)) > 0
    ):
        raise ValueError(
            "FNF nominal shapes and integration weights must define a finite "
            "non-negative measure."
        )
    weights = weights / np.sum(weights)
    factor = np.asarray(
        fnf.shape_factor(event_values, point),
        dtype=np.float64,
    ).reshape(-1)
    if (
        len(factor) != len(shape)
        or not np.isfinite(factor).all()
        or np.any(factor <= 0)
    ):
        raise ValueError("FNF shape factors must align and be finite and positive.")
    morphed = shape * factor
    partition = float(np.sum(weights * morphed))
    if not np.isfinite(partition) or partition <= 0:
        raise ValueError("FNF shape has a non-positive finite-support partition.")
    yield_factor = float(fnf.yield_factor(point))
    if not np.isfinite(yield_factor) or yield_factor <= 0:
        raise ValueError("FNF yield factors must be finite and positive.")
    return FNFSupportEvaluation(
        shape_ratio=morphed / partition,
        shape_partition=partition,
        yield_factor=yield_factor,
    )


def _reference_density(
    manifest_path: Path,
    *,
    features: tuple[str, ...],
    device: str,
) -> Any:
    manifest = ArtifactManifest.load(manifest_path)
    manifest.verify(manifest_path.parent)
    if manifest.artifact_type == "reference-flow-onnx-bundle":
        return FlowOnnxBundle.load(
            manifest_path,
            expected_features=features,
        )
    if manifest.artifact_type != "reference-flow-checkpoint":
        raise ValueError(
            "FNF workspaces require a reference-flow checkpoint or ONNX bundle."
        )
    records = [
        record for record in manifest.files if record.kind == "pytorch-checkpoint"
    ]
    if len(records) != 1:
        raise ValueError(
            "Reference-flow checkpoint manifest must contain one checkpoint."
        )
    checkpoint = manifest_path.parent / records[0].path
    return ReferenceFlow.load(
        checkpoint,
        device=device,
        expected_features=features,
    )


def load_verified_reference_density(
    manifest_path: str | Path,
    *,
    features: tuple[str, ...],
    device: str = "cpu",
    live_density: Any | None = None,
) -> Any:
    """Load a checked reference artifact and authenticate an optional live object.

    The returned density always comes from the manifest. When a live density is
    supplied for a same-process workflow, every scientific state element must
    match the hash-verified artifact exactly.
    """

    path = Path(manifest_path)
    recorded = _reference_density(path, features=features, device=device)
    if live_density is None:
        return recorded
    if type(live_density) is not type(recorded):
        raise ValueError(
            "The live reference density disagrees with its reference-flow manifest."
        )
    if isinstance(recorded, ReferenceFlow):
        import torch

        same_structure = (
            live_density.features == recorded.features
            and live_density.context_names == recorded.context_names
            and live_density.config == recorded.config
            and np.array_equal(live_density.scaler.mean, recorded.scaler.mean)
            and np.array_equal(live_density.scaler.scale, recorded.scaler.scale)
        )
        live_state = live_density.model.state_dict()
        recorded_state = recorded.model.state_dict()
        same_state = live_state.keys() == recorded_state.keys() and all(
            torch.equal(
                live_state[name].detach().cpu(),
                recorded_state[name].detach().cpu(),
            )
            for name in live_state
        )
        if not same_structure or not same_state:
            raise ValueError(
                "The live reference density disagrees with its reference-flow manifest."
            )
        return recorded
    if isinstance(recorded, FlowOnnxBundle):
        from .artifacts import sha256_file

        if (
            live_density.features != recorded.features
            or live_density.context_names != recorded.context_names
            or sha256_file(live_density.manifest_path) != sha256_file(path)
        ):
            raise ValueError(
                "The live reference density disagrees with its reference-flow manifest."
            )
        return recorded
    raise ValueError(
        "The live reference density cannot be authenticated against its manifest."
    )


def load_workspace_fnf_runtime_bundle(
    workspace_model: Any,
    *,
    device: str = "cpu",
    providers: tuple[str, ...] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, FNFSystematic]]:
    """Load the checked nominal proposal, ratios, and bound FNF shapes.

    A serialized FNF is only the nuisance residual. Evaluating it also needs
    the nominal process density, reconstructed here from the workspace's
    reference-flow and native ratio manifests.
    """

    if not workspace_model.fnf_manifests:
        return None, {}, {}
    if workspace_model.reference_manifest is None:
        raise ValueError(
            "An FNF workspace requires a reference-flow manifest to reconstruct "
            "its nominal process density."
        )
    missing = set(workspace_model.fnf_manifests).difference(
        workspace_model.ratio_manifests
    )
    if missing:
        raise ValueError(
            "FNF workspace samples are missing native ratio manifests "
            f"{sorted(missing)}."
        )
    reference = _reference_density(
        workspace_model.reference_manifest,
        features=workspace_model.features,
        device=device,
    )
    ratios = {
        component: load_native_ratio_ensemble(
            manifest_path,
            providers=providers,
            expected_features=workspace_model.features,
        )
        for component, manifest_path in workspace_model.ratio_manifests.items()
    }
    result: dict[str, FNFSystematic] = {}
    for component, manifest_path in workspace_model.fnf_manifests.items():
        base = NativeProcessDensity(
            reference_density=reference,
            ratio=ratios[component],
            ratio_normalization=workspace_model.ratio_normalizer.means[component],
        )
        residual = FactorizableResidualStack.load(
            manifest_path,
            device=device,
        )
        manifest = ArtifactManifest.load(manifest_path)
        yield_payload = manifest.metadata.get("yield_morph")
        if yield_payload is None:
            model_record = next(
                (record for record in manifest.files if record.kind == "model-config"),
                None,
            )
            if model_record is not None:
                payload = json.loads(
                    (manifest_path.parent / model_record.path).read_text(
                        encoding="utf-8"
                    )
                )
                yield_payload = payload.get("metadata", {}).get("yield_morph")
        yield_morph = (
            None
            if yield_payload is None
            else LogQuadraticYieldMorph.from_dict(yield_payload)
        )
        result[component] = FNFSystematic(
            density=FactorizableDensity(
                residual=residual,
                base_density=base,
            ),
            yield_morph=yield_morph,
        )
    return reference, ratios, result


def load_workspace_fnf_systematics(
    workspace_model: Any,
    *,
    device: str = "cpu",
    providers: tuple[str, ...] | None = None,
) -> dict[str, FNFSystematic]:
    """Load all checked FNF shapes and their nominal process densities."""

    return load_workspace_fnf_runtime_bundle(
        workspace_model,
        device=device,
        providers=providers,
    )[2]


__all__ = [
    "FNFSupportEvaluation",
    "FNFSystematic",
    "NativeProcessDensity",
    "evaluate_fnf_on_support",
    "load_workspace_fnf_runtime_bundle",
    "load_workspace_fnf_systematics",
    "validate_fnf_systematics",
]
