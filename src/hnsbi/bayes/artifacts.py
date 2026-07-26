"""Portable artifacts for the five-object dual hNPE--hNDE model.

The artifact boundary intentionally starts *after* training. Callers provide
already-exported ONNX graphs; this module records their exact roles, tensor
names, feature order, preprocessing/Jacobian conventions, ensemble reduction,
and provenance in one checksummed manifest.

ONNX Runtime is imported only on the first inference call. Importing
``hnsbi.bayes`` therefore remains NumPy-only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .._version import __version__
from ._array import align_rows, as_2d, as_vector
from .model import DualModel

DUAL_ARTIFACT_NAMES = ("q_phi", "r_p", "q_eta", "r_c", "z_c")
DUAL_MANIFEST_SCHEMA_VERSION = "1"
DUAL_MANIFEST_ARTIFACT_TYPE = "hnsbi.dual-model"

_GRAPH_ROLES = {"log_prob", "inverse", "log_ratio", "log_normalization"}
_ENSEMBLE_REDUCTIONS = {
    "single",
    "arithmetic_mean_ratio",
    "mean_log_ratio",
}
_REDUCTION_ALIASES = {
    "arithmetic_mean": "arithmetic_mean_ratio",
    "geometric_mean": "mean_log_ratio",
    "geometric_mean_ratio": "mean_log_ratio",
}


class DualArtifactError(ValueError):
    """Raised when a dual-model manifest violates its semantic contract."""


class DualArtifactIntegrityError(RuntimeError):
    """Raised when a graph is missing or differs from its recorded checksum."""


class OnnxRuntimeUnavailable(ImportError):
    """Raised when lazy ONNX inference is requested without ONNX Runtime."""


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str | Path) -> str:
    path = PurePosixPath(Path(value).as_posix())
    if path.is_absolute() or not path.parts:
        raise DualArtifactError("ONNX graph paths must be non-empty and relative.")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise DualArtifactError(
            "ONNX graph paths cannot contain '.', '..', or empty components."
        )
    return path.as_posix()


def _relative_to(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise DualArtifactError(
            f"ONNX graph {path} is outside manifest root {root}."
        ) from exc
    return _safe_relative_path(relative)


def _normalize_json(value: Any, *, location: str) -> Any:
    """Return a detached, strict-JSON representation of *value*."""

    if isinstance(value, np.ndarray):
        return _normalize_json(value.tolist(), location=location)
    if isinstance(value, np.generic):
        return _normalize_json(value.item(), location=location)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise DualArtifactError(
                    f"{location} mapping keys must be non-empty strings."
                )
            result[key] = _normalize_json(
                item,
                location=f"{location}.{key}",
            )
        return result
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item, location=f"{location}[]") for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise DualArtifactError(f"{location} contains a non-finite float.")
        return value
    raise DualArtifactError(
        f"{location} contains non-JSON value of type {type(value).__name__}."
    )


def _json_mapping(value: Mapping[str, Any] | None, *, location: str) -> dict[str, Any]:
    normalized = _normalize_json(dict(value or {}), location=location)
    if not isinstance(normalized, dict):  # pragma: no cover - defensive
        raise DualArtifactError(f"{location} must be a mapping.")
    return normalized


def _string_mapping(
    value: Mapping[str, str],
    *,
    location: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for semantic, onnx_name in value.items():
        if (
            not isinstance(semantic, str)
            or not semantic
            or not isinstance(onnx_name, str)
            or not onnx_name
        ):
            raise DualArtifactError(
                f"{location} semantic keys and ONNX names must be non-empty strings."
            )
        result[semantic] = onnx_name
    if not result:
        raise DualArtifactError(f"{location} cannot be empty.")
    if len(result.values()) != len(set(result.values())):
        raise DualArtifactError(f"{location} contains duplicate ONNX tensor names.")
    return result


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - normalized earlier
        raise DualArtifactError("Manifest is not strict JSON.") from exc
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class FeatureSignature:
    """An ordered physical-space feature signature."""

    names: tuple[str, ...]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if not names:
            raise DualArtifactError("Feature signatures cannot be empty.")
        if any(
            not isinstance(name, str) or not name or name.strip() != name
            for name in names
        ):
            raise DualArtifactError("Feature names must be non-empty, trimmed strings.")
        if len(names) != len(set(names)):
            raise DualArtifactError("Feature names must be unique and ordered.")
        dtype = str(self.dtype)
        if dtype not in {"float32", "float64"}:
            raise DualArtifactError(
                "Bayesian ONNX feature dtypes must be 'float32' or 'float64'."
            )
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "dtype", dtype)

    @property
    def dimension(self) -> int:
        """Number of ordered features."""

        return len(self.names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "dtype": self.dtype,
            "dimension": self.dimension,
        }

    @classmethod
    def from_value(
        cls,
        value: FeatureSignature | Mapping[str, Any] | Sequence[str],
    ) -> FeatureSignature:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            signature = cls(
                names=tuple(value["names"]),
                dtype=str(value.get("dtype", "float32")),
            )
            declared = value.get("dimension")
            if declared is not None and int(declared) != signature.dimension:
                raise DualArtifactError(
                    "Declared signature dimension does not match feature names."
                )
            return signature
        if isinstance(value, str):
            raise DualArtifactError(
                "A signature must be a sequence of names, not one string."
            )
        return cls(names=tuple(value))


@dataclass(frozen=True)
class TransformSpec:
    """Physical/model-space transforms and their density Jacobian convention.

    ``placement`` is ``"embedded"`` when the listed operations and Jacobian
    accounting are part of each deployable ONNX graph. ``"identity"`` records
    that no numerical transform is applied. Lazy adapters deliberately reject
    unspecified external transform code.
    """

    forward: tuple[dict[str, Any], ...]
    inverse: tuple[dict[str, Any], ...]
    log_abs_det_jacobian: dict[str, Any]
    placement: str = "embedded"

    def __post_init__(self) -> None:
        forward_value: Any = self.forward
        inverse_value: Any = self.inverse
        if isinstance(forward_value, Mapping):
            forward_value = (forward_value,)
        if isinstance(inverse_value, Mapping):
            inverse_value = (inverse_value,)
        forward = _normalize_json(forward_value, location="transform.forward")
        inverse = _normalize_json(inverse_value, location="transform.inverse")
        if any(not isinstance(item, dict) for item in (*forward, *inverse)):
            raise DualArtifactError(
                "Transform forward and inverse steps must be mappings."
            )
        jacobian = _json_mapping(
            self.log_abs_det_jacobian,
            location="transform.log_abs_det_jacobian",
        )
        if not jacobian:
            raise DualArtifactError("Transform Jacobian accounting must be explicit.")
        if self.placement not in {"embedded", "identity"}:
            raise DualArtifactError(
                "Transform placement must be 'embedded' or 'identity'."
            )
        object.__setattr__(self, "forward", tuple(dict(item) for item in forward))
        object.__setattr__(self, "inverse", tuple(dict(item) for item in inverse))
        object.__setattr__(self, "log_abs_det_jacobian", jacobian)

    @classmethod
    def identity(cls) -> TransformSpec:
        """Return an explicitly Jacobian-free identity transform."""

        return cls(
            forward=({"operation": "identity"},),
            inverse=({"operation": "identity"},),
            log_abs_det_jacobian={
                "convention": "log_abs_det_d_model_d_physical",
                "value": 0.0,
                "included_in_log_prob": True,
            },
            placement="identity",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "forward": list(self.forward),
            "inverse": list(self.inverse),
            "log_abs_det_jacobian": dict(self.log_abs_det_jacobian),
            "placement": self.placement,
        }

    @classmethod
    def from_value(
        cls,
        value: TransformSpec | Mapping[str, Any],
    ) -> TransformSpec:
        if isinstance(value, cls):
            return value
        forward = value.get("forward", ())
        inverse = value.get("inverse", ())
        if isinstance(forward, Mapping):
            forward = (forward,)
        if isinstance(inverse, Mapping):
            inverse = (inverse,)
        return cls(
            forward=tuple(forward),
            inverse=tuple(inverse),
            log_abs_det_jacobian=dict(value["log_abs_det_jacobian"]),
            placement=str(value.get("placement", "embedded")),
        )


@dataclass(frozen=True)
class OnnxGraphSpec:
    """One checksummed ONNX graph and its semantic tensor names."""

    path: str
    role: str
    inputs: dict[str, str]
    outputs: dict[str, str]
    sha256: str
    size_bytes: int
    member: int = 0
    opset: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = _safe_relative_path(self.path)
        role = str(self.role)
        if role not in _GRAPH_ROLES:
            raise DualArtifactError(
                f"Unknown ONNX graph role {role!r}; expected {sorted(_GRAPH_ROLES)}."
            )
        inputs = _string_mapping(self.inputs, location=f"{role}.inputs")
        outputs = _string_mapping(self.outputs, location=f"{role}.outputs")
        digest = str(self.sha256)
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise DualArtifactError(
                "sha256 must be a lowercase hexadecimal SHA-256 digest."
            )
        size = int(self.size_bytes)
        if size < 0:
            raise DualArtifactError("ONNX graph size cannot be negative.")
        member = int(self.member)
        if member < 0:
            raise DualArtifactError("ONNX ensemble member cannot be negative.")
        opset = None if self.opset is None else int(self.opset)
        if opset is not None and opset < 1:
            raise DualArtifactError("ONNX opset must be positive when recorded.")
        metadata = _json_mapping(self.metadata, location=f"{role}.metadata")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "size_bytes", size)
        object.__setattr__(self, "member", member)
        object.__setattr__(self, "opset", opset)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        root: str | Path,
        role: str,
        inputs: Mapping[str, str],
        outputs: Mapping[str, str],
        member: int = 0,
        opset: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> OnnxGraphSpec:
        """Describe an already-exported ONNX file below *root*."""

        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        return cls(
            path=_relative_to(file_path, Path(root)),
            role=role,
            inputs=dict(inputs),
            outputs=dict(outputs),
            sha256=_sha256_file(file_path),
            size_bytes=file_path.stat().st_size,
            member=member,
            opset=opset,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "member": self.member,
            "opset": self.opset,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OnnxGraphSpec:
        return cls(
            path=str(value["path"]),
            role=str(value["role"]),
            inputs=dict(value["inputs"]),
            outputs=dict(value["outputs"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            member=int(value.get("member", 0)),
            opset=value.get("opset"),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class EnsembleSpec:
    """How independently exported graph members are combined."""

    reduction: str
    members: int
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        reduction = _REDUCTION_ALIASES.get(self.reduction, self.reduction)
        if reduction not in _ENSEMBLE_REDUCTIONS:
            raise DualArtifactError(
                f"Unknown ensemble reduction {self.reduction!r}; "
                f"expected {sorted(_ENSEMBLE_REDUCTIONS)}."
            )
        members = int(self.members)
        if members < 1:
            raise DualArtifactError("An ensemble must contain at least one member.")
        weights = self.weights
        if weights is not None:
            weights = tuple(float(weight) for weight in weights)
            if len(weights) != members:
                raise DualArtifactError(
                    "Ensemble weights must contain one value per member."
                )
            if not np.isfinite(weights).all() or any(
                weight < 0.0 for weight in weights
            ):
                raise DualArtifactError(
                    "Ensemble weights must be finite and non-negative."
                )
            if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1.0e-12):
                raise DualArtifactError("Ensemble weights must sum to one.")
        if reduction == "single":
            if members != 1:
                raise DualArtifactError(
                    "The 'single' ensemble reduction requires exactly one member."
                )
            if weights not in {None, (1.0,)}:
                raise DualArtifactError(
                    "A single-member ensemble can only have unit weight."
                )
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "weights", weights)

    @property
    def normalized_weights(self) -> np.ndarray:
        if self.weights is None:
            return np.full(self.members, 1.0 / self.members, dtype=np.float64)
        return np.asarray(self.weights, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reduction": self.reduction,
            "members": self.members,
            "weights": None if self.weights is None else list(self.weights),
        }

    @classmethod
    def from_value(
        cls,
        value: EnsembleSpec | Mapping[str, Any] | str,
        *,
        members: int,
    ) -> EnsembleSpec:
        if isinstance(value, cls):
            if value.members != members:
                raise DualArtifactError(
                    "Ensemble metadata does not match the number of graphs."
                )
            return value
        if isinstance(value, str):
            return cls(reduction=value, members=members)
        return cls(
            reduction=str(value["reduction"]),
            members=int(value.get("members", members)),
            weights=(None if value.get("weights") is None else tuple(value["weights"])),
        )


@dataclass(frozen=True)
class DualArtifactSpec:
    """One of ``q_phi``, ``r_p``, ``q_eta``, ``r_c``, or ``z_c``."""

    name: str
    graphs: tuple[OnnxGraphSpec, ...]
    transforms: dict[str, TransformSpec]
    ensemble: EnsembleSpec
    base_distribution: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name)
        if name not in DUAL_ARTIFACT_NAMES:
            raise DualArtifactError(
                f"Unknown dual artifact {name!r}; expected {DUAL_ARTIFACT_NAMES}."
            )
        graphs = tuple(self.graphs)
        if not graphs:
            raise DualArtifactError(f"Artifact {name!r} has no ONNX graphs.")
        transforms = {
            str(key): TransformSpec.from_value(value)
            for key, value in self.transforms.items()
        }
        expected_transforms = {"theta"} if name == "z_c" else {"x", "theta"}
        if set(transforms) != expected_transforms:
            raise DualArtifactError(
                f"Artifact {name!r} transforms must be exactly "
                f"{sorted(expected_transforms)}."
            )
        metadata = _json_mapping(
            self.metadata,
            location=f"artifacts.{name}.metadata",
        )
        self._validate_graph_contract(name, graphs)
        role_members = [graph for graph in graphs if graph.role == "log_ratio"]
        ensemble_members = len(role_members) if role_members else 1
        ensemble = EnsembleSpec.from_value(
            self.ensemble,
            members=ensemble_members,
        )
        if name not in {"r_p", "r_c"} and ensemble.reduction != "single":
            raise DualArtifactError(
                f"Artifact {name!r} must use the 'single' ensemble rule."
            )
        base_distribution = self.base_distribution
        if name in {"q_phi", "q_eta"}:
            base_distribution = str(base_distribution or "standard_normal")
            if base_distribution != "standard_normal":
                raise DualArtifactError(
                    "Lazy conditional-flow sampling currently requires a "
                    "'standard_normal' base distribution."
                )
        elif base_distribution is not None:
            raise DualArtifactError(
                f"Artifact {name!r} cannot declare a flow base distribution."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "graphs", graphs)
        object.__setattr__(self, "transforms", transforms)
        object.__setattr__(self, "ensemble", ensemble)
        object.__setattr__(self, "base_distribution", base_distribution)
        object.__setattr__(self, "metadata", metadata)

    @staticmethod
    def _validate_graph_contract(
        name: str,
        graphs: tuple[OnnxGraphSpec, ...],
    ) -> None:
        expected: dict[str, tuple[set[str], set[str]]]
        if name == "q_phi":
            expected = {
                "log_prob": ({"theta", "x"}, {"log_prob"}),
                "inverse": ({"base_noise", "x"}, {"theta"}),
            }
        elif name == "q_eta":
            expected = {
                "log_prob": ({"x", "theta"}, {"log_prob"}),
                "inverse": ({"base_noise", "theta"}, {"x"}),
            }
        elif name in {"r_p", "r_c"}:
            expected = {"log_ratio": ({"x", "theta"}, {"log_ratio"})}
        else:
            expected = {"log_normalization": ({"theta"}, {"log_normalization"})}
        roles = {graph.role for graph in graphs}
        if roles != set(expected):
            raise DualArtifactError(
                f"Artifact {name!r} graph roles must be exactly {sorted(expected)}."
            )
        for role, (inputs, outputs) in expected.items():
            matches = [graph for graph in graphs if graph.role == role]
            if role == "log_ratio":
                members = sorted(graph.member for graph in matches)
                if members != list(range(len(matches))):
                    raise DualArtifactError(
                        f"Artifact {name!r} ratio member indices must be "
                        "contiguous from zero."
                    )
            elif len(matches) != 1 or matches[0].member != 0:
                raise DualArtifactError(
                    f"Artifact {name!r} must contain one member-zero {role!r} graph."
                )
            for graph in matches:
                if set(graph.inputs) != inputs or set(graph.outputs) != outputs:
                    raise DualArtifactError(
                        f"Artifact {name!r} role {role!r} requires semantic "
                        f"inputs {sorted(inputs)} and outputs {sorted(outputs)}."
                    )

    @property
    def kind(self) -> str:
        if self.name in {"q_phi", "q_eta"}:
            return "conditional_density"
        if self.name in {"r_p", "r_c"}:
            return "log_ratio"
        return "log_normalization"

    def graph(self, role: str) -> OnnxGraphSpec:
        """Return the unique graph for a non-ensemble role."""

        matches = [graph for graph in self.graphs if graph.role == role]
        if len(matches) != 1:
            raise DualArtifactError(
                f"Artifact {self.name!r} does not have one graph for role {role!r}."
            )
        return matches[0]

    def graphs_for(self, role: str) -> tuple[OnnxGraphSpec, ...]:
        return tuple(
            sorted(
                (graph for graph in self.graphs if graph.role == role),
                key=lambda graph: graph.member,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "graphs": [graph.to_dict() for graph in self.graphs],
            "transforms": {
                key: value.to_dict() for key, value in sorted(self.transforms.items())
            },
            "ensemble": self.ensemble.to_dict(),
            "base_distribution": self.base_distribution,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        name: str,
        value: Mapping[str, Any],
    ) -> DualArtifactSpec:
        graphs = tuple(
            OnnxGraphSpec.from_dict(graph) for graph in value.get("graphs", ())
        )
        role_members = sum(graph.role == "log_ratio" for graph in graphs) or 1
        result = cls(
            name=name,
            graphs=graphs,
            transforms={
                key: TransformSpec.from_value(transform)
                for key, transform in value.get("transforms", {}).items()
            },
            ensemble=EnsembleSpec.from_value(
                value.get(
                    "ensemble",
                    {"reduction": "single", "members": role_members},
                ),
                members=role_members,
            ),
            base_distribution=value.get("base_distribution"),
            metadata=dict(value.get("metadata", {})),
        )
        declared_kind = value.get("kind")
        if declared_kind is not None and declared_kind != result.kind:
            raise DualArtifactError(
                f"Artifact {name!r} declares kind {declared_kind!r}, "
                f"expected {result.kind!r}."
            )
        return result


@dataclass(frozen=True)
class PosteriorRatioProvenance:
    """The exact denominator used to train the posterior correction."""

    reference: str = "flow"
    defensive_epsilon: float = 0.0
    proposal: str = "rho"
    numerator: str = "p_rho(theta|x)"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reference = str(self.reference)
        epsilon = float(self.defensive_epsilon)
        if reference not in {"flow", "defensive"}:
            raise DualArtifactError(
                "Posterior ratio reference must be 'flow' or 'defensive'."
            )
        if reference == "defensive":
            if not 0.0 < epsilon < 1.0:
                raise DualArtifactError(
                    "A defensive posterior ratio requires 0 < epsilon < 1."
                )
        elif epsilon != 0.0:
            raise DualArtifactError(
                "defensive_epsilon must be zero for a flow-reference ratio."
            )
        if not self.proposal or not self.numerator:
            raise DualArtifactError(
                "Posterior ratio proposal and numerator provenance are required."
            )
        metadata = _json_mapping(
            self.metadata,
            location="posterior_ratio.metadata",
        )
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "defensive_epsilon", epsilon)
        object.__setattr__(self, "metadata", metadata)

    @property
    def denominator(self) -> str:
        if self.reference == "flow":
            return "q_phi(theta|x)"
        return (
            f"(1-{self.defensive_epsilon:.17g})*q_phi(theta|x)"
            f"+{self.defensive_epsilon:.17g}*rho(theta)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "defensive_epsilon": self.defensive_epsilon,
            "proposal": self.proposal,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_value(
        cls,
        value: PosteriorRatioProvenance | Mapping[str, Any],
    ) -> PosteriorRatioProvenance:
        if isinstance(value, cls):
            return value
        result = cls(
            reference=str(value.get("reference", "flow")),
            defensive_epsilon=float(value.get("defensive_epsilon", 0.0)),
            proposal=str(value.get("proposal", "rho")),
            numerator=str(value.get("numerator", "p_rho(theta|x)")),
            metadata=dict(value.get("metadata", {})),
        )
        declared = value.get("denominator")
        if declared is not None and declared != result.denominator:
            raise DualArtifactError(
                "Posterior ratio denominator disagrees with reference/epsilon."
            )
        return result


@dataclass
class DualArtifactManifest:
    """One portable manifest for the five frozen learned objects."""

    x_signature: FeatureSignature
    theta_signature: FeatureSignature
    artifacts: dict[str, DualArtifactSpec]
    posterior_ratio: PosteriorRatioProvenance
    source_provenance: dict[str, Any]
    config_provenance: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = DUAL_MANIFEST_SCHEMA_VERSION
    artifact_type: str = DUAL_MANIFEST_ARTIFACT_TYPE
    package_version: str = __version__
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    _manifest_path: Path | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.schema_version != DUAL_MANIFEST_SCHEMA_VERSION:
            raise DualArtifactError(
                f"Unsupported dual manifest schema {self.schema_version!r}; "
                f"expected {DUAL_MANIFEST_SCHEMA_VERSION!r}."
            )
        if self.artifact_type != DUAL_MANIFEST_ARTIFACT_TYPE:
            raise DualArtifactError(f"Unexpected artifact type {self.artifact_type!r}.")
        self.x_signature = FeatureSignature.from_value(self.x_signature)
        self.theta_signature = FeatureSignature.from_value(self.theta_signature)
        artifacts = {
            str(name): (
                value
                if isinstance(value, DualArtifactSpec)
                else DualArtifactSpec.from_dict(str(name), value)
            )
            for name, value in self.artifacts.items()
        }
        if set(artifacts) != set(DUAL_ARTIFACT_NAMES):
            missing = sorted(set(DUAL_ARTIFACT_NAMES).difference(artifacts))
            extra = sorted(set(artifacts).difference(DUAL_ARTIFACT_NAMES))
            raise DualArtifactError(
                "A dual manifest must contain exactly q_phi, r_p, q_eta, r_c, "
                f"and z_c; missing={missing}, extra={extra}."
            )
        for name, artifact in artifacts.items():
            if artifact.name != name:
                raise DualArtifactError(
                    f"Artifact mapping key {name!r} disagrees with its name."
                )
        self.artifacts = artifacts
        self.posterior_ratio = PosteriorRatioProvenance.from_value(self.posterior_ratio)
        self.source_provenance = _json_mapping(
            self.source_provenance,
            location="source_provenance",
        )
        self.config_provenance = _json_mapping(
            self.config_provenance,
            location="config_provenance",
        )
        if not self.source_provenance or not self.config_provenance:
            raise DualArtifactError(
                "Source and configuration provenance must both be non-empty."
            )
        self.metadata = _json_mapping(self.metadata, location="metadata")

    @property
    def manifest_path(self) -> Path | None:
        """Path from which this manifest was loaded or to which it was written."""

        return self._manifest_path

    @property
    def root(self) -> Path:
        """Bundle root for relative graph paths."""

        if self._manifest_path is None:
            raise DualArtifactError(
                "The manifest has not been written; pass an explicit root."
            )
        return self._manifest_path.parent

    @property
    def graphs(self) -> tuple[OnnxGraphSpec, ...]:
        return tuple(
            graph
            for name in DUAL_ARTIFACT_NAMES
            for graph in self.artifacts[name].graphs
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON representation."""

        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "package_version": self.package_version,
            "created_at": self.created_at,
            "signatures": {
                "x": self.x_signature.to_dict(),
                "theta": self.theta_signature.to_dict(),
            },
            "posterior_ratio": self.posterior_ratio.to_dict(),
            "artifacts": {
                name: self.artifacts[name].to_dict() for name in DUAL_ARTIFACT_NAMES
            },
            "source_provenance": dict(self.source_provenance),
            "config_provenance": dict(self.config_provenance),
            "metadata": dict(self.metadata),
        }

    def verification_errors(
        self,
        root: str | Path | None = None,
    ) -> list[str]:
        """Return all missing, escaping, size, and checksum errors."""

        root_path = Path(root).resolve() if root is not None else self.root.resolve()
        errors: list[str] = []
        checked: set[tuple[str, str, int]] = set()
        for graph in self.graphs:
            identity = (graph.path, graph.sha256, graph.size_bytes)
            if identity in checked:
                continue
            checked.add(identity)
            candidate = root_path.joinpath(*PurePosixPath(graph.path).parts)
            try:
                relative = candidate.resolve().relative_to(root_path)
            except (OSError, ValueError):
                errors.append(f"path escapes bundle root: {graph.path}")
                continue
            path = root_path / relative
            if not path.is_file():
                errors.append(f"missing: {graph.path}")
                continue
            size = path.stat().st_size
            if size != graph.size_bytes:
                errors.append(
                    f"size mismatch: {graph.path} "
                    f"(expected {graph.size_bytes}, found {size})"
                )
                continue
            digest = _sha256_file(path)
            if digest != graph.sha256:
                errors.append(
                    f"checksum mismatch: {graph.path} "
                    f"(expected {graph.sha256}, found {digest})"
                )
        return errors

    def verify(self, root: str | Path | None = None) -> None:
        """Raise unless every recorded ONNX file is intact."""

        errors = self.verification_errors(root)
        if errors:
            details = "\n".join(f"- {error}" for error in errors)
            raise DualArtifactIntegrityError(
                f"Dual artifact bundle failed verification:\n{details}"
            )

    def write(self, path: str | Path, *, verify: bool = True) -> Path:
        """Atomically write the single bundle manifest."""

        manifest_path = Path(path)
        if any(graph.path == manifest_path.name for graph in self.graphs):
            raise DualArtifactError(
                "The manifest path cannot overwrite a recorded ONNX graph."
            )
        if verify:
            self.verify(manifest_path.parent)
        _atomic_json_write(manifest_path, self.to_dict())
        self._manifest_path = manifest_path.resolve()
        return manifest_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verify: bool = False,
    ) -> DualArtifactManifest:
        """Load a manifest, optionally verifying every graph immediately."""

        manifest_path = Path(path)
        with manifest_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        signatures = payload.get("signatures", {})
        manifest = cls(
            x_signature=FeatureSignature.from_value(signatures["x"]),
            theta_signature=FeatureSignature.from_value(signatures["theta"]),
            artifacts={
                name: DualArtifactSpec.from_dict(name, artifact)
                for name, artifact in payload.get("artifacts", {}).items()
            },
            posterior_ratio=PosteriorRatioProvenance.from_value(
                payload["posterior_ratio"]
            ),
            source_provenance=dict(payload.get("source_provenance", {})),
            config_provenance=dict(payload.get("config_provenance", {})),
            metadata=dict(payload.get("metadata", {})),
            schema_version=str(payload.get("schema_version", "")),
            artifact_type=str(payload.get("artifact_type", "")),
            package_version=str(payload.get("package_version", "")),
            created_at=str(payload.get("created_at", "")),
        )
        manifest._manifest_path = manifest_path.resolve()
        if verify:
            manifest.verify()
        return manifest

    def to_dual_model(
        self,
        *,
        rho: Any,
        root: str | Path | None = None,
        verify: bool = True,
        providers: Sequence[str] | None = None,
        session_factory: Any | None = None,
    ) -> DualModel:
        """Build a :class:`DualModel` backed by lazy ONNX adapters.

        Constructing the model never imports ONNX Runtime. Each graph session
        is created only when its first method is evaluated.
        """

        root_path = Path(root).resolve() if root is not None else self.root.resolve()
        if verify:
            self.verify(root_path)
        for artifact in self.artifacts.values():
            for transform in artifact.transforms.values():
                if transform.placement not in {"embedded", "identity"}:
                    raise DualArtifactError(
                        "Lazy ONNX adapters require transforms to be embedded "
                        "or explicitly identity."
                    )
        common = {
            "root": root_path,
            "signatures": {
                "x": self.x_signature,
                "theta": self.theta_signature,
            },
            "providers": providers,
            "session_factory": session_factory,
        }
        metadata = dict(self.metadata)
        metadata["artifact_manifest"] = (
            None if self.manifest_path is None else self.manifest_path.as_posix()
        )
        metadata["source_provenance"] = dict(self.source_provenance)
        metadata["config_provenance"] = dict(self.config_provenance)
        return DualModel(
            q_phi=LazyOnnxConditionalDensity(
                self.artifacts["q_phi"],
                target="theta",
                context="x",
                **common,
            ),
            r_p=LazyOnnxLogRatio(self.artifacts["r_p"], **common),
            q_eta=LazyOnnxConditionalDensity(
                self.artifacts["q_eta"],
                target="x",
                context="theta",
                **common,
            ),
            r_c=LazyOnnxLogRatio(self.artifacts["r_c"], **common),
            z_c=LazyOnnxLogNormalizer(self.artifacts["z_c"], **common),
            rho=rho,
            posterior_ratio_reference=self.posterior_ratio.reference,
            defensive_epsilon=self.posterior_ratio.defensive_epsilon,
            metadata=metadata,
        )


def _coerce_graph(
    value: OnnxGraphSpec | Mapping[str, Any],
    *,
    root: Path,
) -> OnnxGraphSpec:
    if isinstance(value, OnnxGraphSpec):
        return value
    descriptor = dict(value)
    path = descriptor.pop("path")
    if "sha256" in descriptor or "size_bytes" in descriptor:
        descriptor["path"] = path
        return OnnxGraphSpec.from_dict(descriptor)
    return OnnxGraphSpec.from_file(path, root=root, **descriptor)


def _coerce_artifact(
    name: str,
    value: DualArtifactSpec | Mapping[str, Any],
    *,
    root: Path,
) -> DualArtifactSpec:
    if isinstance(value, DualArtifactSpec):
        if value.name != name:
            raise DualArtifactError(
                f"Artifact key {name!r} disagrees with spec {value.name!r}."
            )
        return value
    descriptor = dict(value)
    declared_kind = descriptor.pop("kind", None)
    graphs = tuple(
        _coerce_graph(graph, root=root) for graph in descriptor.pop("graphs")
    )
    ratio_members = sum(graph.role == "log_ratio" for graph in graphs) or 1
    ensemble_value = descriptor.pop(
        "ensemble",
        {"reduction": "single", "members": ratio_members},
    )
    transforms = descriptor.pop("transforms")
    base_distribution = descriptor.pop("base_distribution", None)
    metadata = descriptor.pop("metadata", {})
    if descriptor:
        raise DualArtifactError(
            f"Unknown fields for artifact {name!r}: {sorted(descriptor)}."
        )
    result = DualArtifactSpec(
        name=name,
        graphs=graphs,
        transforms={
            key: TransformSpec.from_value(transform)
            for key, transform in transforms.items()
        },
        ensemble=EnsembleSpec.from_value(
            ensemble_value,
            members=ratio_members,
        ),
        base_distribution=base_distribution,
        metadata=metadata,
    )
    if declared_kind is not None and declared_kind != result.kind:
        raise DualArtifactError(
            f"Artifact {name!r} declares kind {declared_kind!r}, "
            f"expected {result.kind!r}."
        )
    return result


def create_dual_artifact_manifest(
    path: str | Path,
    *,
    x_signature: FeatureSignature | Mapping[str, Any] | Sequence[str],
    theta_signature: FeatureSignature | Mapping[str, Any] | Sequence[str],
    artifacts: Mapping[str, DualArtifactSpec | Mapping[str, Any]],
    source_provenance: Mapping[str, Any],
    config_provenance: Mapping[str, Any],
    posterior_ratio_reference: str = "flow",
    defensive_epsilon: float = 0.0,
    posterior_ratio_provenance: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DualArtifactManifest:
    """Checksum already-exported ONNX files and write one dual manifest.

    Every graph descriptor must contain ``path``, ``role``, semantic-to-ONNX
    ``inputs`` and ``outputs``, and may contain ``member``, ``opset``, and
    metadata. Paths must reside below the manifest directory.
    """

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ratio_metadata = dict(posterior_ratio_provenance or {})
    if (
        "reference" in ratio_metadata
        and ratio_metadata["reference"] != posterior_ratio_reference
    ):
        raise DualArtifactError(
            "posterior_ratio_provenance.reference disagrees with "
            "posterior_ratio_reference."
        )
    if "defensive_epsilon" in ratio_metadata and float(
        ratio_metadata["defensive_epsilon"]
    ) != float(defensive_epsilon):
        raise DualArtifactError(
            "posterior_ratio_provenance.defensive_epsilon disagrees with "
            "defensive_epsilon."
        )
    ratio_metadata.setdefault("reference", posterior_ratio_reference)
    ratio_metadata.setdefault("defensive_epsilon", defensive_epsilon)
    manifest = DualArtifactManifest(
        x_signature=FeatureSignature.from_value(x_signature),
        theta_signature=FeatureSignature.from_value(theta_signature),
        artifacts={
            name: _coerce_artifact(name, value, root=manifest_path.parent)
            for name, value in artifacts.items()
        },
        posterior_ratio=PosteriorRatioProvenance.from_value(ratio_metadata),
        source_provenance=dict(source_provenance),
        config_provenance=dict(config_provenance),
        metadata=dict(metadata or {}),
    )
    manifest.write(manifest_path)
    return manifest


def write_dual_artifact_manifest(
    path: str | Path,
    **kwargs: Any,
) -> Path:
    """Path-returning convenience wrapper around manifest creation."""

    return create_dual_artifact_manifest(path, **kwargs).manifest_path or Path(path)


def load_dual_artifact_manifest(
    path: str | Path,
    *,
    verify: bool = False,
) -> DualArtifactManifest:
    """Load a dual manifest and optionally verify all ONNX graphs."""

    return DualArtifactManifest.load(path, verify=verify)


def verify_dual_artifact_manifest(path: str | Path) -> DualArtifactManifest:
    """Load and verify a dual manifest, returning the validated object."""

    return DualArtifactManifest.load(path, verify=True)


class _LazyOnnxGraph:
    def __init__(
        self,
        graph: OnnxGraphSpec,
        *,
        root: Path,
        providers: Sequence[str] | None,
        session_factory: Any | None,
    ) -> None:
        self.graph = graph
        self.path = root.joinpath(*PurePosixPath(graph.path).parts)
        self.providers = None if providers is None else tuple(providers)
        self.session_factory = session_factory
        self._session: Any | None = None

    @property
    def session(self) -> Any:
        if self._session is None:
            if self.session_factory is not None:
                self._session = self.session_factory(
                    str(self.path),
                    providers=self.providers,
                )
            else:
                try:
                    runtime = importlib.import_module("onnxruntime")
                except ImportError as exc:
                    raise OnnxRuntimeUnavailable(
                        "ONNX Runtime is required for dual artifact inference. "
                        "Install `hnsbi-toolkit[bayes]`."
                    ) from exc
                options: dict[str, Any] = {}
                if self.providers is not None:
                    options["providers"] = list(self.providers)
                self._session = runtime.InferenceSession(str(self.path), **options)
            expected_inputs = set(self.graph.inputs.values())
            expected_outputs = set(self.graph.outputs.values())
            actual_inputs = {value.name for value in self._session.get_inputs()}
            actual_outputs = {value.name for value in self._session.get_outputs()}
            if expected_inputs != actual_inputs:
                raise DualArtifactError(
                    f"ONNX input names for {self.graph.path!r} disagree with "
                    f"the manifest: expected={sorted(expected_inputs)}, "
                    f"actual={sorted(actual_inputs)}."
                )
            if expected_outputs != actual_outputs:
                raise DualArtifactError(
                    f"ONNX output names for {self.graph.path!r} disagree with "
                    f"the manifest: expected={sorted(expected_outputs)}, "
                    f"actual={sorted(actual_outputs)}."
                )
        return self._session

    def run(
        self,
        semantic_inputs: Mapping[str, np.ndarray],
        *,
        dtypes: Mapping[str, str],
    ) -> dict[str, np.ndarray]:
        missing = set(self.graph.inputs).difference(semantic_inputs)
        extra = set(semantic_inputs).difference(self.graph.inputs)
        if missing or extra:
            raise DualArtifactError(
                f"Semantic ONNX input mismatch for {self.graph.path!r}; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )
        feed = {
            self.graph.inputs[semantic]: np.asarray(
                value,
                dtype=np.dtype(dtypes[semantic]),
            )
            for semantic, value in semantic_inputs.items()
        }
        semantics = tuple(self.graph.outputs)
        names = [self.graph.outputs[semantic] for semantic in semantics]
        values = self.session.run(names, feed)
        return {
            semantic: np.asarray(value)
            for semantic, value in zip(semantics, values, strict=True)
        }


class _LazyOnnxAdapter:
    def __init__(
        self,
        artifact: DualArtifactSpec,
        *,
        root: Path,
        signatures: Mapping[str, FeatureSignature],
        providers: Sequence[str] | None,
        session_factory: Any | None,
    ) -> None:
        self.artifact = artifact
        self.root = Path(root)
        self.signatures = dict(signatures)
        self.providers = None if providers is None else tuple(providers)
        self.session_factory = session_factory

    def _runner(self, graph: OnnxGraphSpec) -> _LazyOnnxGraph:
        cache_name = f"_runner_{graph.role}_{graph.member}"
        runner = getattr(self, cache_name, None)
        if runner is None:
            runner = _LazyOnnxGraph(
                graph,
                root=self.root,
                providers=self.providers,
                session_factory=self.session_factory,
            )
            setattr(self, cache_name, runner)
        return runner

    def _dtypes(self, graph: OnnxGraphSpec) -> dict[str, str]:
        result: dict[str, str] = {}
        for semantic in graph.inputs:
            signature_name = (
                self._base_signature if semantic == "base_noise" else semantic
            )
            result[semantic] = self.signatures[signature_name].dtype
        return result


class LazyOnnxConditionalDensity(_LazyOnnxAdapter):
    """Lazy conditional flow adapter with deterministic inverse evaluation."""

    def __init__(
        self,
        artifact: DualArtifactSpec,
        *,
        target: str,
        context: str,
        root: Path,
        signatures: Mapping[str, FeatureSignature],
        providers: Sequence[str] | None = None,
        session_factory: Any | None = None,
    ) -> None:
        super().__init__(
            artifact,
            root=root,
            signatures=signatures,
            providers=providers,
            session_factory=session_factory,
        )
        self.target = target
        self.context = context
        self._base_signature = target
        self._log_prob_graph = artifact.graph("log_prob")
        self._inverse_graph = artifact.graph("inverse")

    def log_prob(
        self,
        values: np.ndarray,
        *,
        context: np.ndarray,
    ) -> np.ndarray:
        values, context = align_rows(values, context, self.target, self.context)
        outputs = self._runner(self._log_prob_graph).run(
            {self.target: values, self.context: context},
            dtypes=self._dtypes(self._log_prob_graph),
        )
        return as_vector(outputs["log_prob"], len(values), "ONNX log probability")

    def inverse(
        self,
        base_noise: np.ndarray,
        *,
        context: np.ndarray,
    ) -> np.ndarray:
        """Apply the frozen inverse graph to caller-provided base noise."""

        context_array = as_2d(context, self.context)
        noise = np.asarray(base_noise, dtype=np.float64)
        dimension = self.signatures[self.target].dimension
        if noise.ndim == 2:
            if noise.shape[1] != dimension:
                raise ValueError(
                    f"base_noise must have trailing dimension {dimension}."
                )
            if len(context_array) == 1:
                flat_context = np.repeat(context_array, len(noise), axis=0)
            elif len(noise) == len(context_array):
                flat_context = context_array
            else:
                raise ValueError(
                    "Two-dimensional base_noise must be row-aligned with context "
                    "or use a single context row."
                )
            leading_shape = noise.shape[:-1]
            flat_noise = noise
        elif noise.ndim == 3:
            if noise.shape[0] != len(context_array):
                raise ValueError(
                    "Three-dimensional base_noise must have one block per context."
                )
            if noise.shape[2] != dimension:
                raise ValueError(
                    f"base_noise must have trailing dimension {dimension}."
                )
            leading_shape = noise.shape[:-1]
            flat_noise = noise.reshape(-1, dimension)
            flat_context = np.repeat(context_array, noise.shape[1], axis=0)
        else:
            raise ValueError("base_noise must be two- or three-dimensional.")
        outputs = self._runner(self._inverse_graph).run(
            {"base_noise": flat_noise, self.context: flat_context},
            dtypes=self._dtypes(self._inverse_graph),
        )
        values = as_2d(outputs[self.target], self.target)
        expected_shape = (*leading_shape, dimension)
        if values.size != int(np.prod(expected_shape)):
            raise ValueError(
                f"Inverse graph returned {values.shape}, expected {expected_shape}."
            )
        result = values.reshape(expected_shape)
        if not np.isfinite(result).all():
            raise ValueError("Inverse graph returned non-finite samples.")
        return result

    def sample(
        self,
        n: int,
        *,
        context: np.ndarray,
        rng: np.random.Generator | None = None,
        base_noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample through the inverse graph, optionally with exact base noise."""

        n = int(n)
        if n < 1:
            raise ValueError("n must be positive.")
        context_array = as_2d(context, self.context)
        shape = (
            len(context_array),
            n,
            self.signatures[self.target].dimension,
        )
        if base_noise is None:
            rng = np.random.default_rng() if rng is None else rng
            noise = rng.standard_normal(shape)
        else:
            if rng is not None:
                raise ValueError("Pass either rng or base_noise, not both.")
            noise = np.asarray(base_noise, dtype=np.float64)
            if noise.ndim == 2 and len(context_array) == 1:
                noise = noise[None, :, :]
            if noise.shape != shape:
                raise ValueError(
                    f"base_noise must have shape {shape}, found {noise.shape}."
                )
        return self.inverse(noise, context=context_array)


class LazyOnnxLogRatio(_LazyOnnxAdapter):
    """Lazy member-wise ONNX log-ratio ensemble."""

    _base_signature = "theta"

    def log_ratio(
        self,
        target: np.ndarray,
        *,
        context: np.ndarray,
    ) -> np.ndarray:
        target_array, context_array = align_rows(
            target,
            context,
            "target",
            "context",
        )
        if self.artifact.name == "r_p":
            semantic_inputs = {"theta": target_array, "x": context_array}
        else:
            semantic_inputs = {"x": target_array, "theta": context_array}
        members = []
        for graph in self.artifact.graphs_for("log_ratio"):
            output = self._runner(graph).run(
                semantic_inputs,
                dtypes=self._dtypes(graph),
            )
            members.append(
                as_vector(output["log_ratio"], len(target_array), "ONNX log ratio")
            )
        stacked = np.stack(members, axis=0)
        weights = self.artifact.ensemble.normalized_weights[:, None]
        reduction = self.artifact.ensemble.reduction
        if reduction == "single":
            return stacked[0]
        if reduction == "mean_log_ratio":
            return np.sum(weights * stacked, axis=0)
        maximum = np.max(stacked, axis=0)
        return maximum + np.log(np.sum(weights * np.exp(stacked - maximum), axis=0))


class LazyOnnxLogNormalizer(_LazyOnnxAdapter):
    """Lazy ``theta -> log Z_C(theta)`` ONNX adapter."""

    _base_signature = "theta"

    def __init__(self, artifact: DualArtifactSpec, **kwargs: Any) -> None:
        super().__init__(artifact, **kwargs)
        self._graph = artifact.graph("log_normalization")

    def log_normalization(self, theta: np.ndarray) -> np.ndarray:
        theta_array = as_2d(theta, "theta")
        output = self._runner(self._graph).run(
            {"theta": theta_array},
            dtypes=self._dtypes(self._graph),
        )
        return as_vector(
            output["log_normalization"],
            len(theta_array),
            "ONNX log normalization",
        )

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        return self.log_normalization(theta)


# Short aliases are convenient in type-oriented code.
OnnxConditionalDensity = LazyOnnxConditionalDensity
OnnxLogRatio = LazyOnnxLogRatio
OnnxLogNormalizer = LazyOnnxLogNormalizer


def load_dual_model(
    path: str | Path,
    *,
    rho: Any,
    verify: bool = True,
    providers: Sequence[str] | None = None,
    session_factory: Any | None = None,
) -> DualModel:
    """Load a verified manifest and reconstruct its lazy :class:`DualModel`."""

    manifest = DualArtifactManifest.load(path, verify=verify)
    return manifest.to_dual_model(
        rho=rho,
        verify=False,
        providers=providers,
        session_factory=session_factory,
    )


__all__ = [
    "DUAL_ARTIFACT_NAMES",
    "DUAL_MANIFEST_ARTIFACT_TYPE",
    "DUAL_MANIFEST_SCHEMA_VERSION",
    "DualArtifactError",
    "DualArtifactIntegrityError",
    "DualArtifactManifest",
    "DualArtifactSpec",
    "EnsembleSpec",
    "FeatureSignature",
    "LazyOnnxConditionalDensity",
    "LazyOnnxLogNormalizer",
    "LazyOnnxLogRatio",
    "OnnxConditionalDensity",
    "OnnxGraphSpec",
    "OnnxLogNormalizer",
    "OnnxLogRatio",
    "OnnxRuntimeUnavailable",
    "PosteriorRatioProvenance",
    "TransformSpec",
    "create_dual_artifact_manifest",
    "load_dual_artifact_manifest",
    "load_dual_model",
    "verify_dual_artifact_manifest",
    "write_dual_artifact_manifest",
]
