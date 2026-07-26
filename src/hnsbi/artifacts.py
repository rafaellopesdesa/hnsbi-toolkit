"""Checksummed manifests for portable model artifacts.

The toolkit deliberately keeps its artifact format small: a JSON manifest
describes a logical bundle and records a SHA-256 digest for every file in that
bundle.  Model-specific modules own the contents of their files; this module
only provides integrity, provenance, and deterministic serialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ._version import __version__

MANIFEST_SCHEMA_VERSION = "1"


class ArtifactIntegrityError(RuntimeError):
    """Raised when a recorded artifact is missing or has changed."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the hexadecimal SHA-256 digest of *path*.

    Files are read incrementally so this is safe for large PyTorch checkpoints
    and ONNX graphs.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(path: str | Path) -> str:
    value = PurePosixPath(Path(path).as_posix())
    if value.is_absolute() or not value.parts:
        raise ValueError("Artifact paths must be non-empty and relative.")
    if any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError(
            "Artifact relative paths cannot contain '.', '..', or empty parts."
        )
    return value.as_posix()


def _relative_to(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{path} is outside artifact root {root}.") from exc
    return _safe_relative_path(relative)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ArtifactRecord:
    """One file recorded in an :class:`ArtifactManifest`."""

    path: str
    sha256: str
    size_bytes: int
    kind: str = "artifact"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _safe_relative_path(self.path))
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("sha256 must be a lowercase hexadecimal SHA-256 digest.")
        if self.size_bytes < 0:
            raise ValueError("size_bytes cannot be negative.")
        if not self.kind:
            raise ValueError("kind cannot be empty.")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        root: str | Path,
        kind: str = "artifact",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Create a record for a file below *root*."""

        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        return cls(
            path=_relative_to(file_path, Path(root)),
            sha256=sha256_file(file_path),
            size_bytes=file_path.stat().st_size,
            kind=kind,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactRecord:
        return cls(
            path=payload["path"],
            sha256=payload["sha256"],
            size_bytes=int(payload["size_bytes"]),
            kind=payload.get("kind", "artifact"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ArtifactManifest:
    """A versioned, checksummed description of a model artifact bundle."""

    artifact_type: str
    files: list[ArtifactRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = MANIFEST_SCHEMA_VERSION
    package_version: str = __version__
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if not self.artifact_type:
            raise ValueError("artifact_type cannot be empty.")
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported artifact manifest schema "
                f"{self.schema_version!r}; expected {MANIFEST_SCHEMA_VERSION!r}."
            )
        self._check_unique_paths()

    def _check_unique_paths(self) -> None:
        paths = [record.path for record in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("A manifest cannot contain duplicate artifact paths.")

    def add_file(
        self,
        path: str | Path,
        *,
        root: str | Path,
        kind: str = "artifact",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Checksum *path* and append it to this manifest."""

        record = ArtifactRecord.from_file(path, root=root, kind=kind, metadata=metadata)
        if any(existing.path == record.path for existing in self.files):
            raise ValueError(f"Artifact path {record.path!r} is already recorded.")
        self.files.append(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "artifact_type": self.artifact_type,
            "created_at": self.created_at,
            "files": [asdict(record) for record in self.files],
            "metadata": self.metadata,
            "package_version": self.package_version,
            "schema_version": self.schema_version,
        }

    def write(self, path: str | Path) -> Path:
        """Atomically write this manifest and return its path."""

        self._check_unique_paths()
        manifest_path = Path(path)
        _atomic_json_write(manifest_path, self.to_dict())
        return manifest_path

    @classmethod
    def load(cls, path: str | Path) -> ArtifactManifest:
        """Load a manifest without verifying its referenced files."""

        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        return cls(
            artifact_type=payload["artifact_type"],
            files=[
                ArtifactRecord.from_dict(record) for record in payload.get("files", [])
            ],
            metadata=dict(payload.get("metadata", {})),
            schema_version=payload.get("schema_version", ""),
            package_version=payload.get("package_version", ""),
            created_at=payload.get("created_at", ""),
        )

    def verification_errors(self, root: str | Path) -> list[str]:
        """Return all missing, size-mismatched, or checksum-mismatched files."""

        root_path = Path(root).resolve()
        errors: list[str] = []
        for record in self.files:
            path = root_path.joinpath(*PurePosixPath(record.path).parts)
            try:
                resolved = path.resolve().relative_to(root_path)
            except (OSError, ValueError):
                errors.append(f"path escapes artifact root: {record.path}")
                continue
            path = root_path / resolved
            if not path.is_file():
                errors.append(f"missing: {record.path}")
                continue
            size = path.stat().st_size
            if size != record.size_bytes:
                errors.append(
                    f"size mismatch: {record.path} "
                    f"(expected {record.size_bytes}, found {size})"
                )
                continue
            digest = sha256_file(path)
            if digest != record.sha256:
                errors.append(
                    f"checksum mismatch: {record.path} "
                    f"(expected {record.sha256}, found {digest})"
                )
        return errors

    def verify(self, root: str | Path) -> None:
        """Raise :class:`ArtifactIntegrityError` unless every file is intact."""

        errors = self.verification_errors(root)
        if errors:
            details = "\n".join(f"- {error}" for error in errors)
            raise ArtifactIntegrityError(
                f"Artifact bundle {self.artifact_type!r} failed verification:\n"
                f"{details}"
            )


def write_artifact_manifest(
    path: str | Path,
    *,
    artifact_type: str,
    files: dict[str, str | Path],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a manifest for named files sharing the manifest directory.

    ``files`` maps each file's semantic kind (for example ``"onnx-log-prob"``)
    to its path.
    """

    manifest_path = Path(path)
    manifest = ArtifactManifest(
        artifact_type=artifact_type, metadata=dict(metadata or {})
    )
    for kind, file_path in files.items():
        manifest.add_file(file_path, root=manifest_path.parent, kind=kind)
    return manifest.write(manifest_path)
