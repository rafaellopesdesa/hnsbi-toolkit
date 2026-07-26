from __future__ import annotations

import json

import pytest

from hnsbi.artifacts import (
    ArtifactIntegrityError,
    ArtifactManifest,
    ArtifactRecord,
    sha256_file,
    write_artifact_manifest,
)
from hnsbi.onnx import compare_outputs


def test_manifest_round_trip_and_verification(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"portable-model")
    manifest_path = write_artifact_manifest(
        tmp_path / "model.manifest.json",
        artifact_type="test-model",
        files={"onnx-model": model},
        metadata={"purpose": "unit-test"},
    )

    manifest = ArtifactManifest.load(manifest_path)

    assert manifest.artifact_type == "test-model"
    assert manifest.metadata["purpose"] == "unit-test"
    assert manifest.files[0].sha256 == sha256_file(model)
    manifest.verify(tmp_path)
    assert json.loads(manifest_path.read_text())["schema_version"] == "1"


def test_manifest_detects_same_size_tampering(tmp_path):
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"abcd")
    manifest = ArtifactManifest(artifact_type="weights")
    manifest.add_file(artifact, root=tmp_path)
    artifact.write_bytes(b"wxyz")

    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        manifest.verify(tmp_path)


def test_manifest_rejects_paths_outside_bundle(tmp_path):
    outside = tmp_path.parent / "outside-artifact.bin"
    outside.write_bytes(b"x")
    manifest = ArtifactManifest(artifact_type="test")

    with pytest.raises(ValueError, match="outside artifact root"):
        manifest.add_file(outside, root=tmp_path)
    with pytest.raises(ValueError, match="relative"):
        ArtifactRecord(
            path="../escape",
            sha256="0" * 64,
            size_bytes=1,
        )


def test_output_comparison_reports_parity_metrics():
    report = compare_outputs(
        [1.0, 2.0, 3.0],
        [1.0, 2.000001, 3.0],
        atol=1e-5,
        rtol=0,
    )

    assert report.passed
    assert report.max_absolute_error > 0
    assert report.shape == (3,)
    report.assert_close()


def test_output_comparison_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_outputs([1.0], [[1.0]])
