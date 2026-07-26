from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hnsbi.bayes import (
    DualArtifactError,
    DualArtifactIntegrityError,
    DualArtifactManifest,
    FeatureSignature,
    LazyOnnxConditionalDensity,
    LazyOnnxLogNormalizer,
    LazyOnnxLogRatio,
    PosteriorRatioProvenance,
    TransformSpec,
    create_dual_artifact_manifest,
    load_dual_model,
)


class StandardNormal:
    def sample(self, n, *, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.standard_normal((int(n), 2))

    def log_prob(self, values):
        values = np.asarray(values)
        return -0.5 * np.sum(values**2, axis=1)


def _graph(
    root: Path,
    filename: str,
    *,
    role: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    member: int = 0,
) -> dict:
    path = root / filename
    path.write_bytes(f"frozen graph: {filename}".encode())
    return {
        "path": path,
        "role": role,
        "inputs": inputs,
        "outputs": outputs,
        "member": member,
        "opset": 17,
        "metadata": {"exporter": "unit-test"},
    }


def _artifact_descriptors(root: Path) -> dict:
    identity = TransformSpec.identity()
    both = {"x": identity, "theta": identity}
    return {
        "q_phi": {
            "graphs": [
                _graph(
                    root,
                    "q_phi_log_prob.onnx",
                    role="log_prob",
                    inputs={"theta": "theta_in", "x": "x_context"},
                    outputs={"log_prob": "log_q_phi"},
                ),
                _graph(
                    root,
                    "q_phi_inverse.onnx",
                    role="inverse",
                    inputs={"base_noise": "base_noise", "x": "x_context"},
                    outputs={"theta": "theta_out"},
                ),
            ],
            "transforms": both,
            "base_distribution": "standard_normal",
        },
        "r_p": {
            "graphs": [
                _graph(
                    root,
                    "r_p_0.onnx",
                    role="log_ratio",
                    inputs={"theta": "theta_in", "x": "x_context"},
                    outputs={"log_ratio": "log_r_p"},
                    member=0,
                ),
                _graph(
                    root,
                    "r_p_1.onnx",
                    role="log_ratio",
                    inputs={"theta": "theta_in", "x": "x_context"},
                    outputs={"log_ratio": "log_r_p"},
                    member=1,
                ),
            ],
            "transforms": both,
            "ensemble": {
                "reduction": "arithmetic_mean_ratio",
                "members": 2,
                "weights": [0.25, 0.75],
            },
        },
        "q_eta": {
            "graphs": [
                _graph(
                    root,
                    "q_eta_log_prob.onnx",
                    role="log_prob",
                    inputs={"x": "x_in", "theta": "theta_context"},
                    outputs={"log_prob": "log_q_eta"},
                ),
                _graph(
                    root,
                    "q_eta_inverse.onnx",
                    role="inverse",
                    inputs={"base_noise": "base_noise", "theta": "theta_context"},
                    outputs={"x": "x_out"},
                ),
            ],
            "transforms": both,
        },
        "r_c": {
            "graphs": [
                _graph(
                    root,
                    "r_c.onnx",
                    role="log_ratio",
                    inputs={"x": "x_in", "theta": "theta_context"},
                    outputs={"log_ratio": "log_r_c"},
                )
            ],
            "transforms": both,
        },
        "z_c": {
            "graphs": [
                _graph(
                    root,
                    "z_c.onnx",
                    role="log_normalization",
                    inputs={"theta": "theta_in"},
                    outputs={"log_normalization": "log_z_c"},
                )
            ],
            "transforms": {"theta": identity},
        },
    }


def _create_bundle(root: Path) -> tuple[Path, DualArtifactManifest]:
    root.mkdir()
    path = root / "manifest.json"
    manifest = create_dual_artifact_manifest(
        path,
        x_signature=FeatureSignature(("energy", "angle"), dtype="float32"),
        theta_signature=("mu", "alpha"),
        artifacts=_artifact_descriptors(root),
        posterior_ratio_reference="defensive",
        defensive_epsilon=0.02,
        posterior_ratio_provenance={
            "proposal": "rho_nu_kappa",
            "numerator": "p_rho(theta|x)",
            "metadata": {"paired_group_split": True},
        },
        source_provenance={
            "repository": "https://example.invalid/hybrid_nsbi",
            "commit": "0123456789abcdef",
            "datasets": ["rho.parquet", "nu.parquet", "kappa.parquet"],
        },
        config_provenance={
            "path": "configs/dual.json",
            "sha256": "a" * 64,
        },
        metadata={"purpose": "portable dual test"},
    )
    return path, manifest


def test_manifest_round_trip_records_complete_scientific_contract(tmp_path):
    path, manifest = _create_bundle(tmp_path / "bundle")
    manifest.verify()
    loaded = DualArtifactManifest.load(path, verify=True)
    assert loaded.to_dict() == manifest.to_dict()

    payload = json.loads(path.read_text())
    assert set(payload["artifacts"]) == {"q_phi", "r_p", "q_eta", "r_c", "z_c"}
    assert payload["signatures"]["x"]["names"] == ["energy", "angle"]
    assert payload["signatures"]["theta"]["names"] == ["mu", "alpha"]
    assert payload["artifacts"]["q_phi"]["graphs"][0]["inputs"] == {
        "theta": "theta_in",
        "x": "x_context",
    }
    assert (
        payload["artifacts"]["q_phi"]["transforms"]["theta"]["log_abs_det_jacobian"][
            "included_in_log_prob"
        ]
        is True
    )
    assert payload["artifacts"]["r_p"]["ensemble"] == {
        "members": 2,
        "reduction": "arithmetic_mean_ratio",
        "weights": [0.25, 0.75],
    }
    ratio = payload["posterior_ratio"]
    assert ratio["reference"] == "defensive"
    assert ratio["defensive_epsilon"] == pytest.approx(0.02)
    assert "q_phi(theta|x)" in ratio["denominator"]
    assert "rho(theta)" in ratio["denominator"]
    assert payload["source_provenance"]["commit"] == "0123456789abcdef"
    assert payload["config_provenance"]["sha256"] == "a" * 64

    for graph in loaded.graphs:
        graph_path = path.parent / graph.path
        assert graph.sha256 == hashlib.sha256(graph_path.read_bytes()).hexdigest()
        assert graph.size_bytes == graph_path.stat().st_size


def test_integrity_detects_same_size_tampering_and_missing_graph(tmp_path):
    path, manifest = _create_bundle(tmp_path / "bundle")
    graph_path = path.parent / manifest.artifacts["r_c"].graphs[0].path
    graph_path.write_bytes(b"x" * graph_path.stat().st_size)
    with pytest.raises(DualArtifactIntegrityError, match="checksum mismatch"):
        DualArtifactManifest.load(path, verify=True)

    graph_path.unlink()
    loaded = DualArtifactManifest.load(path)
    errors = loaded.verification_errors()
    assert errors == [f"missing: {loaded.artifacts['r_c'].graphs[0].path}"]


def test_manifest_rejects_incomplete_or_ambiguous_metadata(tmp_path):
    with pytest.raises(DualArtifactError, match="requires 0 < epsilon < 1"):
        PosteriorRatioProvenance(
            reference="defensive",
            defensive_epsilon=0.0,
        )
    with pytest.raises(DualArtifactError, match="Jacobian accounting"):
        TransformSpec(
            forward=(),
            inverse=(),
            log_abs_det_jacobian={},
        )

    root = tmp_path / "bundle"
    root.mkdir()
    descriptors = _artifact_descriptors(root)
    descriptors.pop("z_c")
    with pytest.raises(DualArtifactError, match="exactly q_phi"):
        create_dual_artifact_manifest(
            root / "manifest.json",
            x_signature=("x",),
            theta_signature=("theta",),
            artifacts=descriptors,
            source_provenance={"commit": "abc"},
            config_provenance={"sha256": "def"},
        )


def test_manifest_rejects_graph_outside_portable_root(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    descriptors = _artifact_descriptors(root)
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    descriptors["z_c"]["graphs"][0]["path"] = outside
    with pytest.raises(DualArtifactError, match="outside manifest root"):
        create_dual_artifact_manifest(
            root / "manifest.json",
            x_signature=("x1", "x2"),
            theta_signature=("t1", "t2"),
            artifacts=descriptors,
            source_provenance={"commit": "abc"},
            config_provenance={"sha256": "def"},
        )


class _Node:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    def __init__(self, path):
        self.stem = Path(path).stem
        names = {
            "q_phi_log_prob": (("theta_in", "x_context"), ("log_q_phi",)),
            "q_phi_inverse": (("base_noise", "x_context"), ("theta_out",)),
            "r_p_0": (("theta_in", "x_context"), ("log_r_p",)),
            "r_p_1": (("theta_in", "x_context"), ("log_r_p",)),
            "q_eta_log_prob": (("x_in", "theta_context"), ("log_q_eta",)),
            "q_eta_inverse": (("base_noise", "theta_context"), ("x_out",)),
            "r_c": (("x_in", "theta_context"), ("log_r_c",)),
            "z_c": (("theta_in",), ("log_z_c",)),
        }
        self.input_names, self.output_names = names[self.stem]

    def get_inputs(self):
        return [_Node(name) for name in self.input_names]

    def get_outputs(self):
        return [_Node(name) for name in self.output_names]

    def run(self, output_names, feed):
        if self.stem == "q_phi_log_prob":
            value = -np.sum((feed["theta_in"] - feed["x_context"]) ** 2, axis=1)
        elif self.stem == "q_phi_inverse":
            value = feed["base_noise"] + feed["x_context"]
        elif self.stem == "q_eta_log_prob":
            value = -np.sum((feed["x_in"] - feed["theta_context"]) ** 2, axis=1)
        elif self.stem == "q_eta_inverse":
            value = feed["base_noise"] + feed["theta_context"]
        elif self.stem == "r_p_0":
            value = np.full((len(feed["theta_in"]), 1), np.log(2.0))
        elif self.stem == "r_p_1":
            value = np.full((len(feed["theta_in"]), 1), np.log(4.0))
        elif self.stem == "r_c":
            value = np.zeros((len(feed["x_in"]), 1))
        else:
            value = np.sum(feed["theta_in"], axis=1, keepdims=True)
        return [value]


def test_lazy_adapters_reconstruct_dual_model_without_onnxruntime(tmp_path):
    path, _ = _create_bundle(tmp_path / "bundle")
    opened = []

    def session_factory(graph_path, *, providers=None):
        opened.append((Path(graph_path).name, providers))
        return _FakeSession(graph_path)

    model = load_dual_model(
        path,
        rho=StandardNormal(),
        session_factory=session_factory,
    )
    assert opened == []
    assert isinstance(model.q_phi, LazyOnnxConditionalDensity)
    assert isinstance(model.r_p, LazyOnnxLogRatio)
    assert isinstance(model.z_c, LazyOnnxLogNormalizer)
    assert model.posterior_ratio_reference == "defensive"
    assert model.defensive_epsilon == pytest.approx(0.02)

    theta = np.array([[1.0, 2.0], [2.0, 4.0]])
    observation = np.array([[0.5, 1.0]])
    assert np.allclose(model.log_q_phi(theta, observation), [-1.25, -11.25])
    assert np.allclose(model.log_r_p(theta, observation), np.log(3.5))
    assert np.allclose(
        model.log_q_eta(observation, theta),
        [-1.25, -11.25],
    )
    assert np.allclose(model.log_r_c(observation, theta), 0.0)
    assert np.allclose(model.log_z_c(theta), [3.0, 6.0])

    base_noise = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ]
    )
    contexts = np.array([[10.0, 20.0], [30.0, 40.0]])
    expected = base_noise + contexts[:, None, :]
    assert np.allclose(
        model.q_phi.inverse(base_noise, context=contexts),
        expected,
    )
    assert np.allclose(
        model.q_phi.sample(
            2,
            context=contexts,
            base_noise=base_noise,
        ),
        expected,
    )
    assert opened
