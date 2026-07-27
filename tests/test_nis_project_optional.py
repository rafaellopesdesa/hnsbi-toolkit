from __future__ import annotations

import json

import numpy as np
import pytest

from hnsbi import Project
from hnsbi.artifacts import ArtifactManifest
from hnsbi.flows import FlowTrainer


class StandardNormal:
    def sample(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        generator = np.random.default_rng() if rng is None else rng
        return generator.normal(size=(int(n), 1))

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)[:, 0]
        return -0.5 * (x**2 + np.log(2.0 * np.pi))


def _config(tmp_path) -> dict:
    source = {"kind": "pyarrow", "registry_key": "unused"}
    return {
        "schema_version": "2.0",
        "features": ["x"],
        "output_dir": str(tmp_path / "artifacts"),
        "frequentist": {
            "reference": source,
            "samples": [
                {
                    "name": "background",
                    "source": source,
                    "nominal_yield": 20.0,
                    "multiplier": "1",
                },
                {
                    "name": "signal",
                    "source": source,
                    "nominal_yield": 10.0,
                    "multiplier": "mu",
                },
            ],
            "flow": {
                "architecture": "realnvp",
                "n_coupling_layers": 2,
                "hidden_features": 8,
                "hidden_layers": 1,
                "training": {
                    "epochs": 2,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "validation_fraction": 0.25,
                    "early_stopping_patience": 2,
                    "device": "cpu",
                },
            },
            "ratios": {
                "backend": "native",
                "ensemble_size": 1,
                "training": {
                    "epochs": 2,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "hidden_layers": 1,
                    "neurons": 8,
                    "validation_fraction": 0.2,
                    "holdout_fraction": 0.2,
                    "early_stopping_patience": 2,
                },
                "normalization": "independent_reference_mean",
                "diagnostics": {},
            },
            "parameters": [
                {
                    "name": "mu",
                    "role": "poi",
                    "nominal": 1.0,
                    "bounds": [0.0, 3.0],
                }
            ],
            "workspace": {
                "backend": "native",
                "measurement": "measurement",
                "channel": "SR",
                "output_path": "workspace.json",
            },
            "nis": {
                "design_points": [{"mu": 0.5}, {"mu": 2.0}],
                "epsilon": 0.2,
                "pilot_events": 96,
                "target_events": 32,
                "flow": {
                    "architecture": "realnvp",
                    "n_coupling_layers": 2,
                    "hidden_features": 8,
                    "hidden_layers": 1,
                    "training": {
                        "epochs": 2,
                        "batch_size": 32,
                        "learning_rate": 0.001,
                        "validation_fraction": 0.25,
                        "early_stopping_patience": 2,
                        "seed": 41,
                        "device": "cpu",
                    },
                    "onnx_opset": 17,
                },
                "output_path": str(tmp_path / "nis"),
            },
        },
    }


def test_project_nis_diagnostics_use_exact_training_holdout(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyarrow")

    project = Project.load(_config(tmp_path))
    artifacts = project.train_nis_asimov(
        reference=StandardNormal(),
        ratios={
            "background": lambda values: np.ones(len(values)),
            "signal": lambda values: np.exp(0.4 * values[:, 0] - 0.08),
        },
        truth_point={"mu": 1.0},
        asimov_point={"mu": 1.0},
        seed=41,
    )

    training_indices, validation_indices = FlowTrainer.random_split_indices(
        len(artifacts.design.pilot_values),
        0.25,
        rng=np.random.default_rng(41),
    )
    assert not np.intersect1d(training_indices, validation_indices).size
    np.testing.assert_allclose(
        artifacts.validation.reference_values,
        artifacts.design.pilot_values[validation_indices],
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    assert artifacts.validation_provenance == {
        "source": "internal_training_holdout",
        "seed": 41,
        "validation_fraction": 0.25,
        "training_rows": 72,
        "validation_rows": 24,
    }

    onnx_manifest = ArtifactManifest.load(artifacts.onnx_bundle.manifest_path)
    validation_manifest = ArtifactManifest.load(artifacts.validation_manifest)
    assert onnx_manifest.metadata["validation"] == artifacts.validation_provenance
    assert validation_manifest.metadata["validation"] == (
        artifacts.validation_provenance
    )
    payload = json.loads(artifacts.validation_report.read_text(encoding="utf-8"))
    assert payload["reference_count"] == 24
