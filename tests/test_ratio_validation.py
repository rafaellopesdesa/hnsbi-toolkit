from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from hnsbi import Project
from hnsbi.artifacts import ArtifactManifest
from hnsbi.data import EventBatch
from hnsbi.ratio_diagnostics import (
    diagnose_ratio_validation,
    plot_ratio_validation_calibration,
    plot_ratio_validation_reweighting,
)
from hnsbi.ratios import RatioBackendResult, RatioEnsemble


def test_independent_ensemble_validation_metrics_plots_and_manifest(tmp_path) -> None:
    rng = np.random.default_rng(2026)
    reference = np.column_stack(
        [rng.normal(0.0, 1.0, 4_000), rng.normal(0.0, 1.0, 4_000)]
    )
    target = np.column_stack([rng.normal(1.0, 1.0, 4_000), rng.normal(0.0, 1.0, 4_000)])

    def exact_ratio(values):
        return np.exp(np.asarray(values)[:, 0] - 0.5)

    ensemble = RatioEnsemble(
        [
            lambda values: 0.9 * exact_ratio(values),
            lambda values: 1.1 * exact_ratio(values),
        ]
    )
    report = diagnose_ratio_validation(
        ensemble=ensemble,
        target_values=target,
        reference_values=reference,
        features=("x", "y"),
        bins=16,
        output_directory=tmp_path,
    )

    assert report.target_count == 4_000
    assert report.reference_count == 4_000
    assert set(report.feature_metrics) == {"x", "y"}
    assert report.metrics["classification"]["weighted_auc"] > 0.7
    assert report.metrics["normalization"]["reference_mean_ratio"] == pytest.approx(
        1.0,
        abs=0.08,
    )
    assert (
        report.feature_metrics["x"]["target_reweighted_reference_weighted_ks"]
        < report.feature_metrics["x"]["target_reference_weighted_ks"]
    )
    assert set(report.curves["features"]["x"]) == {
        "edges",
        "reference",
        "reweighted_reference",
        "target",
    }

    report_path = tmp_path / "ratio_validation.json"
    manifest_path = tmp_path / "ratio_validation.manifest.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["validation"]["source"] == (
        "explicit-target-holdout-and-fresh-reference-flow"
    )
    manifest = ArtifactManifest.load(manifest_path)
    manifest.verify(tmp_path)
    assert manifest.artifact_type == "density-ratio-ensemble-validation"

    if importlib.util.find_spec("matplotlib") is None:
        assert report.figure_paths == ()
    else:
        assert {path.name for path in report.figure_paths} == {
            "calibration.png",
            "reweighted_x.png",
            "reweighted_y.png",
        }
        import matplotlib.pyplot as plt

        calibration = plot_ratio_validation_calibration(report)
        reweighting = plot_ratio_validation_reweighting(report, "x")
        plt.close(calibration)
        plt.close(reweighting)


def _project_config(tmp_path) -> dict:
    return {
        "schema_version": "2.0",
        "output_dir": str(tmp_path / "artifacts"),
        "features": ["x"],
        "frequentist": {
            "reference": {
                "kind": "pyarrow",
                "registry_key": "reference",
            },
            "samples": [
                {
                    "name": "signal",
                    "source": {
                        "kind": "pyarrow",
                        "registry_key": "signal",
                        "split_column": "partition",
                    },
                    "nominal_yield": 8.0,
                    "multiplier": "mu",
                },
                {
                    "name": "background",
                    "source": {
                        "kind": "pyarrow",
                        "registry_key": "background",
                    },
                    "nominal_yield": 20.0,
                    "multiplier": "1",
                },
            ],
            "flow": {
                "architecture": "realnvp",
                "n_coupling_layers": 2,
                "hidden_features": 16,
                "hidden_layers": 1,
                "scale_clip": 1.5,
                "training": {
                    "epochs": 2,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "validation_fraction": 0.2,
                    "early_stopping_patience": 1,
                    "device": "cpu",
                },
            },
            "ratios": {
                "backend": "native",
                "ensemble_size": 2,
                "training": {
                    "epochs": 2,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "hidden_layers": 1,
                    "neurons": 8,
                    "validation_fraction": 0.2,
                    "holdout_fraction": 0.2,
                    "early_stopping_patience": 1,
                },
                "normalization": "independent_reference_mean",
                "diagnostics": {},
            },
            "parameters": [
                {
                    "name": "mu",
                    "role": "poi",
                    "nominal": 1.0,
                    "bounds": [0.0, 5.0],
                }
            ],
            "workspace": {
                "backend": "native",
                "measurement": "measurement",
                "channel": "SR",
                "output_path": "artifacts/workspace.json",
            },
        },
    }


def test_project_validates_only_explicit_common_holdout_with_fresh_reference(
    tmp_path,
    monkeypatch,
) -> None:
    rng = np.random.default_rng(8)
    signal_partitions = np.asarray(
        ["train"] * 50 + ["validation"] * 20 + ["holdout"] * 30
    )
    signal = rng.normal(1.0, 1.0, (len(signal_partitions), 1)).astype(np.float32)
    background = rng.normal(-0.3, 1.0, (80, 1)).astype(np.float32)
    project = Project.load(
        _project_config(tmp_path),
        registry={
            "reference": np.zeros((10, 1), dtype=np.float32),
            "signal": signal,
            "background": background,
        },
    )

    class BatchSource:
        def __init__(self, batch):
            self.batch = batch

        def materialize(self, *, max_events=None):
            if max_events is not None:
                raise AssertionError("This test does not truncate the target samples.")
            return self.batch

    monkeypatch.setattr(
        project,
        "sample_sources",
        lambda: {
            "signal": BatchSource(
                EventBatch(
                    signal,
                    np.ones(len(signal)),
                    np.arange(len(signal)),
                    ("x",),
                    {"partition": signal_partitions},
                )
            ),
            "background": BatchSource(
                EventBatch(
                    background,
                    np.ones(len(background)),
                    np.arange(len(background)),
                    ("x",),
                )
            ),
        },
    )

    class StubBackend:
        name = "stub"

        def train_member(self, *, output_directory, **kwargs):
            artifact = output_directory / "ratio.bin"
            artifact.write_bytes(b"stub")
            return RatioBackendResult(
                evaluator=lambda values: np.exp(
                    np.asarray(values, dtype=np.float64)[:, 0] - 0.5
                ),
                files={"model": artifact},
            )

    class TrackingReference:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def sample(self, count, *, rng=None):
            self.calls.append(int(count))
            assert rng is not None
            return rng.normal(0.0, 1.0, size=(int(count), 1)).astype(np.float32)

    reference = TrackingReference()
    result = project.train_ratios(
        reference,
        backend=StubBackend(),
        normalization_events=200,
        seed=19,
    )

    assert set(result.diagnostics) == {"signal"}
    assert reference.calls == [100, 80, 200, 30]
    report = result.diagnostics["signal"]
    assert report.target_count == 30
    assert report.reference_count == 30
    assert report.metrics["validation"]["source"] == (
        "explicit-target-holdout-and-fresh-reference-flow"
    )
    validation_directory = (
        tmp_path / "artifacts" / "ratios" / "signal" / "ensemble_validation"
    )
    ArtifactManifest.load(
        validation_directory / "ratio_validation.manifest.json"
    ).verify(validation_directory)
    assert not (
        tmp_path / "artifacts" / "ratios" / "background" / "ensemble_validation"
    ).exists()
