from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from hnsbi.artifacts import ArtifactManifest
from hnsbi.native_ratios import (
    NativeRatioBackend,
    OnnxNativeRatioMember,
    _configured_diagnostic_checks,
    _resolve_class_splits,
)
from hnsbi.ratio_diagnostics import diagnose_ratio
from hnsbi.ratios import RatioTrainer, RatioTrainingConfig


def _groups_by_partition(groups, split):
    return {
        "train": set(np.asarray(groups)[split.training]),
        "validation": set(np.asarray(groups)[split.validation]),
        "holdout": set(np.asarray(groups)[split.holdout]),
    }


def test_grouped_ratio_split_is_deterministic_and_leakage_safe() -> None:
    groups = np.repeat(np.arange(8), 2)
    first = _resolve_class_splits(
        numerator_length=len(groups),
        denominator_length=len(groups),
        validation_fraction=0.2,
        holdout_fraction=0.2,
        rng=np.random.default_rng(9),
        seed=9,
        numerator_groups=groups,
        denominator_groups=groups,
    )
    second = _resolve_class_splits(
        numerator_length=len(groups),
        denominator_length=len(groups),
        validation_fraction=0.2,
        holdout_fraction=0.2,
        rng=np.random.default_rng(9),
        seed=9,
        numerator_groups=groups,
        denominator_groups=groups,
    )

    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.training, right.training)
        np.testing.assert_array_equal(left.validation, right.validation)
        np.testing.assert_array_equal(left.holdout, right.holdout)
        partitions = _groups_by_partition(groups, left)
        assert partitions["train"].isdisjoint(partitions["validation"])
        assert partitions["train"].isdisjoint(partitions["holdout"])
        assert partitions["validation"].isdisjoint(partitions["holdout"])
    assert _groups_by_partition(groups, first[0]) == _groups_by_partition(
        groups,
        first[1],
    )


def test_grouped_ratio_split_handles_one_class_group_subset() -> None:
    numerator_groups = np.repeat(np.arange(8), 2)
    denominator_groups = np.repeat(np.arange(4), 2)
    numerator, denominator = _resolve_class_splits(
        numerator_length=len(numerator_groups),
        denominator_length=len(denominator_groups),
        validation_fraction=0.2,
        holdout_fraction=0.2,
        rng=np.random.default_rng(19),
        seed=19,
        numerator_groups=numerator_groups,
        denominator_groups=denominator_groups,
    )

    numerator_partitions = _groups_by_partition(numerator_groups, numerator)
    denominator_partitions = _groups_by_partition(denominator_groups, denominator)
    assert all(denominator_partitions.values())
    for label in ("train", "validation", "holdout"):
        assert denominator_partitions[label].issubset(numerator_partitions[label])


def test_explicit_ratio_splits_are_honored_and_group_conflicts_rejected() -> None:
    labels = np.asarray(["train", "train", "train", "validation", "holdout", "holdout"])
    numerator, denominator = _resolve_class_splits(
        numerator_length=len(labels),
        denominator_length=len(labels),
        validation_fraction=0.2,
        holdout_fraction=0.2,
        rng=np.random.default_rng(4),
        seed=4,
        numerator_split=labels,
        denominator_split=labels,
    )
    np.testing.assert_array_equal(numerator.training, [0, 1, 2])
    np.testing.assert_array_equal(numerator.validation, [3])
    np.testing.assert_array_equal(numerator.holdout, [4, 5])
    np.testing.assert_array_equal(denominator.training, [0, 1, 2])

    conflicting = labels.copy()
    conflicting[0] = "holdout"
    with pytest.raises(ValueError, match="appears in both"):
        _resolve_class_splits(
            numerator_length=len(labels),
            denominator_length=len(labels),
            validation_fraction=0.2,
            holdout_fraction=0.2,
            rng=np.random.default_rng(4),
            seed=4,
            numerator_split=labels,
            denominator_split=conflicting,
            numerator_groups=np.arange(len(labels)),
            denominator_groups=np.arange(len(labels)),
        )


def test_native_backend_preserves_none_or_selected_diagnostic_checks() -> None:
    assert _configured_diagnostic_checks(RatioTrainingConfig()) is None
    config = RatioTrainingConfig(
        backend_options={"diagnostics": {"methods": ["overfit", "normalization"]}}
    )
    assert _configured_diagnostic_checks(config) == ("overfit", "normalization")


def test_ratio_diagnostics_cover_every_native_check(tmp_path) -> None:
    rng = np.random.default_rng(7)
    numerator_train = rng.normal(0.8, 1.0, size=(100, 2))
    denominator_train = rng.normal(0.0, 1.0, size=(100, 2))
    numerator_holdout = rng.normal(0.8, 1.0, size=(80, 2))
    denominator_holdout = rng.normal(0.0, 1.0, size=(80, 2))

    def ratio(values: np.ndarray) -> np.ndarray:
        return np.exp(0.8 * values[:, 0] - 0.32)

    report = diagnose_ratio(
        numerator_train=numerator_train,
        denominator_train=denominator_train,
        numerator_holdout=numerator_holdout,
        denominator_holdout=denominator_holdout,
        numerator_train_weights=np.ones(100),
        denominator_train_weights=np.ones(100),
        numerator_holdout_weights=np.ones(80),
        denominator_holdout_weights=np.ones(80),
        train_ratios=(ratio(numerator_train), ratio(denominator_train)),
        holdout_ratios=(ratio(numerator_holdout), ratio(denominator_holdout)),
        features=("x", "y"),
        history=(
            {"epoch": 1, "training_loss": 0.7, "validation_loss": 0.71},
            {"epoch": 2, "training_loss": 0.6, "validation_loss": 0.62},
        ),
        bins=10,
        output_directory=tmp_path / "plots",
    )
    assert set(report.metrics) == {
        "calibration",
        "classification",
        "loss",
        "normalization",
        "overtraining",
        "ratio_tails",
        "saturation",
    }
    assert report.metrics["saturation"]["warning"] is False
    assert report.curves["training_log_ratio_calibration"]["weighted_rmse"] >= 0.0
    assert set(report.feature_metrics) == {"x", "y"}
    if importlib.util.find_spec("matplotlib") is None:
        assert report.figure_paths == ()
    else:
        assert {path.name for path in report.figure_paths} == {
            "calibration.png",
            "calibration_log_ratio.png",
            "loss.png",
            "overtraining.png",
            "reweighted_x.png",
            "reweighted_y.png",
        }
    report_path, manifest_path = report.write(tmp_path)
    assert json.loads(report_path.read_text())["metrics"]["loss"]["epochs"] == 2
    manifest = ArtifactManifest.load(manifest_path)
    manifest.verify(tmp_path)
    assert set(manifest.metadata["checks"]) == {
        "calibration",
        "loss",
        "normalization",
        "overfit",
        "reweighting",
    }


def test_ratio_diagnostics_honor_individual_check_selection() -> None:
    numerator = np.asarray([[0.2], [0.6], [1.0], [1.4]])
    denominator = np.asarray([[-1.0], [-0.4], [0.0], [0.4]])
    unit = np.ones(4)
    report = diagnose_ratio(
        numerator_train=numerator,
        denominator_train=denominator,
        numerator_holdout=numerator + 0.1,
        denominator_holdout=denominator + 0.1,
        numerator_train_weights=unit,
        denominator_train_weights=unit,
        numerator_holdout_weights=unit,
        denominator_holdout_weights=unit,
        train_ratios=(np.exp(numerator[:, 0]), np.exp(denominator[:, 0])),
        holdout_ratios=(
            np.exp(numerator[:, 0] + 0.1),
            np.exp(denominator[:, 0] + 0.1),
        ),
        features=("x",),
        checks=("normalization",),
        bins=2,
    )
    assert set(report.metrics) == {"loss", "normalization"}
    assert report.feature_metrics == {}
    assert report.curves == {}
    assert report.checks == ("normalization",)


def test_native_ratio_training_exports_embedded_scaler_onnx(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(12)
    numerator = rng.normal(0.9, 1.0, size=(160, 2)).astype(np.float32)
    denominator = rng.normal(0.0, 1.0, size=(160, 2)).astype(np.float32)
    result = RatioTrainer(
        NativeRatioBackend(),
        RatioTrainingConfig(
            ensemble_size=1,
            hidden_layers=1,
            neurons=12,
            epochs=8,
            batch_size=64,
            learning_rate=0.01,
            validation_fraction=0.2,
            holdout_fraction=0.2,
            patience=4,
            run_diagnostics=False,
            seed=5,
        ),
    ).fit(
        numerator,
        denominator,
        features=("x", "y"),
        output_directory=tmp_path,
    )
    member = result.members[0]
    portable = OnnxNativeRatioMember(member.files["log-ratio-onnx"])
    probe = np.vstack([numerator[:10], denominator[:10]])
    assert portable(probe) == pytest.approx(member.evaluator(probe), rel=2.0e-4)
    assert result.backend == "native"
    ArtifactManifest.load(result.manifest_path).verify(tmp_path)
