from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hnsbi.artifacts import ArtifactManifest
from hnsbi.ratios import (
    RatioBackendResult,
    RatioEnsemble,
    RatioTrainer,
    RatioTrainingConfig,
    normalize_class_weights,
)


def test_ratio_ensemble_uses_arithmetic_mean_in_ratio_space():
    ensemble = RatioEnsemble(
        [
            lambda values: np.full(len(values), 1.0),
            lambda values: np.full(len(values), 9.0),
        ]
    )
    values = np.zeros((3, 2))

    assert np.allclose(ensemble(values), 5.0)
    assert np.allclose(ensemble.standard_deviation(values), 4.0)
    assert np.allclose(ensemble.log_ratio(values), np.log(5.0))


def test_ratio_ensemble_validates_member_outputs():
    values = np.zeros((2, 1))
    with pytest.raises(ValueError, match="negative ratio"):
        RatioEnsemble([lambda _: np.array([1.0, -1.0])])(values)
    with pytest.raises(ValueError, match="predictions"):
        RatioEnsemble([lambda _: np.array([1.0])])(values)


def test_class_weights_are_normalized_independently():
    first = normalize_class_weights([1.0, 3.0])
    second = normalize_class_weights([10.0, 20.0, 30.0])

    assert np.isclose(first.sum(), 1.0)
    assert np.isclose(second.sum(), 1.0)
    assert np.allclose(first, [0.25, 0.75])


class _RecordingBackend:
    name = "recording"

    def __init__(self):
        self.weight_sums = []
        self.partitioning = []

    def train_member(
        self,
        *,
        numerator_values,
        denominator_values,
        numerator_weights,
        denominator_weights,
        features,
        output_directory,
        member_index,
        numerator_name,
        denominator_name,
        config,
        numerator_split=None,
        denominator_split=None,
        numerator_groups=None,
        denominator_groups=None,
    ):
        self.weight_sums.append((numerator_weights.sum(), denominator_weights.sum()))
        self.partitioning.append(
            {
                "numerator_split": numerator_split,
                "denominator_split": denominator_split,
                "numerator_groups": numerator_groups,
                "denominator_groups": denominator_groups,
            }
        )
        artifact = output_directory / "member.txt"
        artifact.write_text(str(member_index), encoding="utf-8")
        value = float(member_index + 1)
        return RatioBackendResult(
            evaluator=lambda events, value=value: np.full(len(events), value),
            files={"fake-model": artifact},
            metadata={"member": member_index},
        )


def test_ratio_trainer_delegates_and_manifests_bundle(tmp_path):
    backend = _RecordingBackend()
    trainer = RatioTrainer(
        backend,
        RatioTrainingConfig(
            ensemble_size=2,
            epochs=1,
            run_diagnostics=False,
        ),
    )
    numerator = np.array([[0.0], [1.0]], dtype=np.float32)
    denominator = np.array([[2.0], [3.0], [4.0]], dtype=np.float32)

    result = trainer.fit(
        numerator,
        denominator,
        features=("x",),
        output_directory=tmp_path,
        numerator_weights=[2.0, 8.0],
        denominator_weights=[1.0, 1.0, 2.0],
    )

    assert backend.weight_sums == [(1.0, 1.0), (1.0, 1.0)]
    assert np.allclose(result.ensemble(np.zeros((4, 1))), 1.5)
    manifest = ArtifactManifest.load(result.manifest_path)
    manifest.verify(Path(tmp_path))
    assert manifest.metadata["ensemble_reduction"] == ("arithmetic-mean-of-ratios")


def test_ratio_trainer_forwards_aligned_split_and_group_metadata(tmp_path):
    backend = _RecordingBackend()
    numerator_split = np.asarray(["train", "train", "validation", "holdout", "train"])
    numerator_groups = np.asarray([10, 10, 11, 12, 13])
    trainer = RatioTrainer(
        backend,
        RatioTrainingConfig(
            ensemble_size=1,
            epochs=1,
            run_diagnostics=False,
        ),
    )

    trainer.fit(
        np.arange(5, dtype=np.float32).reshape(-1, 1),
        np.arange(6, dtype=np.float32).reshape(-1, 1),
        features=("x",),
        output_directory=tmp_path,
        numerator_split=numerator_split,
        numerator_groups=numerator_groups,
    )

    np.testing.assert_array_equal(
        backend.partitioning[0]["numerator_split"],
        numerator_split,
    )
    np.testing.assert_array_equal(
        backend.partitioning[0]["numerator_groups"],
        numerator_groups,
    )
    assert backend.partitioning[0]["denominator_split"] is None
    with pytest.raises(ValueError, match="numerator_groups"):
        trainer.fit(
            np.arange(5, dtype=np.float32).reshape(-1, 1),
            np.arange(6, dtype=np.float32).reshape(-1, 1),
            features=("x",),
            output_directory=tmp_path / "invalid",
            numerator_groups=[1, 2],
        )
