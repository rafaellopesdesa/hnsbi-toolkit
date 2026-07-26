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
    ):
        self.weight_sums.append((numerator_weights.sum(), denominator_weights.sum()))
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
