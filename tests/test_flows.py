from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hnsbi.artifacts import write_artifact_manifest
from hnsbi.flow_diagnostics import diagnose_flow
from hnsbi.flows import (
    AffineStandardizer,
    FlowConfig,
    FlowOnnxBundle,
    FlowTrainer,
    FlowTrainingConfig,
)


def test_weighted_standardizer_round_trip_and_jacobian():
    values = np.array([[0.0, -2.0], [10.0, 2.0]], dtype=np.float32)
    scaler = AffineStandardizer.fit(values, weights=[3.0, 1.0])

    assert np.allclose(scaler.mean, [2.5, -1.0])
    assert np.allclose(
        scaler.inverse_transform(scaler.transform(values)),
        values,
        atol=1e-6,
    )
    assert np.isclose(
        scaler.forward_log_abs_det,
        -np.log(scaler.scale).sum(),
    )


def test_flow_config_aliases_and_conditional_contract():
    assert FlowConfig(2, flow_type="rqs").flow_type == "quadratic-spline"
    with pytest.raises(ValueError, match="Conditional context"):
        FlowConfig(2, flow_type="realnvp", context_features=1)
    with pytest.raises(ValueError, match="Unknown flow_type"):
        FlowConfig(2, flow_type="made-up")


def test_flow_random_split_is_reproducible_and_disjoint():
    training, validation = FlowTrainer.random_split_indices(
        20,
        0.25,
        rng=np.random.default_rng(17),
    )
    repeated = FlowTrainer.random_split_indices(
        20,
        0.25,
        rng=np.random.default_rng(17),
    )

    assert len(training) == 15
    assert len(validation) == 5
    assert not set(training).intersection(validation)
    assert set(training).union(validation) == set(range(20))
    np.testing.assert_array_equal(training, repeated[0])
    np.testing.assert_array_equal(validation, repeated[1])


def test_onnx_bundle_loader_verifies_feature_signature(tmp_path):
    files = {}
    for role in (
        "log-prob-onnx",
        "base-to-data-onnx",
        "data-to-base-onnx",
    ):
        path = tmp_path / f"{role}.onnx"
        path.write_bytes(role.encode())
        files[role] = path
    manifest = write_artifact_manifest(
        tmp_path / "reference.manifest.json",
        artifact_type="reference-flow-onnx-bundle",
        files=files,
        metadata={
            "features": ["x", "y"],
            "context_names": [],
            "conditional": False,
        },
    )
    bundle = FlowOnnxBundle.load(manifest, expected_features=["x", "y"])
    assert bundle.features == ("x", "y")
    with pytest.raises(ValueError, match="feature order mismatch"):
        FlowOnnxBundle.load(manifest, expected_features=["y", "x"])


class _GaussianFlow:
    features = ("x", "y")
    is_conditional = False

    def sample(self, n, *, rng=None, context=None):
        assert context is None
        generator = np.random.default_rng() if rng is None else rng
        return generator.normal(size=(n, 2)).astype(np.float32)

    def log_prob(self, values, *, context=None):
        assert context is None
        values = np.asarray(values)
        return -0.5 * np.square(values).sum(axis=1)


def test_flow_diagnostics_are_available_without_plot_dependencies():
    rng = np.random.default_rng(11)
    reference = rng.normal(size=(500, 2))
    result = diagnose_flow(
        _GaussianFlow(),
        reference,
        weights=np.linspace(0.5, 1.5, len(reference)),
        n_generated=400,
        rng=np.random.default_rng(12),
    )

    assert result.report.reference_count == 500
    assert result.report.generated_count == 400
    assert [item.feature for item in result.report.features] == ["x", "y"]
    assert 0 <= result.report.features[0].weighted_ks_distance <= 1


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is an optional flows dependency",
)
def test_realnvp_training_and_deterministic_sampling_smoke():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(80, 2)).astype(np.float32)
    trainer = FlowTrainer(
        FlowConfig(
            n_features=2,
            flow_type="realnvp",
            num_transforms=2,
            hidden_features=8,
            num_blocks=1,
        ),
        FlowTrainingConfig(
            epochs=2,
            batch_size=32,
            patience=2,
            validation_fraction=0.2,
        ),
    )

    result = trainer.fit(values, features=("x", "y"), seed=4)
    first = result.flow.sample(5, rng=np.random.default_rng(19))
    second = result.flow.sample(5, rng=np.random.default_rng(19))

    assert np.allclose(first, second)
    assert result.flow.log_prob(values[:5]).shape == (5,)
    assert len(result.history) == 2
