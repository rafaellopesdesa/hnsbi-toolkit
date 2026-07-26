from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hnsbi.flows import (
    FlowConfig,
    FlowOnnxBundle,
    FlowTrainer,
    FlowTrainingConfig,
    ReferenceFlow,
)
from hnsbi.onnx import OnnxRunner

pytestmark = pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("torch", "onnx", "onnxruntime")
    ),
    reason="PyTorch, ONNX, and ONNX Runtime are optional",
)


def _training_config() -> FlowTrainingConfig:
    return FlowTrainingConfig(
        epochs=1,
        batch_size=32,
        validation_fraction=0.2,
        patience=1,
    )


def _assert_exact_opset_and_dynamic_batch(path) -> None:
    import onnx

    model = onnx.load(path)
    default_opset = next(
        item.version for item in model.opset_import if item.domain in ("", "ai.onnx")
    )
    assert default_opset == 17
    first_dimension = model.graph.input[0].type.tensor_type.shape.dim[0]
    assert first_dimension.dim_param == "batch"


@pytest.mark.parametrize("flow_type", ["realnvp", "quadratic-spline"])
def test_flow_training_export_and_runtime_parity(flow_type, tmp_path):
    rng = np.random.default_rng(14)
    values = rng.normal(size=(96, 2)).astype(np.float32)
    result = FlowTrainer(
        FlowConfig(
            n_features=2,
            flow_type=flow_type,
            num_transforms=2,
            hidden_features=12,
            num_blocks=1,
            num_bins=4,
            tail_bound=3.0,
        ),
        _training_config(),
    ).fit(values, features=["x", "y"], seed=8)
    bundle = result.flow.export_onnx(
        tmp_path / flow_type,
        example_values=values[:8],
    )
    loaded = FlowOnnxBundle.load(
        bundle.manifest_path,
        expected_features=["x", "y"],
    )

    for path in (
        loaded.log_prob_path,
        loaded.base_to_data_path,
        loaded.data_to_base_path,
    ):
        _assert_exact_opset_and_dynamic_batch(path)

    # Exercise the same graph at batches smaller and larger than the export
    # example, including both the learned interior and identity tails.
    for rows in (1, 5, 17):
        interior = rng.uniform(-2.0, 2.0, size=(rows, 2)).astype(np.float32)
        tails = np.resize(
            np.array([[-5.5, 5.5], [6.0, -6.0]], dtype=np.float32),
            (rows, 2),
        )
        for standardized in (interior, tails):
            original = result.flow.scaler.inverse_transform(standardized)
            base = rng.normal(size=(rows, 2)).astype(np.float32)
            if standardized is tails:
                base = tails
            parity = loaded.parity(
                result.flow,
                original,
                base=base,
                atol=3e-5,
                rtol=3e-4,
            )
            for report in parity.values():
                report.assert_close()
            native_codes = result.flow.data_to_base(original)
            np.testing.assert_allclose(
                result.flow.base_to_data(native_codes),
                original,
                atol=4e-5,
                rtol=4e-4,
            )
            portable_codes = loaded.data_to_base(original)
            np.testing.assert_allclose(
                loaded.base_to_data(portable_codes),
                original,
                atol=4e-5,
                rtol=4e-4,
            )

    native_sample = result.flow.sample(17, rng=np.random.default_rng(3))
    portable_sample = loaded.sample(17, rng=np.random.default_rng(3))
    np.testing.assert_allclose(portable_sample, native_sample, atol=3e-5, rtol=3e-4)
    np.testing.assert_allclose(
        loaded.sample(5, rng=np.random.default_rng(4)),
        loaded.sample(5, rng=np.random.default_rng(4)),
    )


def test_conditional_spline_export_broadcast_and_sampling(tmp_path):
    rng = np.random.default_rng(29)
    context = rng.uniform(-1.0, 1.0, size=(128, 1)).astype(np.float32)
    noise = rng.normal(scale=0.45, size=(128, 2)).astype(np.float32)
    values = (
        np.column_stack((context[:, 0], -0.5 * context[:, 0])).astype(np.float32)
        + noise
    )
    result = FlowTrainer(
        FlowConfig(
            n_features=2,
            flow_type="quadratic-spline",
            context_features=1,
            num_transforms=2,
            hidden_features=12,
            num_blocks=1,
            num_bins=4,
            tail_bound=3.0,
        ),
        _training_config(),
    ).fit(
        values,
        features=["x", "y"],
        context=context,
        context_names=["theta"],
        seed=31,
    )
    bundle = result.flow.export_onnx(
        tmp_path / "conditional",
        example_values=values[:8],
        example_context=context[:8],
    )
    loaded = FlowOnnxBundle.load(
        bundle.manifest_path,
        expected_features=["x", "y"],
    )
    assert loaded.conditional
    assert loaded.context_names == ("theta",)

    one_row_context = np.array([[0.35]], dtype=np.float32)
    for rows in (1, 5, 17):
        standardized = rng.uniform(-2.0, 2.0, size=(rows, 2)).astype(np.float32)
        standardized[::2] = np.array([5.5, -5.5], dtype=np.float32)
        original = result.flow.scaler.inverse_transform(standardized)
        base = rng.normal(size=(rows, 2)).astype(np.float32)
        parity = loaded.parity(
            result.flow,
            original,
            base=base,
            context=one_row_context,
            atol=4e-5,
            rtol=4e-4,
        )
        for report in parity.values():
            report.assert_close()
        portable_codes = loaded.data_to_base(original, context=one_row_context)
        np.testing.assert_allclose(
            loaded.base_to_data(portable_codes, context=one_row_context),
            original,
            atol=5e-5,
            rtol=5e-4,
        )

        # The graph itself, not just FlowOnnxBundle, supports a one-row
        # context broadcast against an arbitrary base batch.
        direct = OnnxRunner(loaded.base_to_data_path).run(
            {"base": base, "context": one_row_context}
        )["features"]
        native = result.flow.base_to_data(base, context=one_row_context)
        np.testing.assert_allclose(direct, native, atol=4e-5, rtol=4e-4)

    native_sample = result.flow.sample(
        17,
        rng=np.random.default_rng(12),
        context=one_row_context,
    )
    portable_sample = loaded.sample(
        17,
        rng=np.random.default_rng(12),
        context=one_row_context,
    )
    np.testing.assert_allclose(portable_sample, native_sample, atol=4e-5, rtol=4e-4)


def test_native_checkpoint_round_trip_and_integrity(tmp_path):
    rng = np.random.default_rng(44)
    values = rng.normal(size=(80, 2)).astype(np.float32)
    result = FlowTrainer(
        FlowConfig(
            n_features=2,
            flow_type="realnvp",
            num_transforms=2,
            hidden_features=8,
            num_blocks=1,
        ),
        _training_config(),
    ).fit(values, features=("x", "y"), seed=45)
    checkpoint, manifest = result.save_checkpoint(tmp_path / "flow.pt")

    loaded = ReferenceFlow.load(
        checkpoint,
        expected_features=("x", "y"),
    )
    np.testing.assert_allclose(
        loaded.log_prob(values[:12]),
        result.flow.log_prob(values[:12]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert manifest.is_file()

    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(RuntimeError, match="failed verification"):
        ReferenceFlow.load(checkpoint, expected_features=("x", "y"))
