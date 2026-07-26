from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hnsbi.bayes import (
    BayesianTrainingBackend,
    ConditionalFlowStageConfig,
    LogNormalizerStageConfig,
    NativeConditionalDensity,
    NativeDualBackend,
    NativeLogRatio,
    NativeTorchRatioBackend,
    PairedClassifierDataset,
    RatioStageConfig,
)
from hnsbi.bayes.native_backend import (
    _paired_train_validation_holdout,
    _select_contexts,
)
from hnsbi.ratios import RatioTrainer, RatioTrainingConfig


def _configuration() -> dict:
    path = Path("examples/configs/dual_complete.json")
    return json.loads(path.read_text(encoding="utf-8"))["bayesian"]


def test_configured_backend_is_lazy_and_obeys_training_protocol(tmp_path):
    backend = NativeDualBackend.from_config(
        _configuration(),
        observation_features=("x1", "x2", "x3"),
        output_directory=tmp_path,
    )

    assert isinstance(backend, BayesianTrainingBackend)
    assert backend.theta_features == ("mu", "alpha")
    assert backend.observation_features == ("x1", "x2", "x3")
    assert backend.posterior_flow.training.device == "auto"
    assert backend.normalizer.reference_draws_per_context == 64
    assert backend.normalizer.contexts == 4096
    with pytest.raises(RuntimeError, match="after z_c"):
        _ = backend.manifest


def test_dual_conditional_flow_rejects_realnvp():
    specification = copy.deepcopy(_configuration()["posterior_flow"])
    specification["architecture"] = "realnvp"

    with pytest.raises(ValueError, match="quadratic"):
        ConditionalFlowStageConfig.from_mapping(
            specification,
            target_features=("theta",),
            context_features=("x",),
        )


def test_bayesian_ratio_config_consumes_only_supported_normalization():
    likelihood = copy.deepcopy(_configuration()["likelihood_ratio"])
    parsed = RatioStageConfig.from_mapping(likelihood)
    assert parsed.normalization == "conditional_reference_mean"
    assert not parsed.training.run_diagnostics

    posterior = copy.deepcopy(_configuration()["posterior_ratio"])
    assert RatioStageConfig.from_mapping(posterior).normalization is None

    posterior["diagnostics"] = {"overtraining": True}
    with pytest.raises(ValueError, match="Unknown ratio model fields"):
        RatioStageConfig.from_mapping(posterior)

    likelihood["normalization"] = "independent_reference_mean"
    with pytest.raises(ValueError, match="Bayesian ratio normalization"):
        RatioStageConfig.from_mapping(likelihood)


def test_normalizer_config_rejects_ignored_max_events():
    specification = copy.deepcopy(_configuration()["normalizer"])
    specification["training"]["max_events"] = 100
    with pytest.raises(ValueError, match="Unknown normalizer training fields"):
        LogNormalizerStageConfig.from_mapping(specification)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["likelihood_ratio"].pop("normalization"),
            "likelihood_ratio requires",
        ),
        (
            lambda value: value["posterior_ratio"].update(
                {"normalization": "conditional_reference_mean"}
            ),
            "posterior_ratio must not",
        ),
    ],
)
def test_native_backend_enforces_ratio_normalization_contract(
    tmp_path,
    mutate,
    message,
):
    specification = copy.deepcopy(_configuration())
    mutate(specification)
    with pytest.raises(ValueError, match=message):
        NativeDualBackend.from_config(
            specification,
            observation_features=("x1", "x2", "x3"),
            output_directory=tmp_path,
        )


def test_native_conditional_adapter_batches_every_context():
    class Flow:
        config = SimpleNamespace(n_features=2)

        def log_prob(self, target, *, context):
            return np.sum(target, axis=1) + np.sum(context, axis=1)

        def base_to_data(self, noise, *, context):
            return noise + context[:, :1]

    adapter = NativeConditionalDensity(Flow())
    contexts = np.asarray([[1.0], [2.0], [3.0]])
    draws = adapter.sample(4, context=contexts, rng=np.random.default_rng(12))

    assert draws.shape == (3, 4, 2)
    assert np.allclose(
        adapter.log_prob(np.ones((3, 2)), context=contexts),
        np.asarray([3.0, 4.0, 5.0]),
    )


def test_native_ratio_adapter_preserves_theta_x_order():
    class Ensemble:
        def __init__(self):
            self.values = None

        def log_ratio(self, values):
            self.values = np.asarray(values)
            return np.sum(values, axis=1)

    theta = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    observation = np.asarray([[10.0], [20.0]])

    posterior_ensemble = Ensemble()
    posterior = NativeLogRatio(posterior_ensemble, "r_p")
    posterior.log_ratio(theta, context=observation)
    assert np.array_equal(
        posterior_ensemble.values,
        np.column_stack([theta, observation]),
    )

    likelihood_ensemble = Ensemble()
    likelihood = NativeLogRatio(likelihood_ensemble, "r_c")
    likelihood.log_ratio(observation, context=theta)
    assert np.array_equal(
        likelihood_ensemble.values,
        np.column_stack([theta, observation]),
    )


def test_normalizer_context_selection_uses_requested_count():
    values = np.arange(30, dtype=np.float64).reshape(15, 2)
    selected = _select_contexts(
        values,
        count=11,
        rng=np.random.default_rng(4),
    )

    assert selected.shape == (11, 2)
    assert set(map(tuple, selected)).issubset(set(map(tuple, values)))
    assert len(np.unique(selected, axis=0)) == 11
    with pytest.raises(ValueError, match="only 15"):
        _select_contexts(values, count=16, rng=np.random.default_rng(5))


def test_ratio_split_keeps_paired_groups_out_of_validation_and_holdout():
    pairs = PairedClassifierDataset(
        positive=np.arange(120, dtype=np.float64).reshape(40, 3),
        negative=-np.arange(120, dtype=np.float64).reshape(40, 3),
        group_ids=np.arange(40),
        shared_quantity="theta",
    )
    training, validation, holdout = _paired_train_validation_holdout(
        pairs,
        validation_fraction=0.2,
        holdout_fraction=0.25,
        seed=7,
    )

    assert not np.intersect1d(training.group_ids, validation.group_ids).size
    assert not np.intersect1d(training.group_ids, holdout.group_ids).size
    assert not np.intersect1d(validation.group_ids, holdout.group_ids).size
    assert (
        len(training.group_ids) + len(validation.group_ids) + len(holdout.group_ids)
        == 40
    )


def test_native_ratio_real_stack_exports_split_input_onnx(tmp_path):
    for dependency in ("torch", "onnx", "onnxruntime", "onnxscript"):
        pytest.importorskip(dependency)
    rng = np.random.default_rng(9)
    positive = rng.normal(0.7, 1.0, size=(48, 3))
    negative = rng.normal(-0.7, 1.0, size=(48, 3))
    validation = PairedClassifierDataset(
        positive=positive[-12:],
        negative=negative[-12:],
        group_ids=np.arange(12),
        shared_quantity="observation",
    )
    backend = NativeTorchRatioBackend(
        theta_features=("theta",),
        observation_features=("x1", "x2"),
        artifact_name="r_p",
        validation_pairs=validation,
    )
    result = RatioTrainer(
        backend,
        RatioTrainingConfig(
            ensemble_size=1,
            hidden_layers=1,
            neurons=8,
            epochs=2,
            batch_size=16,
            learning_rate=1.0e-3,
            validation_fraction=0.2,
            holdout_fraction=0.2,
            patience=2,
            run_diagnostics=False,
        ),
    ).fit(
        positive[:-12],
        negative[:-12],
        features=("theta", "x1", "x2"),
        output_directory=tmp_path,
    )

    graph = result.members[0].files["dual-log-ratio-onnx"]
    assert graph.is_file()
    assert result.members[0].metadata["onnx_parity"]["passed"]
