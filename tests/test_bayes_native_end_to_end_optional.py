from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hnsbi.bayes import (
    DUAL_ARTIFACT_NAMES,
    DualTrainingData,
    IndependentNormal,
    ProposalDataset,
    load_dual_model,
    sample_posterior,
    train_native_dual,
    verify_dual_artifact_manifest,
)


def _tiny_native_config() -> dict:
    flow = {
        "architecture": "quadratic_spline",
        "n_coupling_layers": 1,
        "hidden_features": 4,
        "hidden_layers": 1,
        "spline_num_bins": 2,
        "spline_tail_bound": 3.0,
        "training": {
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 1.0e-3,
            "validation_fraction": 0.2,
            "early_stopping_patience": 1,
            "seed": 11,
            "device": "cpu",
        },
        "onnx_opset": 17,
    }
    ratio = {
        "backend": "native",
        "ensemble_size": 1,
        "training": {
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 1.0e-3,
            "hidden_layers": 1,
            "neurons": 4,
            "activation": "swish",
            "validation_fraction": 0.2,
            "holdout_fraction": 0.2,
            "early_stopping_patience": 1,
            "seed": 12,
        },
        "onnx_opset": 17,
    }
    return {
        "theta_features": ["theta"],
        "posterior_flow": flow,
        "posterior_ratio": ratio,
        "likelihood_flow": {
            **flow,
            "training": {**flow["training"], "seed": 13},
        },
        "likelihood_ratio": {
            **ratio,
            "normalization": "conditional_reference_mean",
            "training": {**ratio["training"], "seed": 14},
        },
        "normalizer": {
            "reference_draws_per_context": 2,
            "contexts": 8,
            "hidden_features": 4,
            "hidden_layers": 1,
            "training": {
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 1.0e-3,
                "validation_fraction": 0.25,
                "early_stopping_patience": 1,
                "seed": 15,
                "device": "cpu",
            },
            "onnx_opset": 17,
        },
        "defensive_epsilon": 0.0,
    }


def _proposal(seed: int, *, design: str, rows: int = 32) -> ProposalDataset:
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(rows, 1))
    observation = theta + rng.normal(scale=0.35, size=(rows, 1))
    return ProposalDataset(
        theta=theta,
        observation=observation,
        simulation_ids=np.arange(seed * 1_000, seed * 1_000 + rows),
        design=design,
        parameter_names=("theta",),
        observation_names=("x",),
    )


def test_native_dual_trains_loads_and_runs_portable_consensus(tmp_path: Path):
    for dependency in ("torch", "onnx", "onnxruntime", "onnxscript"):
        pytest.importorskip(dependency)

    rho = IndependentNormal(mean=np.zeros(1), scale=np.ones(1))
    data = DualTrainingData(
        rho_flow=_proposal(1, design="rho-flow"),
        rho_ratio=_proposal(2, design="rho-ratio"),
        nu_flow=_proposal(3, design="nu-flow"),
        kappa_ratio=_proposal(4, design="kappa-ratio"),
        validation=_proposal(5, design="independent-validation", rows=16),
    )

    result = train_native_dual(
        data,
        rho=rho,
        bayesian_config=_tiny_native_config(),
        observation_features=("x",),
        output_directory=tmp_path / "dual",
        seed=101,
    )

    assert tuple(result.manifest.artifacts) == DUAL_ARTIFACT_NAMES
    assert set(result.stages) == set(DUAL_ARTIFACT_NAMES)
    for artifact_name in ("q_phi", "q_eta"):
        split = result.stages[artifact_name].training.flow.metadata["split"]
        assert split == {
            "external_validation": True,
            "training_rows": 32,
            "validation_rows": 16,
        }
    for artifact_name in ("r_p", "r_c"):
        validation = json.loads(
            (
                result.stages[artifact_name].artifact_manifest_path.parent
                / "validation.json"
            ).read_text(encoding="utf-8")
        )
        assert validation["split_source"] == "configured"
        assert validation["split_counts"] == {
            "train": 32,
            "validation": 16,
            "holdout": 16,
        }
    normalizer_validation = json.loads(
        (
            result.stages["z_c"].artifact_manifest_path.parent / "validation.json"
        ).read_text(encoding="utf-8")
    )
    assert normalizer_validation["split_source"] == "configured"
    assert normalizer_validation["split_counts"] == {
        "train": 8,
        "validation": 8,
        "holdout": 0,
    }
    verified = verify_dual_artifact_manifest(result.manifest_path)
    assert verified.verification_errors() == []

    portable = load_dual_model(result.manifest_path, rho=rho, verify=True)
    posterior = sample_posterior(
        portable,
        np.asarray([[0.25]]),
        n=12,
        route="dual",
        rng=np.random.default_rng(202),
    )

    assert posterior.metadata["route"] == "dual"
    assert posterior.values.shape == (12, 1)
    assert np.isfinite(posterior.log_weights).all()
    assert np.isclose(np.sum(posterior.weights), 1.0)
    assert 1.0 <= posterior.ess <= 12.0
