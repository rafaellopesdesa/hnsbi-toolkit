from __future__ import annotations

import numpy as np

from hnsbi import Project
from hnsbi.ratios import RatioBackendResult


def _config() -> dict:
    return {
        "schema_version": "1.0",
        "features": ["x", "y"],
        "frequentist": {
            "reference": {
                "kind": "pyarrow",
                "registry_key": "reference",
            },
            "samples": [
                {
                    "name": "signal",
                    "source": {
                        "kind": "awkward",
                        "registry_key": "signal",
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
                "n_coupling_layers": 4,
                "hidden_features": 32,
                "hidden_layers": 2,
                "scale_clip": 1.5,
                "training": {
                    "epochs": 4,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "validation_fraction": 0.2,
                    "early_stopping_patience": 2,
                    "device": "cpu",
                },
            },
            "ratios": {
                "backend": "nsbi_common_utils",
                "ensemble_size": 2,
                "training": {
                    "epochs": 3,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "hidden_layers": 2,
                    "neurons": 16,
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
                    "bounds": [0.0, 5.0],
                }
            ],
            "workspace": {
                "backend": "nsbi_common_utils",
                "measurement": "measurement",
                "channel": "SR",
                "output_path": "artifacts/workspace.json",
            },
        },
    }


def test_project_builds_intensity_and_registry_sources() -> None:
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    project = Project.load(
        _config(),
        registry={
            "reference": values,
            "signal": values + 1,
            "background": values - 1,
        },
    )
    model = project.intensity_model()
    assert model.component_names == ("signal", "background")
    assert model.expected_yield({"mu": 2.0}) == 36.0
    assert project.reference_source().materialize().values.shape == (10, 2)
    assert set(project.sample_sources()) == {"signal", "background"}


def test_project_translates_flow_and_ratio_configs() -> None:
    values = np.ones((3, 2), dtype=np.float32)
    project = Project.load(
        _config(),
        registry={
            "reference": values,
            "signal": values,
            "background": values,
        },
    )
    flow, training = project.flow_configs()
    assert flow.flow_type == "realnvp"
    assert flow.num_transforms == 4
    assert training.patience == 2
    assert training.learning_rate_factor == 0.5
    ratio = project.ratio_config()
    assert ratio.ensemble_size == 2
    assert ratio.neurons == 16


def test_project_systematic_yield_override_precedes_weight_inference(tmp_path) -> None:
    config = _config()
    config["output_dir"] = str(tmp_path / "artifacts")
    config["frequentist"]["parameters"].append(
        {
            "name": "alpha",
            "role": "nuisance",
            "nominal": 0.0,
            "bounds": [-5.0, 5.0],
            "constraint": {
                "kind": "normal",
                "mean": 0.0,
                "sigma": 1.0,
            },
        }
    )
    config["frequentist"]["systematics"] = [
        {
            "name": "calibration",
            "parameter": "alpha",
            "type": "norm_plus_shape",
            "interpolation": "nsbi_code4p",
            "variations": [
                {
                    "sample": "signal",
                    "up": {"kind": "pyarrow", "registry_key": "signal_up"},
                    "down": {"kind": "pyarrow", "registry_key": "signal_down"},
                    "yield_up": 1.25,
                }
            ],
        }
    ]
    nominal = np.arange(12, dtype=np.float64).reshape(6, 2)
    project = Project.load(
        config,
        registry={
            "reference": nominal,
            "signal": nominal,
            "background": nominal,
            "signal_up": nominal[:3],
            "signal_down": np.tile(nominal, (2, 1)),
        },
    )

    class StubBackend:
        name = "stub"

        def train_member(self, *, output_directory, **kwargs):
            artifact = output_directory / "ratio.onnx"
            artifact.write_bytes(b"stub")
            return RatioBackendResult(
                evaluator=lambda values: np.ones(len(values)),
                files={"model": artifact},
            )

    result = project.train_systematics(backend=StubBackend())
    trained = result["calibration"]["signal"]
    assert trained["yield_up"] == 1.25
    assert trained["yield_down"] == 2.0
    assert trained["yield_source"] == {
        "up": "configured",
        "down": "integrated_mc_weights",
    }
