from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import yaml

from hnsbi import (
    AsimovBuilder,
    ConfigError,
    ExtendedUnbinnedLikelihood,
    FNFModelArtifacts,
    Project,
    ToyGenerator,
    load_workspace_model,
)

pa = pytest.importorskip("pyarrow")
torch = pytest.importorskip("torch")


def _source(key: str) -> dict:
    return {
        "kind": "pyarrow",
        "registry_key": key,
        "weight_column": "weight",
    }


def _config() -> dict:
    return {
        "schema_version": "2.0",
        "features": ["x"],
        "output_dir": "artifacts",
        "frequentist": {
            "reference": _source("reference"),
            "samples": [
                {
                    "name": "signal",
                    "source": _source("signal"),
                    "nominal_yield": 8.0,
                    "multiplier": "mu",
                },
                {
                    "name": "background",
                    "source": _source("background"),
                    "nominal_yield": 20.0,
                    "multiplier": "1",
                },
            ],
            "flow": {
                "architecture": "realnvp",
                "n_coupling_layers": 2,
                "hidden_features": 8,
                "hidden_layers": 1,
                "training": {
                    "epochs": 1,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                },
            },
            "ratios": {
                "backend": "native",
                "ensemble_size": 1,
                "training": {
                    "epochs": 1,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "hidden_layers": 1,
                    "neurons": 8,
                },
                "normalization": "independent_reference_mean",
            },
            "parameters": [
                {
                    "name": "mu",
                    "role": "poi",
                    "nominal": 1.0,
                    "bounds": [0.0, 5.0],
                },
                {
                    "name": "shift",
                    "role": "nuisance",
                    "nominal": 0.0,
                    "bounds": [-3.0, 3.0],
                    "constraint": {
                        "kind": "normal",
                        "mean": 0.0,
                        "sigma": 0.5,
                    },
                },
            ],
            "fnf": {
                "models": [
                    {
                        "name": "signal_shape",
                        "sample": "signal",
                        "nuisances": ["shift"],
                        "architecture": {
                            "num_layers": 1,
                            "hidden_features": [8],
                        },
                        "training": {
                            "epochs": 2,
                            "batch_size": 16,
                            "learning_rate": 0.001,
                            "validation_fraction": 0.2,
                            "holdout_fraction": 0.2,
                            "early_stopping_patience": 2,
                            "seed": 13,
                        },
                        "anchors": [
                            {
                                "name": "shift_down",
                                "point": {"shift": -0.5},
                                "source": {
                                    **_source("signal_down"),
                                    "group_column": "group",
                                },
                            },
                            {
                                "name": "shift_up",
                                "point": {"shift": 0.5},
                                "source": {
                                    **_source("signal_up"),
                                    "group_column": "group",
                                },
                            },
                        ],
                        "yield_anchors": {"shift": [0.9, 1.1]},
                        "output_path": ("artifacts/fnf/signal/fnf.manifest.json"),
                    }
                ]
            },
            "workspace": {
                "backend": "native",
                "measurement": "measurement",
                "channel": "SR",
                "output_path": "artifacts/workspace/workspace.json",
            },
        },
    }


def _table(values: np.ndarray, *, groups: bool = False):
    payload = {
        "x": pa.array(np.asarray(values, dtype=np.float32)),
        "weight": pa.array(np.ones(len(values), dtype=np.float64)),
    }
    if groups:
        payload["group"] = pa.array(np.arange(len(values), dtype=np.int64))
    return pa.table(payload)


def _project(tmp_path) -> Project:
    rng = np.random.default_rng(4)
    registry = {
        "reference": _table(rng.normal(size=64)),
        "signal": _table(rng.normal(0.4, 1.0, size=64)),
        "background": _table(rng.normal(-0.4, 1.0, size=64)),
        "signal_down": _table(rng.normal(-0.5, 1.0, size=64), groups=True),
        "signal_up": _table(rng.normal(0.5, 1.0, size=64), groups=True),
    }
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")
    return Project.load(path, registry=registry)


def test_yaml_fnf_configuration_materializes_grouped_anchors(tmp_path) -> None:
    project = _project(tmp_path)
    residual, training = project.fnf_configs("signal_shape")
    assert residual.nuisance_names == ("shift",)
    assert residual.nuisance_centers == (0.0,)
    assert residual.nuisance_scales == (0.5,)
    assert training.holdout_fraction == pytest.approx(0.2)

    anchors = project.fnf_anchors("signal_shape", max_events_per_anchor=12)
    assert [anchor.name for anchor in anchors] == ["shift_down", "shift_up"]
    assert all(len(anchor.values) == 12 for anchor in anchors)
    np.testing.assert_array_equal(anchors[0].groups, np.arange(12))


def test_yaml_fnf_rejects_uncovered_interaction() -> None:
    config = _config()
    model = config["frequentist"]["fnf"]["models"][0]
    config["frequentist"]["parameters"].append(
        {
            "name": "resolution",
            "role": "nuisance",
            "nominal": 0.0,
            "bounds": [-3.0, 3.0],
        }
    )
    model["nuisances"].append("resolution")
    model["interactions"] = [["shift", "resolution"]]
    model["anchors"].extend(
        [
            {
                "name": "resolution_down",
                "point": {"resolution": -1.0},
                "source": _source("resolution_down"),
            },
            {
                "name": "resolution_up",
                "point": {"resolution": 1.0},
                "source": _source("resolution_up"),
            },
        ]
    )
    with pytest.raises(ConfigError, match="joint non-nominal anchor"):
        Project.load(config)


def test_project_trains_loads_and_serializes_fnf_workspace(tmp_path) -> None:
    project = _project(tmp_path)
    reference_artifacts = project.train_reference(max_events=48, seed=11)
    reference = reference_artifacts.training.flow
    ratios = project.train_ratios(
        reference,
        max_events_per_sample=48,
        denominator_events=48,
        normalization_events=64,
        seed=12,
    )
    with pytest.raises(ValueError, match="reference-flow manifest"):
        project.train_fnf_systematics(
            reference=reference,
            ratios=ratios,
            max_events_per_anchor=48,
        )
    trained = project.train_fnf_systematics(
        reference=reference,
        ratios=ratios,
        reference_manifest=reference_artifacts.checkpoint_manifest,
        max_events_per_anchor=48,
    )
    assert set(trained) == {"signal"}
    assert isinstance(trained["signal"], FNFModelArtifacts)
    assert trained["signal"].manifest_path.is_file()
    assert trained["signal"].yield_morph.factor({"shift": 0.5}) == pytest.approx(1.1)

    loaded = project.load_fnf_systematics()
    assert loaded["signal"].yield_morph.factor({"shift": -0.5}) == pytest.approx(0.9)
    probe = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(-1, 1)
    transformed, log_det = loaded["signal"].residual.to_nominal(probe, {"shift": 0.0})
    np.testing.assert_allclose(transformed, probe, atol=1.0e-7)
    np.testing.assert_allclose(log_det, 0.0, atol=1.0e-7)

    class Sampler:
        def sample(self, n, *, rng=None):
            return (rng or np.random.default_rng()).normal(size=(n, 1))

    fnf_systematics = project.build_fnf_runtime_systematics(
        reference_density=reference,
        ratios=ratios,
        reference_manifest=reference_artifacts.checkpoint_manifest,
    )
    torch_log_prob = fnf_systematics["signal"].density.torch_log_prob(
        torch.as_tensor(probe),
        {"shift": 0.0},
    )
    assert torch_log_prob.shape == (len(probe),)
    assert torch.isfinite(torch_log_prob).all()
    result = AsimovBuilder(
        reference=reference,
        ratios=ratios.evaluators,
        normalizer=ratios.normalizer,
        intensity=project.intensity_model(),
        features=["x"],
        fnf_systematics=fnf_systematics,
    ).build(
        {"mu": 1.0, "shift": 0.0},
        n_events=64,
        seed=2,
        normalization="fixed",
    )
    ratio_manifests = {
        name: artifacts.manifest_path for name, artifacts in ratios.training.items()
    }
    alternate_reference_manifest = (
        reference_artifacts.checkpoint_manifest.parent
        / "alternate-reference.manifest.json"
    )
    alternate_reference_payload = json.loads(
        reference_artifacts.checkpoint_manifest.read_text(encoding="utf-8")
    )
    alternate_reference_payload["created_at"] = "different-reference-identity"
    alternate_reference_manifest.write_text(
        json.dumps(alternate_reference_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different reference-flow manifest"):
        project.build_fnf_runtime_systematics(
            reference_density=reference,
            ratios=ratios,
            reference_manifest=alternate_reference_manifest,
        )
    with pytest.raises(ValueError, match="different reference-flow manifest"):
        project.write_configured_workspace(
            result,
            reference_manifest=alternate_reference_manifest,
            ratio_manifests=ratio_manifests,
        )

    signal_ratio_manifest = ratio_manifests["signal"]
    alternate_ratio_manifest = (
        signal_ratio_manifest.parent / "alternate-signal.manifest.json"
    )
    alternate_ratio_payload = json.loads(
        signal_ratio_manifest.read_text(encoding="utf-8")
    )
    alternate_ratio_payload["created_at"] = "different-ratio-identity"
    alternate_ratio_manifest.write_text(
        json.dumps(alternate_ratio_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mismatched_ratio_manifests = {
        **ratio_manifests,
        "signal": alternate_ratio_manifest,
    }
    with pytest.raises(ValueError, match="different native ratio manifest"):
        project.write_configured_workspace(
            result,
            reference_manifest=reference_artifacts.checkpoint_manifest,
            ratio_manifests=mismatched_ratio_manifests,
        )

    export = project.write_configured_workspace(
        result,
        reference_manifest=reference_artifacts.checkpoint_manifest,
        ratio_manifests=ratio_manifests,
    )
    signal = next(
        sample
        for sample in export.workspace["channels"][0]["samples"]
        if sample["name"] == "signal"
    )
    assert signal["hnsbi"]["fnf_manifest"].endswith("fnf.manifest.json")
    recovered = load_workspace_model(export.path)
    assert set(recovered.fnf_manifests) == {"signal"}
    assert (
        recovered.fnf_manifests["signal"].resolve()
        == trained["signal"].manifest_path.resolve()
    )
    tampered_workspace = deepcopy(export.workspace)
    tampered_workspace["hnsbi"]["reference_manifest"] = str(
        alternate_reference_manifest.resolve()
    )
    tampered_path = export.path.with_name("cross-bound-workspace.json")
    tampered_path.write_text(
        json.dumps(tampered_workspace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different reference-flow manifest"):
        load_workspace_model(tampered_path)

    likelihood = ExtendedUnbinnedLikelihood.from_workspace(export.path)
    assert set(likelihood.fnf_systematics) == {"signal"}
    with pytest.raises(RuntimeError, match="differentiable base_torch_log_prob"):
        likelihood.fnf_systematics["signal"].density.torch_log_prob(
            torch.as_tensor(probe),
            {"shift": 0.0},
        )
    with pytest.raises(ValueError, match="Explicit FNF runtime overrides"):
        ExtendedUnbinnedLikelihood.from_workspace(
            export.path,
            fnf_systematics=fnf_systematics,
        )

    generator = ToyGenerator.from_workspace(export.path)
    assert generator.reference is not None
    assert set(generator.ratios) == {"signal", "background"}
    assert generator.component_samplers == {}
    toy = generator.generate({"mu": 1.0, "shift": 0.0}, seed=3)
    assert toy.observed_count >= 0
    diagnostics = toy.events.metadata["component_sampling_diagnostics"]
    assert diagnostics["signal"]["method"] == "reference_importance_resampling"
    with pytest.raises(ValueError, match="Explicit FNF runtime overrides"):
        ToyGenerator.from_workspace(
            export.path,
            fnf_systematics=fnf_systematics,
        )
    with pytest.raises(ValueError, match="custom nominal component samplers"):
        ToyGenerator.from_workspace(
            export.path,
            component_samplers={"signal": Sampler()},
        )
    with pytest.raises(ValueError, match="custom reference or ratio"):
        ToyGenerator.from_workspace(
            export.path,
            reference=Sampler(),
        )


def test_fnf_configuration_copy_is_not_mutated() -> None:
    config = _config()
    original = deepcopy(config)
    Project.load(config)
    assert config == original


def test_fnf_rejects_cross_bound_live_reference_and_ratio_artifacts(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = _project(first_root)
    second = _project(second_root)
    first_reference = first.train_reference(max_events=48, seed=21)
    second_reference = second.train_reference(max_events=48, seed=22)
    first_ratios = first.train_ratios(
        first_reference.training.flow,
        max_events_per_sample=48,
        denominator_events=48,
        normalization_events=64,
        seed=23,
    )
    second_ratios = second.train_ratios(
        second_reference.training.flow,
        max_events_per_sample=48,
        denominator_events=48,
        normalization_events=64,
        seed=24,
    )
    cross_bound_ratios = replace(
        first_ratios,
        training={
            **first_ratios.training,
            "signal": replace(
                first_ratios.training["signal"],
                manifest_path=second_ratios.training["signal"].manifest_path,
            ),
        },
    )
    cross_bound_normalizer = replace(
        first_ratios,
        normalizer=second_ratios.normalizer,
    )

    with pytest.raises(ValueError, match="live reference density disagrees"):
        first.train_fnf_systematics(
            reference=first_reference.training.flow,
            ratios=first_ratios,
            reference_manifest=second_reference.checkpoint_manifest,
            max_events_per_anchor=48,
        )
    with pytest.raises(ValueError, match="live ratio ensemble disagrees"):
        first.train_fnf_systematics(
            reference=first_reference.training.flow,
            ratios=cross_bound_ratios,
            reference_manifest=first_reference.checkpoint_manifest,
            max_events_per_anchor=48,
        )
    with pytest.raises(ValueError, match="live ratio normalizer disagrees"):
        first.train_fnf_systematics(
            reference=first_reference.training.flow,
            ratios=cross_bound_normalizer,
            reference_manifest=first_reference.checkpoint_manifest,
            max_events_per_anchor=48,
        )

    first.train_fnf_systematics(
        reference=first_reference.training.flow,
        ratios=first_ratios,
        reference_manifest=first_reference.checkpoint_manifest,
        max_events_per_anchor=48,
    )
    with pytest.raises(ValueError, match="live reference density disagrees"):
        first.build_fnf_runtime_systematics(
            reference_density=first_reference.training.flow,
            ratios=first_ratios,
            reference_manifest=second_reference.checkpoint_manifest,
        )
    with pytest.raises(ValueError, match="live ratio ensemble disagrees"):
        first.build_fnf_runtime_systematics(
            reference_density=first_reference.training.flow,
            ratios=cross_bound_ratios,
            reference_manifest=first_reference.checkpoint_manifest,
        )
    with pytest.raises(ValueError, match="live ratio normalizer disagrees"):
        first.build_fnf_runtime_systematics(
            reference_density=first_reference.training.flow,
            ratios=cross_bound_normalizer,
            reference_manifest=first_reference.checkpoint_manifest,
        )


def test_project_generation_helpers_never_ignore_configured_fnf(tmp_path) -> None:
    project = _project(tmp_path)

    class Sampler:
        def sample(self, n, *, rng=None):
            return (rng or np.random.default_rng()).normal(size=(int(n), 1))

    with pytest.raises(ValueError, match="would ignore configured FNF"):
        project.asimov_builder(
            reference=Sampler(),
            ratios={
                "signal": lambda values: np.ones(len(values)),
                "background": lambda values: np.ones(len(values)),
            },
        )
    with pytest.raises(ValueError, match="would ignore configured FNF"):
        project.toy_generator(
            component_samplers={
                "signal": Sampler(),
                "background": Sampler(),
            }
        )
