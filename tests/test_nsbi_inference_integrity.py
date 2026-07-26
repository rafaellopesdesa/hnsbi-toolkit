from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hnsbi.artifacts import ArtifactIntegrityError, write_artifact_manifest
from hnsbi.integrations.nsbi_inference import (
    NsbiCommonUtilsInference,
    load_upstream_workspace,
)
from hnsbi.intensity import Component, IntensityModel, Parameter


class _FakeModel:
    num_unconstrained_param = 0

    def __init__(self, *, workspace, measurement_to_fit):
        self.workspace = workspace
        self.measurement_to_fit = measurement_to_fit
        self.model = lambda values: np.square(values).sum()

    def get_model_parameters(self):
        return ["mu"], np.asarray([1.0])


class _FakeInference:
    def __init__(self, **kwargs):
        self.list_parameters = kwargs["list_parameters"]


def _hnsbi_workspace(tmp_path: Path, *, systematic: bool = False) -> Path:
    arrays = tmp_path / "arrays"
    arrays.mkdir(parents=True)
    weights = arrays / "asimov_weights.npy"
    reference_weights = arrays / "reference_weights.npy"
    background = arrays / "ratio_background.npy"
    signal = arrays / "ratio_signal.npy"
    np.save(weights, np.asarray([4.0, 5.0, 6.0]))
    np.save(reference_weights, np.full(3, 1.0 / 3.0))
    np.save(background, np.ones(3))
    np.save(signal, np.asarray([0.5, 1.0, 1.5]))
    metadata = arrays / "asimov_metadata.json"

    intensity = IntensityModel(
        [
            Component("background", 10.0, "1"),
            Component("signal", 5.0, "mu"),
        ],
        [Parameter("mu", 1.0, (0.0, 3.0))],
    )

    modifiers: list[dict[str, object]] = [
        {"name": "mu", "data": None, "type": "normfactor"}
    ]
    parameters: list[dict[str, object]] = [
        {"name": "mu", "inits": [1.0], "bounds": [[0.0, 3.0]]}
    ]
    if systematic:
        parameters.append(
            {
                "name": "alpha",
                "inits": [0.0],
                "hnsbi_constraint": {
                    "kind": "normal",
                    "mean": 0.0,
                    "sigma": 1.0,
                },
            }
        )
        intensity = IntensityModel(
            [
                Component("background", 10.0, "1"),
                Component("signal", 5.0, "mu"),
            ],
            [
                Parameter("mu", 1.0, (0.0, 3.0)),
                Parameter("alpha", 0.0, constrained=True),
            ],
        )
        up = arrays / "signal_alpha_up.npy"
        down = arrays / "signal_alpha_down.npy"
        np.save(up, np.asarray([0.8, 1.0, 1.2]))
        np.save(down, np.asarray([1.2, 1.0, 0.8]))
        write_artifact_manifest(
            arrays / "signal_alpha.manifest.json",
            artifact_type="systematic-anchor",
            files={"up-ratio": up, "down-ratio": down},
            metadata={
                "component": "signal",
                "interpolation": "nsbi_code4p",
                "parameter": "alpha",
                "rows": 3,
                "yield_down": 0.9,
                "yield_up": 1.1,
            },
        )
        modifiers.append(
            {
                "name": "alpha",
                "type": "normplusshape",
                "data": {
                    "hi_data": [1.1],
                    "lo_data": [0.9],
                    "hi_ratio": "arrays/signal_alpha_up.npy",
                    "lo_ratio": "arrays/signal_alpha_down.npy",
                },
                "hnsbi": {
                    "interpolation": "nsbi_code4p",
                    "manifest": "arrays/signal_alpha.manifest.json",
                    "shape_normalization": "reference_support",
                },
            }
        )

    point = {"mu": 1.0, **({"alpha": 0.0} if systematic else {})}
    metadata.write_text(
        json.dumps(
            {
                "kind": "test",
                "intensity_fingerprint": intensity.fingerprint,
                "point": point,
                "ratio_normalizers": {
                    "background": 1.0,
                    "signal": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    write_artifact_manifest(
        arrays / "asimov_arrays.manifest.json",
        artifact_type="asimov-array-bundle",
        files={
            "event-weights": weights,
            "reference-integration-weights": reference_weights,
            "metadata": metadata,
            "process-ratio:background": background,
            "process-ratio:signal": signal,
        },
        metadata={
            "features": ["x"],
            "intensity_fingerprint": intensity.fingerprint,
            "rows": 3,
            "samples": ["background", "signal"],
        },
    )

    workspace = {
        "version": "1.0.0",
        "channels": [
            {
                "name": "SR",
                "type": "unbinned",
                "weights": "arrays/asimov_weights.npy",
                "samples": [
                    {
                        "name": "background",
                        "data": [10.0],
                        "ratios": "arrays/ratio_background.npy",
                        "modifiers": [],
                        "hnsbi": {
                            "multiplier": "1",
                            "ratio_normalizer": 1.0,
                        },
                    },
                    {
                        "name": "signal",
                        "data": [5.0],
                        "ratios": "arrays/ratio_signal.npy",
                        "modifiers": modifiers,
                        "hnsbi": {
                            "multiplier": "mu",
                            "ratio_normalizer": 1.0,
                        },
                    },
                ],
            }
        ],
        "measurements": [
            {
                "name": "measurement",
                "config": {"parameters": parameters, "poi": "mu"},
            }
        ],
        "hnsbi": {
            "schema_version": "1.0",
            "upstream_compatible": not systematic,
            "sample_multipliers": {"background": "1", "signal": "mu"},
            "intensity_fingerprint": intensity.fingerprint,
            "intensity_specification": intensity.specification(),
            "ratio_normalization": {"background": 1.0, "signal": 1.0},
            "asimov_point": point,
            "asimov_raw_count": 3,
            "features": ["x"],
            "array_manifest": "arrays/asimov_arrays.manifest.json",
            "reference_weights": "arrays/reference_weights.npy",
            "parameter_nominals": {
                "mu": 1.0,
                **({"alpha": 0.0} if systematic else {}),
            },
            "upstream_incompatibilities": {},
        },
    }
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps(workspace), encoding="utf-8")
    return path


def _mutate_workspace(path: Path, mutation) -> None:
    workspace = json.loads(path.read_text(encoding="utf-8"))
    mutation(workspace)
    path.write_text(json.dumps(workspace), encoding="utf-8")


def test_hnsbi_workspace_is_verified_before_upstream_construction(
    tmp_path,
) -> None:
    path = _hnsbi_workspace(tmp_path)
    runtime = NsbiCommonUtilsInference.from_workspace(
        path,
        model_factory=_FakeModel,
        inference_factory=_FakeInference,
    )
    assert runtime.measurement_to_fit == "measurement"
    assert Path(runtime.workspace["channels"][0]["weights"]).is_absolute()


def test_legacy_workspace_without_hnsbi_extension_remains_loadable(
    tmp_path,
) -> None:
    weights = tmp_path / "weights.npy"
    np.save(weights, np.ones(2))
    workspace = {
        "measurements": [{"name": "measurement", "config": {}}],
        "channels": [{"weights": "weights.npy", "samples": []}],
    }
    resolved = load_upstream_workspace(workspace, base_directory=tmp_path)
    assert resolved["channels"][0]["weights"] == str(weights.resolve())


def test_hnsbi_manifest_role_swap_is_rejected(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path)

    def swap(workspace):
        samples = workspace["channels"][0]["samples"]
        samples[0]["ratios"], samples[1]["ratios"] = (
            samples[1]["ratios"],
            samples[0]["ratios"],
        )

    _mutate_workspace(path, swap)
    with pytest.raises(ValueError, match="process-ratio:background.*manifest"):
        load_upstream_workspace(path)


def test_hnsbi_array_checksum_tampering_is_rejected(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path)
    np.save(tmp_path / "arrays" / "ratio_signal.npy", np.ones(3))
    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        load_upstream_workspace(path)


def test_hnsbi_array_manifest_type_is_rejected(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path)
    manifest_path = tmp_path / "arrays" / "asimov_arrays.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_type"] = "density-ratio-ensemble"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected artifact type"):
        load_upstream_workspace(path)


def test_hnsbi_workspace_intensity_tampering_is_rejected(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path)
    _mutate_workspace(
        path,
        lambda workspace: workspace["channels"][0]["samples"][1].update(
            {"data": [500.0]}
        ),
    )
    with pytest.raises(ValueError, match="intensity specification"):
        load_upstream_workspace(path)


def test_hnsbi_asimov_point_must_match_upstream_initial_values(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path)
    _mutate_workspace(
        path,
        lambda workspace: workspace["measurements"][0]["config"]["parameters"][
            0
        ].update({"inits": [2.0]}),
    )
    with pytest.raises(ValueError, match="Asimov generating point"):
        load_upstream_workspace(path)


def test_systematic_manifest_metadata_tampering_is_rejected(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path, systematic=True)
    manifest_path = tmp_path / "arrays" / "signal_alpha.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["parameter"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata 'parameter'"):
        load_upstream_workspace(path)


def test_systematic_manifest_type_is_rejected(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path, systematic=True)
    manifest_path = tmp_path / "arrays" / "signal_alpha.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_type"] = "asimov-array-bundle"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected artifact type"):
        load_upstream_workspace(path)


def test_systematic_manifest_role_swap_is_rejected(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path, systematic=True)

    def swap(workspace):
        data = workspace["channels"][0]["samples"][1]["modifiers"][1]["data"]
        data["hi_ratio"], data["lo_ratio"] = (
            data["lo_ratio"],
            data["hi_ratio"],
        )

    _mutate_workspace(path, swap)
    with pytest.raises(ValueError, match="hi_ratio.*manifest role"):
        load_upstream_workspace(path)


def test_systematic_manifest_checksum_tampering_is_rejected(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path, systematic=True)
    np.save(tmp_path / "arrays" / "signal_alpha_up.npy", np.ones(3))
    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        load_upstream_workspace(path)


def test_false_upstream_compatibility_override_is_rejected(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path, systematic=True)
    _mutate_workspace(
        path,
        lambda workspace: workspace["hnsbi"].update({"upstream_compatible": True}),
    )
    with pytest.raises(ValueError, match="normplusshape systematics"):
        load_upstream_workspace(path)


def test_incompatible_workspace_reports_general_and_recorded_reasons(
    tmp_path,
) -> None:
    path = _hnsbi_workspace(tmp_path, systematic=True)
    _mutate_workspace(
        path,
        lambda workspace: workspace["hnsbi"]["upstream_incompatibilities"].update(
            {"nonstandard_constraints": {"alpha": {"mean": 2.0, "sigma": 3.0}}}
        ),
    )
    with pytest.raises(ValueError) as error:
        NsbiCommonUtilsInference.from_workspace(
            path,
            model_factory=_FakeModel,
            inference_factory=_FakeInference,
        )
    message = str(error.value)
    assert "upstream_compatible=false" in message
    assert "reference-normalized normplusshape systematics" in message
    assert "nonstandard_constraints" in message


def test_workspace_mapping_input_is_not_mutated(tmp_path) -> None:
    path = _hnsbi_workspace(tmp_path)
    workspace = json.loads(path.read_text(encoding="utf-8"))
    original = copy.deepcopy(workspace)
    load_upstream_workspace(workspace, base_directory=tmp_path)
    assert workspace == original
