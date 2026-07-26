from __future__ import annotations

import importlib.util
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import hnsbi.integrations.nsbi_common_utils as nsbi_integration
from hnsbi.artifacts import ArtifactManifest, write_artifact_manifest
from hnsbi.integrations.nsbi_common_utils import NsbiCommonUtilsBackend
from hnsbi.integrations.nsbi_inference import (
    NsbiCommonUtilsInference,
    load_upstream_workspace,
)
from hnsbi.onnx import (
    OnnxRunner,
    OptionalDependencyError,
    convert_joblib_scaler_to_onnx,
)
from hnsbi.ratios import OnnxRatioMember, RatioTrainingConfig


class _FakeCalibrator:
    def cali_pred(self, values):
        return values


def _portable_member_manifest(
    tmp_path,
    *,
    use_log_loss,
    calibration,
    member_index=0,
    numerator_name="signal",
    denominator_name="reference",
    features=("x", "y"),
    config=None,
    manifest_name="member.manifest.json",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    classifier = tmp_path / f"portable-classifier-{member_index}.ort"
    scaler = tmp_path / f"portable-transform-{member_index}.ort"
    classifier.write_bytes(b"classifier")
    scaler.write_bytes(b"scaler")
    ratio_config = (
        asdict(
            RatioTrainingConfig(
                ensemble_size=1,
                use_log_loss=use_log_loss,
                calibration=calibration,
                run_diagnostics=False,
            )
        )
        if config is None
        else dict(config)
    )
    files = {
        "classifier-onnx": classifier,
        "scaler-onnx": scaler,
    }
    if calibration:
        calibrator = tmp_path / f"portable-calibrator-{member_index}.pickle"
        calibrator.write_bytes(pickle.dumps(_FakeCalibrator()))
        files["calibrator"] = calibrator
    return write_artifact_manifest(
        tmp_path / manifest_name,
        artifact_type="nsbi-common-utils-ratio-member",
        files=files,
        metadata={
            "backend": "nsbi_common_utils",
            "config": ratio_config,
            "denominator_name": denominator_name,
            "features": list(features),
            "member_index": member_index,
            "numerator_name": numerator_name,
        },
    )


def _portable_ensemble(
    root,
    *,
    ensemble_size=2,
    numerator_names=None,
    member_configs=None,
    member_features=None,
):
    root.mkdir(parents=True, exist_ok=True)
    config = asdict(
        RatioTrainingConfig(
            ensemble_size=ensemble_size,
            use_log_loss=True,
            run_diagnostics=False,
        )
    )
    manifests = {}
    for index in range(ensemble_size):
        directory = root / "objects" / f"portable-{index}"
        member_manifest = _portable_member_manifest(
            directory,
            use_log_loss=True,
            calibration=False,
            member_index=index,
            numerator_name=(
                "signal" if numerator_names is None else numerator_names[index]
            ),
            features=(
                ("x", "y") if member_features is None else member_features[index]
            ),
            config=(config if member_configs is None else member_configs[index]),
            manifest_name=f"member-record-{index}.json",
        )
        manifests[f"member-{index:03d}-member-manifest"] = member_manifest
    return write_artifact_manifest(
        root / "ratio_ensemble.manifest.json",
        artifact_type="density-ratio-ensemble",
        files=manifests,
        metadata={
            "backend": "nsbi_common_utils",
            "config": config,
            "denominator_name": "reference",
            "ensemble_reduction": "arithmetic-mean-of-ratios",
            "features": ["x", "y"],
            "numerator_name": "signal",
        },
    )


def test_nsbi_backend_reports_availability_as_boolean():
    assert isinstance(NsbiCommonUtilsBackend.available(), bool)


def test_ratio_loader_derives_output_semantics_from_manifest(tmp_path, monkeypatch):
    _portable_member_manifest(tmp_path, use_log_loss=True, calibration=False)
    monkeypatch.setattr(
        nsbi_integration, "require_optional", lambda *args, **kwargs: object()
    )
    member = NsbiCommonUtilsBackend.load_member(
        tmp_path, 0, expected_features=["x", "y"]
    )
    assert member.use_log_loss is True
    assert member.model_path.name == "portable-classifier-0.ort"
    assert member.scaler_path.name == "portable-transform-0.ort"
    with pytest.raises(ValueError, match="conflicts"):
        NsbiCommonUtilsBackend.load_member(tmp_path, 0, use_log_loss=False)
    with pytest.raises(ValueError, match="feature order mismatch"):
        NsbiCommonUtilsBackend.load_member(tmp_path, 0, expected_features=["y", "x"])


def test_calibrated_pickle_requires_explicit_unsafe_opt_in(tmp_path, monkeypatch):
    _portable_member_manifest(tmp_path, use_log_loss=False, calibration=True)
    monkeypatch.setattr(
        nsbi_integration, "require_optional", lambda *args, **kwargs: object()
    )
    with pytest.raises(ValueError, match="allow_unsafe_pickle"):
        NsbiCommonUtilsBackend.load_member(tmp_path, 0)


def test_calibrated_pickle_is_loaded_exactly_once(tmp_path, monkeypatch):
    _portable_member_manifest(tmp_path, use_log_loss=False, calibration=True)
    monkeypatch.setattr(
        nsbi_integration, "require_optional", lambda *args, **kwargs: object()
    )
    original_load = pickle.load
    calls = 0

    def counting_load(stream):
        nonlocal calls
        calls += 1
        return original_load(stream)

    monkeypatch.setattr(nsbi_integration.pickle, "load", counting_load)
    member = NsbiCommonUtilsBackend.load_member(tmp_path, 0, allow_unsafe_pickle=True)

    assert calls == 1
    assert member.calibrator is not None


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload: payload.update({"artifact_type": "wrong-type"}),
            "artifact type",
        ),
        (
            lambda payload: payload["metadata"].update({"member_index": 9}),
            "member index mismatch",
        ),
        (
            lambda payload: payload["metadata"]["config"].pop("epochs"),
            "config fields",
        ),
        (
            lambda payload: payload["metadata"].update({"features": ["x", "x"]}),
            "feature signature",
        ),
    ],
)
def test_member_loader_rejects_tampered_semantic_contract(
    tmp_path, monkeypatch, mutation, error
):
    manifest = _portable_member_manifest(
        tmp_path, use_log_loss=False, calibration=False
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        nsbi_integration, "require_optional", lambda *args, **kwargs: object()
    )

    with pytest.raises(ValueError, match=error):
        NsbiCommonUtilsBackend.load_member(tmp_path, 0)


def test_member_loader_rejects_duplicate_model_roles(tmp_path, monkeypatch):
    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    scaler = tmp_path / "scaler.onnx"
    for path in (first, second, scaler):
        path.write_bytes(path.name.encode())
    config = asdict(RatioTrainingConfig(ensemble_size=1, run_diagnostics=False))
    manifest = ArtifactManifest(
        artifact_type="nsbi-common-utils-ratio-member",
        metadata={
            "backend": "nsbi_common_utils",
            "config": config,
            "denominator_name": "reference",
            "features": ["x"],
            "member_index": 0,
            "numerator_name": "signal",
        },
    )
    manifest.add_file(first, root=tmp_path, kind="classifier-onnx")
    manifest.add_file(second, root=tmp_path, kind="classifier-onnx")
    manifest.add_file(scaler, root=tmp_path, kind="scaler-onnx")
    manifest.write(tmp_path / "member.manifest.json")
    monkeypatch.setattr(
        nsbi_integration, "require_optional", lambda *args, **kwargs: object()
    )

    with pytest.raises(ValueError, match="repeats artifact role"):
        NsbiCommonUtilsBackend.load_member(tmp_path, 0)


def test_ensemble_loader_derives_members_and_rejects_size_conflict(
    tmp_path, monkeypatch
):
    _portable_ensemble(tmp_path)
    monkeypatch.setattr(
        nsbi_integration, "require_optional", lambda *args, **kwargs: object()
    )

    inferred = NsbiCommonUtilsBackend.load_ensemble(tmp_path)
    explicit = NsbiCommonUtilsBackend.load_ensemble(tmp_path, 2)

    assert [member.model_path.name for member in inferred.members] == [
        "portable-classifier-0.ort",
        "portable-classifier-1.ort",
    ]
    assert len(explicit.members) == 2
    with pytest.raises(ValueError, match="ensemble_size conflicts"):
        NsbiCommonUtilsBackend.load_ensemble(tmp_path, 1)


def test_ensemble_loader_binds_member_config_and_numerator(tmp_path, monkeypatch):
    root_config = asdict(
        RatioTrainingConfig(
            ensemble_size=2,
            use_log_loss=True,
            run_diagnostics=False,
        )
    )
    wrong_config = dict(root_config)
    wrong_config["epochs"] += 1
    _portable_ensemble(
        tmp_path / "config",
        member_configs=[root_config, wrong_config],
    )
    _portable_ensemble(
        tmp_path / "name",
        numerator_names=["signal", "different-signal"],
    )
    monkeypatch.setattr(
        nsbi_integration, "require_optional", lambda *args, **kwargs: object()
    )

    with pytest.raises(ValueError, match="config conflicts"):
        NsbiCommonUtilsBackend.load_ensemble(tmp_path / "config")
    with pytest.raises(ValueError, match="numerator name conflicts"):
        NsbiCommonUtilsBackend.load_ensemble(tmp_path / "name")


def test_ensemble_loader_rejects_reduction_and_feature_order_tampering(
    tmp_path, monkeypatch
):
    reduction = _portable_ensemble(tmp_path / "reduction")
    payload = json.loads(reduction.read_text(encoding="utf-8"))
    payload["metadata"]["ensemble_reduction"] = "mean-of-log-ratios"
    reduction.write_text(json.dumps(payload), encoding="utf-8")
    _portable_ensemble(tmp_path / "features")
    monkeypatch.setattr(
        nsbi_integration, "require_optional", lambda *args, **kwargs: object()
    )

    with pytest.raises(ValueError, match="ensemble reduction"):
        NsbiCommonUtilsBackend.load_ensemble(tmp_path / "reduction")
    with pytest.raises(ValueError, match="feature order mismatch"):
        NsbiCommonUtilsBackend.load_ensemble(
            tmp_path / "features", expected_features=["y", "x"]
        )


class _IdentityRunner:
    def run(self, values):
        return {"output": np.asarray(values)}


class _FixedRunner:
    def __init__(self, output):
        self.output = np.asarray(output)

    def run(self, values):
        return {"output": self.output}


def _onnx_member(output, *, use_log_loss=False, calibrator=None):
    member = OnnxRatioMember(
        "scaler.onnx",
        "classifier.onnx",
        use_log_loss=use_log_loss,
        calibrator=calibrator,
    )
    member._scaler = _IdentityRunner()
    member._model = _FixedRunner(output)
    return member


@pytest.mark.parametrize("output", [[-1e-12, 0.5], [0.5, 1.0 + 1e-12]])
def test_probability_member_rejects_out_of_range_outputs(output):
    member = _onnx_member(output)
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        member(np.zeros((2, 1)))


def test_probability_member_rejects_nonfinite_and_clips_only_boundaries():
    with pytest.raises(ValueError, match="non-finite raw output"):
        _onnx_member([np.nan, 0.5])(np.zeros((2, 1)))

    member = _onnx_member([0.0, 1.0])
    assert np.allclose(
        member.score(np.zeros((2, 1))),
        [member.score_clip, 1.0 - member.score_clip],
    )


def test_calibrator_probability_is_validated_before_boundary_clip():
    member = _onnx_member(
        [0.25, 0.75],
        calibrator=lambda _: np.array([0.5, 1.01]),
    )
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        member(np.zeros((2, 1)))


def test_log_ratio_member_allows_unbounded_finite_outputs():
    member = _onnx_member([-100.0, 100.0], use_log_loss=True)
    ratios = member(np.zeros((2, 1)))

    assert np.isfinite(ratios).all()
    assert np.allclose(np.log(ratios), [-80.0, 80.0])


@pytest.mark.skipif(
    importlib.util.find_spec("nsbi_common_utils") is not None,
    reason="Only exercises the actionable missing-dependency branch",
)
def test_nsbi_backend_has_actionable_optional_dependency_error():
    with pytest.raises(OptionalDependencyError, match=r"hnsbi-toolkit\[lhc\]"):
        NsbiCommonUtilsBackend()._factory()


def test_backend_refuses_upstream_recursive_deletion(tmp_path, monkeypatch):
    class _Pandas:
        @staticmethod
        def DataFrame(values, columns):
            return object()

    monkeypatch.setattr(
        nsbi_integration,
        "require_optional",
        lambda module, **kwargs: _Pandas() if module == "pandas" else object(),
    )
    config = RatioTrainingConfig(
        ensemble_size=1,
        epochs=1,
        run_diagnostics=False,
        backend_options={"initializer": {"delete_existing_models": True}},
    )

    with pytest.raises(ValueError, match="not supported"):
        NsbiCommonUtilsBackend(trainer_factory=lambda **kwargs: None).train_member(
            numerator_values=np.zeros((2, 1), dtype=np.float32),
            denominator_values=np.ones((2, 1), dtype=np.float32),
            numerator_weights=np.full(2, 0.5),
            denominator_weights=np.full(2, 0.5),
            features=("x",),
            output_directory=tmp_path,
            member_index=0,
            numerator_name="signal",
            denominator_name="reference",
            config=config,
        )


def test_joblib_scaler_conversion_denies_pickle_by_default(tmp_path):
    with pytest.raises(ValueError, match="Refusing to load a pickle"):
        convert_joblib_scaler_to_onnx(
            tmp_path / "untrusted.joblib",
            tmp_path / "scaler.onnx",
            n_features=2,
        )


@pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("joblib", "onnxruntime", "skl2onnx", "sklearn")
    ),
    reason="Scaler conversion uses optional LHC dependencies",
)
def test_upstream_joblib_scaler_converts_to_onnx(tmp_path):
    import joblib
    from sklearn.preprocessing import StandardScaler

    values = np.array([[0.0, 1.0], [2.0, 5.0], [4.0, 9.0]], dtype=np.float32)
    scaler = StandardScaler().fit(values)
    native_path = tmp_path / "scaler.bin"
    onnx_path = tmp_path / "scaler.onnx"
    joblib.dump(scaler, native_path)

    model_path, manifest_path = convert_joblib_scaler_to_onnx(
        native_path, onnx_path, allow_unsafe_pickle=True
    )
    portable = next(iter(OnnxRunner(model_path).run(values).values()))

    assert manifest_path.is_file()
    assert np.allclose(portable, scaler.transform(values), atol=1e-6)


@pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in (
            "joblib",
            "onnxruntime",
            "pandas",
            "skl2onnx",
            "sklearn",
        )
    ),
    reason="ColumnTransformer conversion uses optional LHC dependencies",
)
def test_named_upstream_column_transformer_has_one_onnx_input(tmp_path):
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import MinMaxScaler

    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [3.0, 4.0, 7.0],
            "passthrough": [5.0, 6.0, 9.0],
        }
    )
    scaler = ColumnTransformer(
        [("scaler", MinMaxScaler(feature_range=(-1.5, 1.5)), ["x", "y"])],
        remainder="passthrough",
    ).fit(frame)
    native_path = tmp_path / "column_scaler.bin"
    onnx_path = tmp_path / "column_scaler.onnx"
    joblib.dump(scaler, native_path)

    model_path, _ = convert_joblib_scaler_to_onnx(
        native_path,
        onnx_path,
        feature_names=tuple(frame.columns),
        allow_unsafe_pickle=True,
    )
    runner = OnnxRunner(model_path)
    portable = next(iter(runner.run(frame.to_numpy(dtype=np.float32)).values()))

    assert runner.input_names == ("features",)
    assert np.allclose(portable, scaler.transform(frame), atol=1e-6)


class _FakeModel:
    num_unconstrained_param = 1

    def __init__(self, *, workspace, measurement_to_fit):
        self.workspace = workspace
        self.measurement_to_fit = measurement_to_fit
        self.model = lambda values: np.square(values).sum()
        self.model_grad = lambda values: 2 * np.asarray(values)

    def get_model_parameters(self):
        return ["mu", "alpha"], np.array([1.0, 0.0])


class _FakeInference:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.list_parameters = kwargs["list_parameters"]
        self.pulls_global_fit = None
        self.last_scan = None

    def perform_fit(self, *, fit_strategy, freeze_params):
        self.pulls_global_fit = np.array([1.1, -0.2])

    def perform_profile_scan(self, **kwargs):
        self.last_scan = kwargs
        points = np.linspace(*kwargs["bound_range"], kwargs["size"])
        return points, np.square(points - 1.0)


def test_workspace_runtime_resolves_arrays_and_delegates(tmp_path):
    array_directory = tmp_path / "arrays"
    array_directory.mkdir()
    for name in ("weights", "signal", "up", "down"):
        np.save(array_directory / f"{name}.npy", np.ones(3))
    workspace_path = tmp_path / "workspace.json"
    workspace_path.write_text(
        """{
          "measurements": [{"name": "meas", "config": {}}],
          "channels": [{
            "weights": "arrays/weights.npy",
            "samples": [{
              "ratios": "arrays/signal.npy",
              "modifiers": [{"data": {
                "hi_ratio": "arrays/up.npy",
                "lo_ratio": "arrays/down.npy"
              }}]
            }]
          }]
        }""",
        encoding="utf-8",
    )

    resolved = load_upstream_workspace(workspace_path)
    assert Path(resolved["channels"][0]["weights"]).is_absolute()
    runtime = NsbiCommonUtilsInference.from_workspace(
        workspace_path,
        model_factory=_FakeModel,
        inference_factory=_FakeInference,
    )

    assert np.allclose(runtime.perform_fit(), [1.1, -0.2])
    points, values = runtime.profile_scan("mu", bound_range=(-1.0, 2.0), size=7)
    assert len(points) == len(values) == 7
    assert runtime.engine.last_scan["doStatOnly"] is False


def test_workspace_runtime_rejects_nonlinear_hnsbi_formulas(tmp_path):
    array_path = tmp_path / "weights.npy"
    np.save(array_path, np.ones(2))
    workspace = {
        "measurements": [{"name": "meas", "config": {}}],
        "channels": [{"weights": str(array_path), "samples": []}],
        "hnsbi": {"upstream_compatible": False},
    }
    with pytest.raises(ValueError, match="nonlinear hNSBI"):
        NsbiCommonUtilsInference.from_workspace(
            workspace,
            model_factory=_FakeModel,
            inference_factory=_FakeInference,
        )
