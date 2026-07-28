from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from hnsbi import Project
from hnsbi.bayes.native_backend import NativeDualBackend
from hnsbi.config import ConfigError, ToolkitConfig

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "toolkit.schema.json"
CONFIG_DIR = ROOT / "examples" / "configs"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _validator():
    schema = _schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def _example(filename: str) -> dict:
    return json.loads((CONFIG_DIR / filename).read_text(encoding="utf-8"))


def _schema_errors(value: dict) -> list:
    return list(_validator().iter_errors(value))


@pytest.mark.parametrize(
    "filename",
    [
        "frequentist_complete.json",
        "dual_complete.json",
    ],
)
def test_complete_example_satisfies_schema(filename: str) -> None:
    value = _example(filename)
    errors = sorted(_validator().iter_errors(value), key=lambda error: list(error.path))
    assert not errors, "\n".join(
        f"{'.'.join(map(str, error.absolute_path))}: {error.message}"
        for error in errors
    )


@pytest.mark.parametrize(
    "filename",
    [
        "frequentist_complete.json",
        "dual_complete.json",
    ],
)
def test_complete_example_loads_through_public_api(filename: str) -> None:
    config = ToolkitConfig.load(CONFIG_DIR / filename)
    assert config.features
    assert config.output_dir.name


def test_schema_requires_a_workflow() -> None:
    value = {
        "schema_version": "2.0",
        "features": ["x"],
    }
    errors = list(_validator().iter_errors(value))
    assert errors
    with pytest.raises(ConfigError, match="At least one"):
        ToolkitConfig.load(value)


def test_schema_accepts_source_weight_sum_nominal_yield() -> None:
    value = _example("frequentist_complete.json")
    value["frequentist"]["samples"][0]["nominal_yield"] = {
        "kind": "source_weight_sum"
    }
    assert not _schema_errors(value)
    loaded = ToolkitConfig.load(value)
    assert loaded.frequentist["samples"][0]["nominal_yield"] == {
        "kind": "source_weight_sum"
    }


@pytest.mark.parametrize(
    "nominal_yield",
    [
        {"kind": "weights"},
        {"kind": "source_weight_sum", "column": "weight"},
        {},
        "source_weight_sum",
        True,
        -1.0,
    ],
)
def test_schema_rejects_invalid_source_weight_sum_nominal_yield(
    nominal_yield,
) -> None:
    value = _example("frequentist_complete.json")
    value["frequentist"]["samples"][0]["nominal_yield"] = nominal_yield
    assert _schema_errors(value)
    with pytest.raises(ConfigError, match="nominal_yield"):
        ToolkitConfig.load(value, validate_schema=False)


@pytest.mark.parametrize("nominal_yield", [float("nan"), float("inf")])
def test_manual_validation_rejects_non_finite_nominal_yield(
    nominal_yield,
) -> None:
    value = _example("frequentist_complete.json")
    value["frequentist"]["samples"][0]["nominal_yield"] = nominal_yield
    with pytest.raises(ConfigError, match="nominal_yield"):
        ToolkitConfig.load(value, validate_schema=False)


def test_schema_rejects_invalid_nis_epsilon() -> None:
    value = _example("frequentist_complete.json")
    value["frequentist"]["nis"]["epsilon"] = 0.0
    errors = list(_validator().iter_errors(value))
    assert any(list(error.absolute_path)[-1:] == ["epsilon"] for error in errors)
    with pytest.raises(ConfigError):
        ToolkitConfig.load(value)


def test_manual_validation_rejects_duplicate_features() -> None:
    value = _example("dual_complete.json")
    value["features"] = ["x1", "x1"]
    with pytest.raises(ConfigError, match="unique"):
        ToolkitConfig.load(value)


def test_complete_examples_translate_to_runtime_configuration() -> None:
    frequentist = Project.load(CONFIG_DIR / "frequentist_complete.json")
    flow, training = frequentist.flow_configs()
    ratio = frequentist.ratio_config()
    assert flow.flow_type == "quadratic-spline"
    assert training.validation_fraction == pytest.approx(0.2)
    assert ratio.ensemble_size == 4
    assert frequentist.intensity_model().component_names == (
        "signal",
        "background",
    )

    dual = ToolkitConfig.load(CONFIG_DIR / "dual_complete.json")
    backend = NativeDualBackend.from_config(
        dual.bayesian,
        observation_features=dual.features,
    )
    assert backend.posterior_flow.flow.flow_type == "quadratic-spline"
    assert backend.posterior_ratio.onnx_opset == 17
    assert backend.likelihood_ratio.onnx_opset == 17
    assert backend.normalizer.onnx_opset == 17


@pytest.mark.parametrize(
    ("filename", "mutate"),
    [
        (
            "frequentist_complete.json",
            lambda value: value.update({"featurez": ["x"]}),
        ),
        (
            "frequentist_complete.json",
            lambda value: value["frequentist"]["flow"].update({"unknown_setting": 1}),
        ),
        (
            "dual_complete.json",
            lambda value: value["bayesian"]["posterior_ratio"].update(
                {"diagnostics": {"overtraining": True}}
            ),
        ),
        (
            "dual_complete.json",
            lambda value: value["bayesian"]["normalizer"]["training"].update(
                {"max_events": 100}
            ),
        ),
    ],
)
def test_schema_rejects_unknown_or_ignored_fields(
    filename: str,
    mutate,
) -> None:
    value = _example(filename)
    mutate(value)
    assert _schema_errors(value)
    with pytest.raises(ConfigError):
        ToolkitConfig.load(value)


@pytest.mark.parametrize(
    ("filename", "mutate"),
    [
        (
            "frequentist_complete.json",
            lambda value: value["frequentist"]["reference"].update(
                {"selection": "x1 > 0"}
            ),
        ),
        (
            "frequentist_complete.json",
            lambda value: value["frequentist"]["toys"].update(
                {"method": "reference_bootstrap"}
            ),
        ),
        (
            "frequentist_complete.json",
            lambda value: value["frequentist"]["ratios"]["diagnostics"].update(
                {"tail_summary": True}
            ),
        ),
        (
            "dual_complete.json",
            lambda value: value["bayesian"].update(
                {"validation": {"conditional_flow_c2st": True}}
            ),
        ),
    ],
)
def test_schema_does_not_advertise_unsupported_workflow_switches(
    filename: str,
    mutate,
) -> None:
    value = _example(filename)
    mutate(value)
    assert _schema_errors(value)


@pytest.mark.parametrize(
    "location",
    [
        "posterior_ratio",
        "likelihood_ratio",
        "normalizer",
    ],
)
def test_dual_onnx_opset_is_configurable_but_requires_17(
    location: str,
) -> None:
    value = _example("dual_complete.json")
    value["bayesian"][location]["onnx_opset"] = 18
    assert not _schema_errors(value)
    ToolkitConfig.load(value)

    value["bayesian"][location]["onnx_opset"] = 16
    assert _schema_errors(value)
    with pytest.raises(ConfigError):
        ToolkitConfig.load(value)


@pytest.mark.parametrize(
    "field",
    ["posterior_flow", "likelihood_flow"],
)
def test_dual_conditional_flows_reject_realnvp(field: str) -> None:
    value = _example("dual_complete.json")
    value["bayesian"][field]["architecture"] = "realnvp"
    assert _schema_errors(value)
    with pytest.raises(ConfigError, match="quadratic_spline"):
        ToolkitConfig.load(value)


def test_mapping_and_json_reject_nonfinite_numbers(tmp_path: Path) -> None:
    value = _example("frequentist_complete.json")
    value["frequentist"]["parameters"][0]["nominal"] = float("nan")
    with pytest.raises(ConfigError, match=r"\$.*nominal must be finite"):
        ToolkitConfig.load(value)

    path = tmp_path / "invalid.json"
    path.write_text(
        '{"schema_version":"2.0","features":["x"],"bayesian":NaN}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not valid JSON"):
        ToolkitConfig.load(path)


def test_json_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"2.0","schema_version":"2.0",'
        '"features":["x"],"bayesian":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not valid JSON"):
        ToolkitConfig.load(path)


def test_yaml_is_the_human_interface_and_json_remains_canonical(
    tmp_path: Path,
) -> None:
    value = _example("dual_complete.json")
    authored = tmp_path / "analysis.yaml"
    authored.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    config = ToolkitConfig.load(authored)
    yaml_roundtrip = config.dump(tmp_path / "roundtrip.yaml")
    json_artifact = config.dump_json(tmp_path / "runtime.json")

    assert ToolkitConfig.load(yaml_roundtrip).raw == value
    assert json.loads(json_artifact.read_text(encoding="utf-8")) == value
    assert not json_artifact.read_text(encoding="utf-8").startswith("---")


def test_yaml_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "schema_version: '2.0'\nschema_version: '2.0'\nfeatures: [x]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="duplicate key"):
        ToolkitConfig.load(duplicate)

    nonfinite = tmp_path / "nonfinite.yaml"
    value = _example("dual_complete.json")
    value["bayesian"]["defensive_epsilon"] = float("inf")
    nonfinite.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ConfigError, match="must be finite"):
        ToolkitConfig.load(nonfinite)


def test_parameter_points_must_have_exact_declared_keys() -> None:
    value = _example("frequentist_complete.json")
    value["frequentist"]["asimov"]["parameter_point"]["typo"] = 1.0
    with pytest.raises(ConfigError, match="exactly"):
        ToolkitConfig.load(value)


def test_exactly_one_poi_is_required_by_schema_and_loader() -> None:
    value = _example("frequentist_complete.json")
    value["frequentist"]["parameters"][1]["role"] = "poi"
    assert _schema_errors(value)
    with pytest.raises(ConfigError, match="Exactly one"):
        ToolkitConfig.load(value)


def test_multiplier_is_validated_during_config_load() -> None:
    value = _example("frequentist_complete.json")
    value["frequentist"]["samples"][0]["multiplier"] = "__import__('os')"
    with pytest.raises(ConfigError, match="Invalid frequentist intensity"):
        ToolkitConfig.load(value)


def test_systematic_variations_cannot_repeat_sample_parameter_pair() -> None:
    value = _example("frequentist_complete.json")
    repeated = deepcopy(value["frequentist"]["systematics"][0]["variations"][0])
    value["frequentist"]["systematics"][0]["variations"].append(repeated)
    with pytest.raises(ConfigError, match="repeats variation sample"):
        ToolkitConfig.load(value)


def test_design_distribution_dimension_matches_theta() -> None:
    value = _example("dual_complete.json")
    value["bayesian"]["design_distributions"]["rho"]["scale"].pop()
    with pytest.raises(ConfigError, match="2 positive numeric values"):
        ToolkitConfig.load(value)


def test_data_role_columns_cannot_overlap_model_features() -> None:
    value = _example("dual_complete.json")
    value["bayesian"]["datasets"]["rho"]["split_column"] = "x1"
    with pytest.raises(ConfigError, match="overlap model features"):
        ToolkitConfig.load(value)


def _frequentist_config() -> dict:
    return json.loads(
        (CONFIG_DIR / "frequentist_complete.json").read_text(encoding="utf-8")
    )


def test_python_dictionary_rejects_nested_nonfinite_number() -> None:
    value = _frequentist_config()
    value["frequentist"]["parameters"][0]["nominal"] = float("nan")
    with pytest.raises(
        ConfigError,
        match=r"\$\.frequentist\.parameters\[0\]\.nominal must be finite",
    ):
        ToolkitConfig.load(value, validate_schema=False)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_configuration_rejects_nonstandard_numeric_constants(
    tmp_path,
    constant: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(_frequentist_config()).replace("611.6", constant, 1),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="not valid JSON"):
        ToolkitConfig.load(path, validate_schema=False)


@pytest.mark.parametrize(
    "location",
    [
        ("asimov", "parameter_point", None),
        ("toys", "parameter_points", 0),
        ("nis", "design_points", 0),
    ],
    ids=["asimov", "toys", "nis"],
)
def test_parameter_points_require_exact_declared_keys(location) -> None:
    value = _frequentist_config()
    section, key, index = location
    point = value["frequentist"][section][key]
    if index is not None:
        point = point[index]
    point.pop("alpha")
    point["undeclared"] = 1.0

    with pytest.raises(ConfigError, match="must contain exactly.*alpha.*mu"):
        ToolkitConfig.load(value, validate_schema=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("poi", "must reference a nuisance parameter"),
        ("undeclared", "undeclared parameter"),
        ("unknown_sample", "unknown samples.*ghost"),
    ],
)
def test_systematic_must_bind_declared_nuisance_and_known_sample(
    mutation: str,
    message: str,
) -> None:
    value = _frequentist_config()
    systematic = value["frequentist"]["systematics"][0]
    if mutation == "poi":
        systematic["parameter"] = "mu"
    elif mutation == "undeclared":
        systematic["parameter"] = "missing"
    else:
        systematic["variations"][0]["sample"] = "ghost"

    with pytest.raises(ConfigError, match=message):
        ToolkitConfig.load(value, validate_schema=False)


def test_systematic_yield_anchors_are_optional_finite_rate_factors() -> None:
    value = _frequentist_config()
    variation = value["frequentist"]["systematics"][0]["variations"][0]
    variation["yield_up"] = 1.25
    variation["yield_down"] = 0.8

    assert not _schema_errors(value)
    loaded = ToolkitConfig.load(value)
    loaded_variation = loaded.frequentist["systematics"][0]["variations"][0]
    assert loaded_variation["yield_up"] == pytest.approx(1.25)
    assert loaded_variation["yield_down"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("interpolation", "anchor", "message"),
    [
        ("linear", -0.1, "finite non-negative"),
        ("linear", "one", "finite non-negative"),
        ("nsbi_code4p", 0.0, "strictly positive"),
    ],
)
def test_systematic_yield_anchor_contract_is_enforced_without_jsonschema(
    interpolation: str,
    anchor,
    message: str,
) -> None:
    value = _frequentist_config()
    value["frequentist"]["systematics"][0]["interpolation"] = interpolation
    value["frequentist"]["systematics"][0]["variations"][0]["yield_up"] = anchor

    with pytest.raises(ConfigError, match=message):
        ToolkitConfig.load(value, validate_schema=False)


def test_linear_systematic_allows_zero_yield_anchor() -> None:
    value = _frequentist_config()
    value["frequentist"]["systematics"][0]["interpolation"] = "linear"
    value["frequentist"]["systematics"][0]["variations"][0]["yield_down"] = 0.0

    assert not _schema_errors(value)
    ToolkitConfig.load(value)


def test_schema_rejects_zero_code4p_yield_anchor() -> None:
    value = _frequentist_config()
    value["frequentist"]["systematics"][0]["variations"][0]["yield_up"] = 0.0

    assert _schema_errors(value)
