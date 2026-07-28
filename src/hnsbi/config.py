"""YAML-first configuration loading with canonical JSON serialization."""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Configuration does not satisfy the toolkit contract."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant {value!r}.")


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key {key!r}.")
        result[key] = value
    return result


def _load_yaml(text: str, *, path: Path) -> Any:
    """Load strict YAML while rejecting duplicate mapping keys.

    YAML is the human-authored interface.  It is converted immediately to the
    same ordinary Python mapping used by JSON and dictionary callers, so every
    downstream validation and serialization rule is format-independent.
    """

    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "YAML configuration requires PyYAML; reinstall hnsbi-toolkit."
        ) from exc

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: yaml.SafeLoader,
        node: Any,
        deep: bool = False,
    ) -> dict[Any, Any]:
        loader.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in result
            except TypeError as exc:
                raise ConfigError(
                    f"Configuration {path} contains an unhashable mapping key."
                ) from exc
            if duplicate:
                raise ConfigError(
                    f"Configuration {path} contains duplicate key {key!r}."
                )
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        return yaml.load(text, Loader=UniqueKeyLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration {path} is not valid YAML.") from exc


def _require_finite_numbers(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_numbers(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_numbers(item, f"{path}[{index}]")
    elif (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and not math.isfinite(float(value))
    ):
        raise ConfigError(f"{path} must be finite.")


def _load_schema() -> dict[str, Any]:
    """Load the packaged schema, with a source-tree fallback for development."""

    try:
        text = (
            resources.files("hnsbi")
            .joinpath("schemas", "toolkit.schema.json")
            .read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        source_path = (
            Path(__file__).resolve().parents[2] / "schemas" / "toolkit.schema.json"
        )
        if not source_path.is_file():
            raise ConfigError(
                "The packaged toolkit JSON Schema is missing; reinstall hnsbi-toolkit."
            ) from None
        text = source_path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError("The packaged toolkit JSON Schema is invalid.") from exc
    if not isinstance(value, dict):
        raise ConfigError("The packaged toolkit JSON Schema is not an object.")
    return value


def _read_mapping(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return deepcopy(dict(source))
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read configuration {path}.") from exc
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = _load_yaml(text, path=path)
    elif path.suffix.lower() == ".json":
        try:
            value = json.loads(
                text,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ConfigError(f"Configuration {path} is not valid JSON.") from exc
    else:
        raise ConfigError(
            f"Configuration {path} must use a .yaml, .yml, or .json extension."
        )
    if not isinstance(value, dict):
        raise ConfigError("The top-level configuration must be an object.")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError("Top-level configuration keys must be strings.")
    return value


def _manual_validate(value: Mapping[str, Any]) -> None:
    version = value.get("schema_version")
    if version != "2.0":
        raise ConfigError(f"schema_version must be '2.0', received {version!r}.")
    features = value.get("features")
    if not isinstance(features, list) or not features:
        raise ConfigError("features must be a non-empty list.")
    if not all(isinstance(name, str) and name for name in features):
        raise ConfigError("Every feature name must be a non-empty string.")
    if len(set(features)) != len(features):
        raise ConfigError("Feature names must be unique.")
    if "frequentist" not in value and "bayesian" not in value:
        raise ConfigError("At least one of 'frequentist' or 'bayesian' is required.")

    def validate_source_columns(
        specification: Any,
        *,
        model_features: set[str],
        location: str,
    ) -> None:
        if not isinstance(specification, Mapping):
            raise ConfigError(f"{location} must be an object.")
        roles = {
            field: specification[field]
            for field in (
                "weight_column",
                "event_id_column",
                "split_column",
                "group_column",
                "log_density_column",
            )
            if field in specification
        }
        if not all(isinstance(column, str) and column for column in roles.values()):
            raise ConfigError(f"{location} column names must be non-empty strings.")
        role_columns = list(roles.values())
        if len(set(role_columns)) != len(role_columns):
            raise ConfigError(f"{location} assigns one column to multiple data roles.")
        overlap = model_features.intersection(role_columns)
        if overlap:
            raise ConfigError(
                f"{location} data-role columns overlap model features "
                f"{sorted(overlap)}."
            )

    frequentist = value.get("frequentist")
    if frequentist is not None:
        if not isinstance(frequentist, Mapping):
            raise ConfigError("frequentist must be an object.")
        samples = frequentist.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ConfigError("frequentist.samples must be a non-empty list.")
        names = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                raise ConfigError("Every frequentist sample must be an object.")
            name = sample.get("name")
            if not isinstance(name, str) or not name:
                raise ConfigError("Every frequentist sample needs a name.")
            nominal_yield = sample.get("nominal_yield")
            if isinstance(nominal_yield, Mapping):
                if dict(nominal_yield) != {"kind": "source_weight_sum"}:
                    raise ConfigError(
                        f"Frequentist sample {name!r} nominal_yield must be a "
                        "non-negative number or exactly "
                        "{'kind': 'source_weight_sum'}."
                    )
            elif (
                isinstance(nominal_yield, bool)
                or not isinstance(nominal_yield, numbers.Real)
                or not math.isfinite(float(nominal_yield))
                or float(nominal_yield) < 0.0
            ):
                raise ConfigError(
                    f"Frequentist sample {name!r} nominal_yield must be a "
                    "non-negative number or exactly "
                    "{'kind': 'source_weight_sum'}."
                )
            names.append(name)
        if len(names) != len(samples):
            raise ConfigError("Every frequentist sample needs a name.")
        if len(set(names)) != len(names):
            raise ConfigError("Frequentist sample names must be unique.")
        observation_features = set(features)
        validate_source_columns(
            frequentist.get("reference"),
            model_features=observation_features,
            location="frequentist.reference",
        )
        for index, sample in enumerate(samples):
            validate_source_columns(
                sample.get("source"),
                model_features=observation_features,
                location=f"frequentist.samples[{index}].source",
            )
        parameters = frequentist.get("parameters")
        if not isinstance(parameters, list) or not parameters:
            raise ConfigError("frequentist.parameters must be a non-empty list.")
        parameter_names = []
        for parameter in parameters:
            if not isinstance(parameter, Mapping):
                raise ConfigError("Every frequentist parameter must be an object.")
            name = parameter.get("name")
            if not isinstance(name, str) or not name:
                raise ConfigError("Every frequentist parameter needs a name.")
            parameter_names.append(name)
        if len(set(parameter_names)) != len(parameter_names):
            raise ConfigError("Frequentist parameter names must be unique.")
        pois = [
            parameter["name"]
            for parameter in parameters
            if parameter.get("role") == "poi"
        ]
        if len(pois) != 1:
            raise ConfigError("Exactly one frequentist parameter must have role='poi'.")
        inference = frequentist.get("inference")
        if inference is not None:
            if not isinstance(inference, Mapping):
                raise ConfigError("frequentist.inference must be an object.")
            if inference.get("poi") != pois[0]:
                raise ConfigError(
                    "frequentist.inference.poi must match the declared POI "
                    f"{pois[0]!r}."
                )
            projection = inference.get("pyhf_projection")
            if isinstance(projection, Mapping):
                interval = projection.get("range")
                if (
                    not isinstance(interval, list)
                    or len(interval) != 2
                    or float(interval[0]) >= float(interval[1])
                ):
                    raise ConfigError(
                        "frequentist.inference.pyhf_projection.range requires "
                        "two increasing values."
                    )
                scan = projection.get("scan")
                if (
                    not isinstance(scan, list)
                    or len(scan) < 2
                    or any(
                        float(right) <= float(left)
                        for left, right in zip(scan, scan[1:], strict=False)
                    )
                ):
                    raise ConfigError(
                        "frequentist.inference.pyhf_projection.scan must be "
                        "strictly increasing."
                    )
        ratios = frequentist.get("ratios")
        if not isinstance(ratios, Mapping):
            raise ConfigError("frequentist.ratios must be an object.")
        if ratios.get("backend", "native") != "native":
            raise ConfigError("The frequentist ratio backend must be 'native'.")
        if (
            ratios.get("normalization", "independent_reference_mean")
            != "independent_reference_mean"
        ):
            raise ConfigError(
                "Frequentist ratios require normalization='independent_reference_mean'."
            )
        declared = set(parameter_names)
        parameter_roles = {
            parameter["name"]: parameter.get("role") for parameter in parameters
        }

        def validate_point(point: Any, location: str) -> None:
            if not isinstance(point, Mapping):
                raise ConfigError(f"{location} must be an object.")
            if not all(isinstance(name, str) for name in point):
                raise ConfigError(f"{location} parameter names must be strings.")
            names_at_point = set(point)
            if names_at_point != declared:
                raise ConfigError(
                    f"{location} must contain exactly {sorted(declared)}; "
                    f"found {sorted(names_at_point)}."
                )
            for name, item in point.items():
                if not isinstance(item, numbers.Real) or isinstance(item, bool):
                    raise ConfigError(f"{location}.{name} must be a number.")

        asimov = frequentist.get("asimov")
        if isinstance(asimov, Mapping):
            validate_point(
                asimov.get("parameter_point"),
                "frequentist.asimov.parameter_point",
            )
            if "normalization_source" in asimov:
                validate_source_columns(
                    asimov["normalization_source"],
                    model_features=observation_features,
                    location="frequentist.asimov.normalization_source",
                )
        toys = frequentist.get("toys")
        if isinstance(toys, Mapping):
            points = toys.get("parameter_points")
            if not isinstance(points, list) or not points:
                raise ConfigError(
                    "frequentist.toys.parameter_points must be a non-empty list."
                )
            for index, point in enumerate(points):
                validate_point(
                    point,
                    f"frequentist.toys.parameter_points[{index}]",
                )
        nis = frequentist.get("nis")
        if isinstance(nis, Mapping):
            points = nis.get("design_points")
            if not isinstance(points, list) or not points:
                raise ConfigError(
                    "frequentist.nis.design_points must be a non-empty list."
                )
            for index, point in enumerate(points):
                validate_point(
                    point,
                    f"frequentist.nis.design_points[{index}]",
                )
        systematics = frequentist.get("systematics", ())
        if not isinstance(systematics, (list, tuple)):
            raise ConfigError("frequentist.systematics must be a list.")
        systematic_names: set[str] = set()
        modifier_pairs: set[tuple[str, str]] = set()
        for systematic in systematics:
            if not isinstance(systematic, Mapping):
                raise ConfigError("Every frequentist systematic must be an object.")
            systematic_name = systematic.get("name")
            if not isinstance(systematic_name, str) or not systematic_name:
                raise ConfigError("Every systematic needs a name.")
            if systematic_name in systematic_names:
                raise ConfigError("Frequentist systematic names must be unique.")
            systematic_names.add(systematic_name)
            systematic_parameter = systematic.get("parameter")
            if not isinstance(systematic_parameter, str):
                raise ConfigError(
                    f"Systematic {systematic_name!r} needs a parameter name."
                )
            if systematic_parameter not in declared:
                raise ConfigError(
                    f"Systematic {systematic.get('name')!r} references an "
                    "undeclared parameter."
                )
            if parameter_roles[systematic_parameter] != "nuisance":
                raise ConfigError(
                    f"Systematic {systematic.get('name')!r} must reference "
                    "a nuisance parameter."
                )
            interpolation = systematic.get("interpolation", "nsbi_code4p")
            if interpolation not in {"linear", "nsbi_code4p"}:
                raise ConfigError(
                    f"Systematic {systematic_name!r} interpolation must be "
                    "'linear' or 'nsbi_code4p'."
                )
            variations = systematic.get("variations")
            if not isinstance(variations, list) or not variations:
                raise ConfigError(
                    f"Systematic {systematic.get('name')!r} needs at least "
                    "one variation."
                )
            variation_samples: set[str] = set()
            for variation in variations:
                if not isinstance(variation, Mapping):
                    raise ConfigError("Every systematic variation must be an object.")
                sample_name = variation.get("sample")
                if not isinstance(sample_name, str) or not sample_name:
                    raise ConfigError("Every systematic variation needs a sample name.")
                if sample_name in variation_samples:
                    raise ConfigError(
                        f"Systematic {systematic_name!r} repeats variation "
                        f"sample {sample_name!r}."
                    )
                for anchor_name in ("yield_up", "yield_down"):
                    if anchor_name not in variation:
                        continue
                    anchor = variation[anchor_name]
                    if (
                        isinstance(anchor, bool)
                        or not isinstance(anchor, numbers.Real)
                        or float(anchor) < 0
                    ):
                        raise ConfigError(
                            f"Systematic {systematic_name!r} variation "
                            f"{sample_name!r} {anchor_name} must be a finite "
                            "non-negative number."
                        )
                    if interpolation == "nsbi_code4p" and float(anchor) <= 0:
                        raise ConfigError(
                            f"Systematic {systematic_name!r} uses nsbi_code4p, "
                            f"so variation {sample_name!r} {anchor_name} must "
                            "be strictly positive."
                        )
                variation_samples.add(sample_name)
                pair = (sample_name, systematic_parameter)
                if pair in modifier_pairs:
                    raise ConfigError(
                        "A sample/parameter systematic modifier may only be "
                        f"declared once; repeated {pair!r}."
                    )
                modifier_pairs.add(pair)
                for direction in ("up", "down"):
                    validate_source_columns(
                        variation.get(direction),
                        model_features=observation_features,
                        location=(
                            f"frequentist.systematics[{systematic_name!r}]"
                            f".{sample_name}.{direction}"
                        ),
                    )
            unknown_samples = variation_samples.difference(names)
            if unknown_samples:
                raise ConfigError(
                    f"Systematic {systematic.get('name')!r} references "
                    f"unknown samples {sorted(unknown_samples)}."
                )
        fnf = frequentist.get("fnf")
        if fnf is not None:
            if not isinstance(fnf, Mapping):
                raise ConfigError("frequentist.fnf must be an object.")
            models = fnf.get("models")
            if not isinstance(models, list) or not models:
                raise ConfigError("frequentist.fnf.models must be a non-empty list.")
            parameter_entries = {
                parameter["name"]: parameter for parameter in parameters
            }
            model_names: set[str] = set()
            model_samples: set[str] = set()
            output_paths: set[str] = set()
            for model_index, model in enumerate(models):
                location = f"frequentist.fnf.models[{model_index}]"
                if not isinstance(model, Mapping):
                    raise ConfigError(f"{location} must be an object.")
                model_name = model.get("name")
                if not isinstance(model_name, str) or not model_name:
                    raise ConfigError(f"{location}.name must be a non-empty string.")
                if model_name in model_names:
                    raise ConfigError("FNF model names must be unique.")
                model_names.add(model_name)
                sample = model.get("sample")
                if sample not in names:
                    raise ConfigError(
                        f"FNF model {model_name!r} references unknown sample "
                        f"{sample!r}."
                    )
                if sample in model_samples:
                    raise ConfigError(
                        "Only one FNF model may be bound to each physics sample; "
                        f"sample {sample!r} is repeated."
                    )
                model_samples.add(sample)
                nuisance_names = model.get("nuisances")
                if (
                    not isinstance(nuisance_names, list)
                    or not nuisance_names
                    or any(
                        not isinstance(name, str) or not name for name in nuisance_names
                    )
                    or len(set(nuisance_names)) != len(nuisance_names)
                ):
                    raise ConfigError(
                        f"{location}.nuisances must contain unique parameter names."
                    )
                unknown_nuisances = set(nuisance_names).difference(declared)
                if unknown_nuisances:
                    raise ConfigError(
                        f"FNF model {model_name!r} references undeclared "
                        f"parameters {sorted(unknown_nuisances)}."
                    )
                non_nuisance = [
                    name
                    for name in nuisance_names
                    if parameter_roles[name] != "nuisance"
                ]
                if non_nuisance:
                    raise ConfigError(
                        f"FNF model {model_name!r} parameters must have "
                        f"role='nuisance'; found {sorted(non_nuisance)}."
                    )
                centers = {
                    name: float(parameter_entries[name]["nominal"])
                    for name in nuisance_names
                }
                configured_centers = model.get("centers", {})
                configured_scales = model.get("scales", {})
                for field, configured in (
                    ("centers", configured_centers),
                    ("scales", configured_scales),
                ):
                    if not isinstance(configured, Mapping):
                        raise ConfigError(f"{location}.{field} must be an object.")
                    unknown = set(configured).difference(nuisance_names)
                    if unknown:
                        raise ConfigError(
                            f"{location}.{field} references parameters "
                            f"{sorted(unknown)} outside this FNF model."
                        )
                    if any(
                        isinstance(item, bool) or not isinstance(item, numbers.Real)
                        for item in configured.values()
                    ):
                        raise ConfigError(f"{location}.{field} values must be numbers.")
                if any(float(item) <= 0 for item in configured_scales.values()):
                    raise ConfigError(f"{location}.scales values must be positive.")
                centers.update(
                    {name: float(item) for name, item in configured_centers.items()}
                )
                anchors = model.get("anchors")
                if not isinstance(anchors, list) or not anchors:
                    raise ConfigError(
                        f"FNF model {model_name!r} needs at least one anchor."
                    )
                anchor_names: set[str] = set()
                resolved_points: list[dict[str, float]] = []
                for anchor_index, anchor in enumerate(anchors):
                    anchor_location = f"{location}.anchors[{anchor_index}]"
                    if not isinstance(anchor, Mapping):
                        raise ConfigError(f"{anchor_location} must be an object.")
                    anchor_name = anchor.get("name")
                    if not isinstance(anchor_name, str) or not anchor_name:
                        raise ConfigError(
                            f"{anchor_location}.name must be a non-empty string."
                        )
                    if anchor_name in anchor_names:
                        raise ConfigError(
                            f"FNF model {model_name!r} anchor names must be unique."
                        )
                    anchor_names.add(anchor_name)
                    point = anchor.get("point")
                    if not isinstance(point, Mapping) or not point:
                        raise ConfigError(
                            f"{anchor_location}.point must be a non-empty object."
                        )
                    unknown = set(point).difference(nuisance_names)
                    if unknown:
                        raise ConfigError(
                            f"{anchor_location}.point references parameters "
                            f"{sorted(unknown)} outside this FNF model."
                        )
                    if any(
                        isinstance(item, bool) or not isinstance(item, numbers.Real)
                        for item in point.values()
                    ):
                        raise ConfigError(
                            f"{anchor_location}.point values must be numbers."
                        )
                    resolved = dict(centers)
                    resolved.update({name: float(item) for name, item in point.items()})
                    resolved_points.append(resolved)
                    validate_source_columns(
                        anchor.get("source"),
                        model_features=observation_features,
                        location=f"{anchor_location}.source",
                    )
                uncovered = [
                    name
                    for name in nuisance_names
                    if not any(
                        not math.isclose(
                            point[name],
                            centers[name],
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                        for point in resolved_points
                    )
                ]
                if uncovered:
                    raise ConfigError(
                        f"FNF model {model_name!r} has no non-nominal anchor "
                        f"for parameters {uncovered}."
                    )
                interactions = model.get("interactions", ())
                if not isinstance(interactions, (list, tuple)):
                    raise ConfigError(
                        f"{location}.interactions must be a list of pairs."
                    )
                canonical_interactions: set[frozenset[str]] = set()
                for interaction in interactions:
                    if (
                        not isinstance(interaction, (list, tuple))
                        or len(interaction) != 2
                        or interaction[0] == interaction[1]
                        or not all(
                            isinstance(name, str) and name in nuisance_names
                            for name in interaction
                        )
                    ):
                        raise ConfigError(
                            f"{location}.interactions entries must contain "
                            "two distinct model nuisances."
                        )
                    pair = frozenset(interaction)
                    if pair in canonical_interactions:
                        raise ConfigError(
                            f"FNF model {model_name!r} repeats an interaction."
                        )
                    canonical_interactions.add(pair)
                    first, second = interaction
                    if not any(
                        not math.isclose(
                            point[first],
                            centers[first],
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                        and not math.isclose(
                            point[second],
                            centers[second],
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                        for point in resolved_points
                    ):
                        raise ConfigError(
                            f"FNF interaction {(first, second)!r} requires a "
                            "joint non-nominal anchor."
                        )
                yield_anchors = model.get("yield_anchors", {})
                if not isinstance(yield_anchors, Mapping):
                    raise ConfigError(f"{location}.yield_anchors must be an object.")
                unknown_yields = set(yield_anchors).difference(nuisance_names)
                if unknown_yields:
                    raise ConfigError(
                        f"{location}.yield_anchors references parameters "
                        f"{sorted(unknown_yields)} outside this FNF model."
                    )
                for name, pair in yield_anchors.items():
                    if (
                        not isinstance(pair, (list, tuple))
                        or len(pair) != 2
                        or any(
                            isinstance(item, bool)
                            or not isinstance(item, numbers.Real)
                            or float(item) <= 0
                            for item in pair
                        )
                    ):
                        raise ConfigError(
                            f"{location}.yield_anchors.{name} must contain "
                            "two positive relative factors."
                        )
                training = model.get("training", {})
                if not isinstance(training, Mapping):
                    raise ConfigError(f"{location}.training must be an object.")
                if (
                    float(training.get("validation_fraction", 0.2))
                    + float(training.get("holdout_fraction", 0.0))
                    >= 1.0
                ):
                    raise ConfigError(
                        f"{location}.training validation_fraction and "
                        "holdout_fraction must sum to less than one."
                    )
                output_path = model.get("output_path")
                if (
                    not isinstance(output_path, str)
                    or not output_path.endswith(".manifest.json")
                    or Path(output_path).name == ".manifest.json"
                ):
                    raise ConfigError(
                        f"{location}.output_path must end in '.manifest.json'."
                    )
                if output_path in output_paths:
                    raise ConfigError("FNF output paths must be unique.")
                output_paths.add(output_path)
        try:
            from .intensity import IntensityModel

            intensity_config = deepcopy(frequentist)
            for sample in intensity_config["samples"]:
                if isinstance(sample["nominal_yield"], Mapping):
                    sample["nominal_yield"] = 1.0
            IntensityModel.from_config(intensity_config)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid frequentist intensity model: {exc}") from exc
    bayesian = value.get("bayesian")
    if bayesian is not None:
        if not isinstance(bayesian, Mapping):
            raise ConfigError("bayesian must be an object.")
        theta = bayesian.get("theta_features")
        if not isinstance(theta, list) or not theta:
            raise ConfigError("bayesian.theta_features must be a non-empty list.")
        if not all(isinstance(name, str) and name for name in theta):
            raise ConfigError(
                "Every Bayesian theta feature must be a non-empty string."
            )
        if len(set(theta)) != len(theta):
            raise ConfigError("Bayesian theta features must be unique.")
        overlap = set(theta).intersection(features)
        if overlap:
            raise ConfigError(
                "Bayesian theta and observation features must be disjoint; "
                f"found {sorted(overlap)}."
            )
        datasets = bayesian.get("datasets")
        if not isinstance(datasets, Mapping):
            raise ConfigError("bayesian.datasets must be an object.")
        missing = {"rho", "nu", "kappa"}.difference(datasets)
        if missing:
            raise ConfigError(
                f"Bayesian dual training requires proposal datasets {sorted(missing)}."
            )
        model_features = set(theta).union(features)
        for name, specification in datasets.items():
            validate_source_columns(
                specification,
                model_features=model_features,
                location=f"bayesian.datasets.{name}",
            )
        designs = bayesian.get("design_distributions")
        if not isinstance(designs, Mapping):
            raise ConfigError("bayesian.design_distributions must be an object.")
        missing_designs = {"rho", "nu", "kappa"}.difference(designs)
        if missing_designs:
            raise ConfigError(
                "Bayesian dual training requires design distributions "
                f"{sorted(missing_designs)}."
            )
        dimension = len(theta)

        def numeric_vector(
            item: Any,
            *,
            location: str,
            positive: bool = False,
        ) -> tuple[float, ...]:
            if (
                not isinstance(item, list)
                or len(item) != dimension
                or any(
                    not isinstance(entry, numbers.Real) or isinstance(entry, bool)
                    for entry in item
                )
            ):
                qualifier = "positive numeric" if positive else "numeric"
                raise ConfigError(
                    f"{location} must contain {dimension} {qualifier} values."
                )
            result = tuple(float(entry) for entry in item)
            if positive and any(entry <= 0 for entry in result):
                raise ConfigError(f"{location} values must be positive.")
            return result

        for name in ("rho", "nu", "kappa"):
            design = designs[name]
            if not isinstance(design, Mapping):
                raise ConfigError(
                    f"bayesian.design_distributions.{name} must be an object."
                )
            kind = design.get("kind")
            if kind == "independent_normal":
                numeric_vector(
                    design.get("mean"),
                    location=f"Bayesian design {name!r} mean",
                )
                numeric_vector(
                    design.get("scale"),
                    location=f"Bayesian design {name!r} scale",
                    positive=True,
                )
            elif kind == "box_uniform":
                low = numeric_vector(
                    design.get("low"),
                    location=f"Bayesian design {name!r} low",
                )
                high = numeric_vector(
                    design.get("high"),
                    location=f"Bayesian design {name!r} high",
                )
                if any(lower >= upper for lower, upper in zip(low, high, strict=True)):
                    raise ConfigError(
                        f"Bayesian design {name!r} requires low < high in "
                        "every dimension."
                    )
            elif kind == "registry":
                if (
                    not isinstance(design.get("registry_key"), str)
                    or not design["registry_key"]
                ):
                    raise ConfigError(
                        f"Bayesian registry design {name!r} needs registry_key."
                    )
            else:
                raise ConfigError(
                    f"Bayesian design {name!r} has unsupported kind {kind!r}."
                )
        for field in ("posterior_flow", "likelihood_flow"):
            flow = bayesian.get(field)
            if not isinstance(flow, Mapping):
                raise ConfigError(f"bayesian.{field} must be an object.")
            if flow.get("architecture") != "quadratic_spline":
                raise ConfigError(
                    f"bayesian.{field}.architecture must be "
                    "'quadratic_spline' for conditional density training."
                )


def load_config(
    source: str | Path | Mapping[str, Any],
    *,
    validate_schema: bool = True,
) -> dict[str, Any]:
    """Load a YAML/JSON file or copy a Python dictionary.

    If ``jsonschema`` is installed, the distributed schema is applied in
    addition to the dependency-free structural checks.
    """

    value = _read_mapping(source)
    _require_finite_numbers(value)
    _manual_validate(value)
    if validate_schema:
        try:
            import jsonschema
        except ImportError:
            jsonschema = None
        if jsonschema is None:
            raise ConfigError(
                "JSON Schema validation requires jsonschema; reinstall "
                "hnsbi-toolkit with its required dependencies or pass "
                "validate_schema=False explicitly."
            )
        schema = _load_schema()
        try:
            jsonschema.validate(value, schema)
        except jsonschema.ValidationError as exc:
            location = ".".join(map(str, exc.absolute_path))
            prefix = f"{location}: " if location else ""
            raise ConfigError(prefix + exc.message) from exc
    return value


@dataclass(frozen=True)
class ToolkitConfig:
    """Validated configuration with convenient section access."""

    raw: dict[str, Any]

    @classmethod
    def load(
        cls,
        source: str | Path | Mapping[str, Any],
        *,
        validate_schema: bool = True,
    ) -> ToolkitConfig:
        return cls(load_config(source, validate_schema=validate_schema))

    @property
    def features(self) -> tuple[str, ...]:
        return tuple(self.raw["features"])

    @property
    def output_dir(self) -> Path:
        return Path(self.raw.get("output_dir", "artifacts"))

    @property
    def frequentist(self) -> dict[str, Any] | None:
        value = self.raw.get("frequentist")
        return None if value is None else deepcopy(dict(value))

    @property
    def bayesian(self) -> dict[str, Any] | None:
        value = self.raw.get("bayesian")
        return None if value is None else deepcopy(dict(value))

    def dump(self, path: str | Path) -> Path:
        """Write YAML for ``.yaml``/``.yml`` paths and canonical JSON otherwise."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ConfigError(
                    "YAML configuration requires PyYAML; reinstall hnsbi-toolkit."
                ) from exc
            payload = yaml.safe_dump(
                self.raw,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
        elif path.suffix.lower() == ".json":
            payload = (
                json.dumps(
                    self.raw,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        else:
            raise ConfigError(
                "Configuration output must use a .yaml, .yml, or .json extension."
            )
        path.write_text(payload, encoding="utf-8")
        return path

    def dump_json(self, path: str | Path) -> Path:
        """Serialize the validated runtime contract as canonical JSON."""

        target = Path(path)
        if target.suffix.lower() != ".json":
            raise ConfigError("Canonical serialization paths must end in .json.")
        return self.dump(target)
