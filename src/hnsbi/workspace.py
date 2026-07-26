"""Export the hNSBI intensity into the nsbi-common-utils workspace contract."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactManifest
from .asimov import AsimovResult
from .diagnostics import json_safe
from .intensity import Component, IntensityModel, Parameter, RatioNormalizer
from .systematics import SystematicSpecification


@dataclass(frozen=True)
class WorkspaceExport:
    """Workspace dictionary and its associated pre-evaluated array paths."""

    workspace: dict[str, Any]
    path: Path
    array_paths: Mapping[str, Path]
    upstream_compatible: bool


@dataclass(frozen=True)
class WorkspaceModel:
    """The generative hNSBI extension recovered from a workspace."""

    intensity: IntensityModel
    features: tuple[str, ...]
    ratio_normalizer: RatioNormalizer
    reference_manifest: Path | None
    ratio_manifests: Mapping[str, Path]
    systematics: Mapping[tuple[str, str], SystematicSpecification]
    path: Path


def _workspace_path(path: Path, base: Path, relative: bool) -> str:
    if relative:
        return os.path.relpath(path.resolve(), start=base.resolve())
    return str(path.resolve())


def _verified_manifest(
    path: str | Path,
    *,
    expected_types: set[str],
) -> tuple[Path, ArtifactManifest]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = ArtifactManifest.load(manifest_path)
    if manifest.artifact_type not in expected_types:
        raise ValueError(
            f"Unexpected artifact type {manifest.artifact_type!r} for "
            f"{manifest_path}; expected one of {sorted(expected_types)}."
        )
    manifest.verify(manifest_path.parent)
    return manifest_path, manifest


def _validated_systematic_modifier(
    modifier: Mapping[str, Any],
    *,
    component: str,
    parameters: set[str],
    rows: int,
    output_dir: Path,
    relative_paths: bool,
) -> dict[str, Any]:
    value = dict(modifier)
    name = value.get("name")
    if not isinstance(name, str) or name not in parameters:
        raise ValueError(
            f"Systematic modifier for {component!r} references undeclared "
            f"parameter {name!r}."
        )
    if value.get("type") != "normplusshape":
        raise ValueError(
            "Only nsbi-common-utils 'normplusshape' systematic modifiers are supported."
        )
    data = value.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("A normplusshape modifier requires a data object.")
    parsed_data = dict(data)
    for role in ("hi_data", "lo_data"):
        array = np.asarray(parsed_data.get(role), dtype=np.float64).reshape(-1)
        if len(array) != 1 or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError(
                f"Systematic {component}/{name} {role} must contain one "
                "finite non-negative yield factor."
            )
        parsed_data[role] = [float(array[0])]
    ratio_paths: dict[str, Path] = {}
    for role in ("hi_ratio", "lo_ratio"):
        if not isinstance(parsed_data.get(role), (str, Path)):
            raise ValueError(f"Systematic {component}/{name} is missing {role}.")
        path = Path(parsed_data[role])
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, allow_pickle=False)
        if (
            np.asarray(array).ndim != 1
            or len(array) != rows
            or not np.isfinite(array).all()
            or np.any(array < 0)
        ):
            raise ValueError(
                f"Systematic {component}/{name} {role} must be a finite "
                f"non-negative one-dimensional array with {rows} rows."
            )
        ratio_paths[role] = path.resolve()
        parsed_data[role] = _workspace_path(path, output_dir, relative_paths)
    extension = value.get("hnsbi")
    if not isinstance(extension, Mapping) or "manifest" not in extension:
        raise ValueError(f"Systematic {component}/{name} has no integrity manifest.")
    parsed_extension = dict(extension)
    manifest_path, manifest = _verified_manifest(
        parsed_extension["manifest"],
        expected_types={"systematic-anchor"},
    )
    expected_metadata = {
        "component": component,
        "parameter": name,
        "rows": rows,
    }
    for key, expected in expected_metadata.items():
        if manifest.metadata.get(key) != expected:
            raise ValueError(
                f"Systematic manifest metadata {key!r} does not match {expected!r}."
            )
    interpolation = parsed_extension.get("interpolation", "nsbi_code4p")
    if manifest.metadata.get("interpolation") != interpolation:
        raise ValueError(
            "Systematic manifest interpolation does not match its workspace modifier."
        )
    for metadata_key, data_key in (
        ("yield_up", "hi_data"),
        ("yield_down", "lo_data"),
    ):
        if not np.isclose(
            float(manifest.metadata.get(metadata_key, np.nan)),
            parsed_data[data_key][0],
        ):
            raise ValueError(
                f"Systematic manifest {metadata_key} does not match its "
                "workspace modifier."
            )
    records: dict[str, Path] = {}
    for record in manifest.files:
        if record.kind in records:
            raise ValueError(f"Systematic manifest repeats role {record.kind!r}.")
        records[record.kind] = (manifest_path.parent / record.path).resolve()
    if records.get("up-ratio") != ratio_paths["hi_ratio"]:
        raise ValueError("Systematic hi_ratio does not match its manifest.")
    if records.get("down-ratio") != ratio_paths["lo_ratio"]:
        raise ValueError("Systematic lo_ratio does not match its manifest.")
    parsed_extension["manifest"] = _workspace_path(
        manifest_path, output_dir, relative_paths
    )
    value["data"] = parsed_data
    value["hnsbi"] = parsed_extension
    return value


def write_nsbi_workspace(
    *,
    result: AsimovResult,
    intensity: IntensityModel,
    output_dir: str | Path,
    measurement: str,
    poi: str,
    channel: str = "SR",
    systematic_modifiers: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    reference_manifest: str | Path | None = None,
    ratio_manifests: Mapping[str, str | Path] | None = None,
    relative_paths: bool = True,
    require_upstream_compatible: bool = False,
    base_workspace: Mapping[str, Any] | None = None,
    workspace_filename: str = "workspace.json",
) -> WorkspaceExport:
    """Write an unbinned workspace plus the ratio/weight arrays it references.

    Products of ordinary parameter names are translated to upstream
    ``normfactor`` modifiers. General formulas are retained under the
    versioned ``hnsbi`` extension. They require the hNSBI formula model and
    are never silently simplified for the upstream backend.
    """

    output_dir = Path(output_dir)
    observed_fingerprint = result.events.metadata.get("intensity_fingerprint")
    if observed_fingerprint != intensity.fingerprint:
        raise ValueError(
            "The Asimov result was not built from this exact intensity "
            "definition (component order, yields, formulas, or parameters "
            "differ)."
        )
    arrays_dir = output_dir / "arrays"
    output_dir.mkdir(parents=True, exist_ok=True)
    array_paths = result.write_nsbi_arrays(arrays_dir)
    if result.systematic_anchors and systematic_modifiers:
        raise ValueError(
            "The Asimov result already contains support-bound systematics; "
            "do not also supply systematic_modifiers."
        )
    if result.systematic_anchors:
        modifiers_by_sample: Mapping[str, Sequence[Mapping[str, Any]]] = {
            component: [
                anchor.write_workspace_modifier(
                    output_dir / "systematics" / component / anchor.parameter
                )
                for anchor in anchors
            ]
            for component, anchors in result.systematic_anchors.items()
        }
    else:
        modifiers_by_sample = systematic_modifiers or {}
    unknown_modifier_samples = set(modifiers_by_sample).difference(
        intensity.component_names
    )
    if unknown_modifier_samples:
        raise ValueError(
            "Systematic modifiers reference unknown samples "
            f"{sorted(unknown_modifier_samples)}."
        )
    ratio_manifest_paths = {
        name: Path(path) for name, path in (ratio_manifests or {}).items()
    }
    unknown_manifests = set(ratio_manifest_paths).difference(intensity.component_names)
    if unknown_manifests:
        raise ValueError(
            f"Ratio manifests reference unknown samples {sorted(unknown_manifests)}."
        )
    if reference_manifest is not None:
        reference_manifest, reference_artifact = _verified_manifest(
            reference_manifest,
            expected_types={
                "reference-flow-checkpoint",
                "reference-flow-onnx-bundle",
            },
        )
        reference_features = tuple(reference_artifact.metadata.get("features", ()))
        if reference_features and reference_features != tuple(result.events.features):
            raise ValueError(
                "Reference-flow manifest has a different feature order from "
                "the Asimov sample."
            )
    for name, path in ratio_manifest_paths.items():
        verified, manifest = _verified_manifest(
            path,
            expected_types={"density-ratio-ensemble"},
        )
        features = tuple(manifest.metadata.get("features", ()))
        if features and features != tuple(result.events.features):
            raise ValueError(
                f"Ratio manifest for {name!r} has feature order {features}, "
                f"expected {tuple(result.events.features)}."
            )
        if manifest.metadata.get("numerator_name") != name:
            raise ValueError(
                f"Ratio manifest for {name!r} is bound to numerator "
                f"{manifest.metadata.get('numerator_name')!r}."
            )
        ratio_manifest_paths[name] = verified
    samples: list[dict[str, Any]] = []
    upstream_compatible = True
    formulas: dict[str, str] = {}
    for component in intensity.components:
        factors = component.multiplier.simple_normfactors()
        formulas[component.name] = component.multiplier.source
        if factors is None:
            upstream_compatible = False
            modifiers: list[dict[str, Any]] = []
        else:
            modifiers = [
                {"name": name, "data": None, "type": "normfactor"} for name in factors
            ]
        declared_parameters = {parameter.name for parameter in intensity.parameters}
        systematics = [
            _validated_systematic_modifier(
                modifier,
                component=component.name,
                parameters=declared_parameters,
                rows=result.raw_count,
                output_dir=output_dir,
                relative_paths=relative_paths,
            )
            for modifier in modifiers_by_sample.get(component.name, ())
        ]
        names = [modifier["name"] for modifier in systematics]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Sample {component.name!r} repeats a systematic parameter."
            )
        modifiers.extend(systematics)
        if systematics:
            # The hNSBI likelihood normalizes the joint shape morph under the
            # fixed reference quadrature at every nuisance point. The pinned
            # upstream code4p implementation does not, so routing this model
            # upstream would silently change its extended-rate semantics.
            upstream_compatible = False
        samples.append(
            {
                "name": component.name,
                "data": [float(component.nominal_yield)],
                "ratios": _workspace_path(
                    array_paths[f"ratio:{component.name}"],
                    output_dir,
                    relative_paths,
                ),
                "modifiers": modifiers,
                "hnsbi": {
                    "multiplier": component.multiplier.source,
                    "ratio_normalizer": result.normalizer.means[component.name],
                    **(
                        {
                            "ratio_manifest": _workspace_path(
                                ratio_manifest_paths[component.name],
                                output_dir,
                                relative_paths,
                            )
                        }
                        if component.name in ratio_manifest_paths
                        else {}
                    ),
                },
            }
        )
    constrained_offsets = {
        parameter.name: result.point[parameter.name]
        for parameter in intensity.parameters
        if parameter.constrained
        and not np.isclose(result.point[parameter.name], parameter.constraint_mean)
    }
    nonstandard_constraints = {
        parameter.name: {
            "mean": parameter.constraint_mean,
            "sigma": parameter.constraint_sigma,
        }
        for parameter in intensity.parameters
        if parameter.constrained
        and (
            not np.isclose(parameter.constraint_mean, 0.0)
            or not np.isclose(parameter.constraint_sigma, 1.0)
        )
    }
    expected_auxiliary_observations = {
        parameter.name: float(result.point[parameter.name])
        for parameter in intensity.parameters
        if parameter.constrained
    }
    if result.auxiliary_observations != expected_auxiliary_observations:
        raise ValueError(
            "The Asimov auxiliary observations must equal the generating "
            "values of every constrained parameter."
        )
    if constrained_offsets or nonstandard_constraints:
        # The upstream NLL hard-codes unit-Gaussian auxiliary observations
        # at zero. It cannot close an Asimov generated at another constraint
        # center without silently changing the statistical model.
        upstream_compatible = False
    if require_upstream_compatible and not upstream_compatible:
        unsupported = {
            component.name: component.multiplier.source
            for component in intensity.components
            if component.multiplier.simple_normfactors() is None
        }
        details = []
        if unsupported:
            details.append(f"Unsupported formulas: {unsupported}")
        if constrained_offsets:
            details.append(
                f"non-nominal constrained Asimov parameters: {constrained_offsets}"
            )
        if nonstandard_constraints:
            details.append(
                f"nonstandard Gaussian constraints: {nonstandard_constraints}"
            )
        systematic_samples = {
            component: [
                modifier["name"] for modifier in modifiers_by_sample.get(component, ())
            ]
            for component in intensity.component_names
            if modifiers_by_sample.get(component)
        }
        if systematic_samples:
            details.append(
                f"reference-normalized normplusshape systematics: {systematic_samples}"
            )
        raise ValueError(
            "The upstream nsbi-common-utils backend cannot represent this "
            + "; ".join(details)
            + "."
        )
    parameters: list[dict[str, Any]] = []
    for parameter in intensity.parameters:
        entry: dict[str, Any] = {
            "name": parameter.name,
            # nsbi-common-utils currently uses this vector both as the fit
            # start and to freeze the observed Asimov rate. It must therefore
            # be the generating point, not merely a convenient optimizer seed.
            "inits": [float(result.point[parameter.name])],
        }
        if parameter.bounds is not None:
            entry["bounds"] = [[*map(float, parameter.bounds)]]
        if parameter.constrained:
            entry["hnsbi_constraint"] = {
                "kind": "normal",
                "mean": float(parameter.constraint_mean),
                "sigma": float(parameter.constraint_sigma),
            }
        parameters.append(entry)
    if poi not in {parameter.name for parameter in intensity.parameters}:
        raise ValueError(f"POI {poi!r} is not an intensity parameter.")
    channel_payload = {
        "name": channel,
        "type": "unbinned",
        "weights": _workspace_path(array_paths["weights"], output_dir, relative_paths),
        "samples": samples,
    }
    measurement_payload = {
        "name": measurement,
        "config": {"parameters": parameters, "poi": poi},
    }
    if base_workspace is None:
        workspace = {
            "channels": [channel_payload],
            "measurements": [measurement_payload],
            "version": "1.0.0",
        }
    else:
        workspace = copy.deepcopy(dict(base_workspace))
        base_channels = workspace.get("channels", ())
        base_measurements = workspace.get("measurements", ())
        if len(base_channels) != 1 or base_channels[0].get("name") != channel:
            raise ValueError(
                "The configured upstream base workspace must contain exactly "
                f"the channel {channel!r}."
            )
        if (
            len(base_measurements) != 1
            or base_measurements[0].get("name") != measurement
        ):
            raise ValueError(
                "The configured upstream base workspace must contain exactly "
                f"the measurement {measurement!r}."
            )
        base_sample_names = {
            sample.get("name") for sample in base_channels[0].get("samples", ())
        }
        if base_sample_names != set(intensity.component_names):
            raise ValueError(
                "The upstream base workspace sample names do not match the "
                "hNSBI intensity components."
            )
        decorated_channel = dict(base_channels[0])
        decorated_channel.update(channel_payload)
        decorated_measurement = dict(base_measurements[0])
        decorated_measurement.update(measurement_payload)
        workspace["channels"] = [decorated_channel]
        workspace["measurements"] = [decorated_measurement]
        workspace.setdefault("version", "1.0.0")
    nonlinear_formulas = {
        component.name: component.multiplier.source
        for component in intensity.components
        if component.multiplier.simple_normfactors() is None
    }
    systematic_samples = {
        component: [
            modifier["name"] for modifier in modifiers_by_sample.get(component, ())
        ]
        for component in intensity.component_names
        if modifiers_by_sample.get(component)
    }
    workspace["hnsbi"] = {
        "schema_version": "1.0",
        "upstream_compatible": upstream_compatible,
        "sample_multipliers": formulas,
        "intensity_fingerprint": intensity.fingerprint,
        "intensity_specification": intensity.specification(),
        "ratio_normalization": dict(result.normalizer.means),
        "asimov_point": result.point,
        "asimov_raw_count": result.raw_count,
        "asimov_ess": result.ess,
        "auxiliary_observations": dict(result.auxiliary_observations),
        "features": list(result.events.features),
        "array_manifest": _workspace_path(
            array_paths["manifest"], output_dir, relative_paths
        ),
        "reference_weights": _workspace_path(
            array_paths["reference_weights"],
            output_dir,
            relative_paths,
        ),
        "parameter_nominals": {
            parameter.name: parameter.nominal for parameter in intensity.parameters
        },
        "upstream_inits_are_asimov_point": True,
        "upstream_incompatibilities": {
            "non_nominal_constrained_parameters": constrained_offsets,
            "nonlinear_formulas": nonlinear_formulas,
            "nonstandard_constraints": nonstandard_constraints,
            "reference_normalized_systematics": systematic_samples,
        },
        **(
            {
                "reference_manifest": _workspace_path(
                    Path(reference_manifest),
                    output_dir,
                    relative_paths,
                )
            }
            if reference_manifest is not None
            else {}
        ),
    }
    filename = Path(workspace_filename)
    if filename.name != workspace_filename or filename.suffix != ".json":
        raise ValueError("workspace_filename must be a plain filename ending in .json.")
    path = output_dir / workspace_filename
    path.write_text(
        json.dumps(
            json_safe(workspace),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return WorkspaceExport(
        workspace=workspace,
        path=path,
        array_paths=array_paths,
        upstream_compatible=upstream_compatible,
    )


def load_workspace(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or "channels" not in value:
        raise ValueError("Not an hNSBI/nsbi-common-utils workspace.")
    return value


def load_workspace_model(path: str | Path) -> WorkspaceModel:
    """Load the hNSBI intensity and portable artifact references.

    The returned model is backend independent. Callers may bind the manifest
    paths to native or ONNX runtime objects and pass them to
    :class:`hnsbi.toys.ToyGenerator`.
    """

    workspace_path = Path(path)
    workspace = load_workspace(workspace_path)
    extension = workspace.get("hnsbi")
    if not isinstance(extension, Mapping):
        raise ValueError("Workspace has no hnsbi generative extension.")
    channels = workspace.get("channels", [])
    if len(channels) != 1 or channels[0].get("type") != "unbinned":
        raise ValueError(
            "The hNSBI workspace loader currently expects one unbinned channel."
        )
    measurements = workspace.get("measurements", [])
    if len(measurements) != 1:
        raise ValueError("Workspace must contain exactly one measurement.")
    parameter_entries = measurements[0]["config"].get("parameters", [])
    parameters = []
    for entry in parameter_entries:
        bounds = entry.get("bounds")
        nominal = extension.get("parameter_nominals", {}).get(
            entry["name"], entry.get("inits", [0.0])[0]
        )
        parameters.append(
            Parameter(
                name=entry["name"],
                nominal=float(nominal),
                bounds=(tuple(map(float, bounds[0])) if bounds is not None else None),
                constrained="hnsbi_constraint" in entry,
                constraint_mean=float(
                    entry.get("hnsbi_constraint", {}).get("mean", 0.0)
                ),
                constraint_sigma=float(
                    entry.get("hnsbi_constraint", {}).get("sigma", 1.0)
                ),
            )
        )
    components = []
    normalizers: dict[str, float] = {}
    ratio_manifests: dict[str, Path] = {}
    systematics: dict[tuple[str, str], SystematicSpecification] = {}
    base = workspace_path.parent
    features = tuple(extension.get("features", ()))
    if not features:
        raise ValueError("Workspace hnsbi extension has no ordered features.")

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else base / candidate

    for sample in channels[0].get("samples", []):
        sample_extension = sample.get("hnsbi", {})
        if "multiplier" not in sample_extension:
            raise ValueError(f"Sample {sample.get('name')!r} has no hnsbi multiplier.")
        name = sample["name"]
        components.append(
            Component(
                name=name,
                nominal_yield=float(sample["data"][0]),
                multiplier=sample_extension["multiplier"],
            )
        )
        normalizers[name] = float(sample_extension["ratio_normalizer"])
        if "ratio_manifest" in sample_extension:
            ratio_path, ratio_manifest = _verified_manifest(
                resolve(sample_extension["ratio_manifest"]),
                expected_types={"density-ratio-ensemble"},
            )
            if tuple(ratio_manifest.metadata.get("features", ())) != features:
                raise ValueError(
                    f"Ratio manifest for {name!r} has the wrong feature order."
                )
            if ratio_manifest.metadata.get("numerator_name") != name:
                raise ValueError(
                    f"Ratio manifest for {name!r} is bound to a different "
                    "numerator sample."
                )
            ratio_manifests[name] = ratio_path
        for modifier in sample.get("modifiers", ()):
            modifier_type = modifier.get("type")
            if modifier_type == "normfactor":
                continue
            if modifier_type != "normplusshape":
                raise ValueError(
                    f"Unsupported generative modifier type {modifier_type!r}."
                )
            modifier_extension = modifier.get("hnsbi")
            if not isinstance(modifier_extension, Mapping) or (
                "manifest" not in modifier_extension
            ):
                raise ValueError(
                    f"Systematic modifier {name!r}/{modifier.get('name')!r} "
                    "has no integrity manifest."
                )
            systematic_manifest_path, systematic_manifest = _verified_manifest(
                resolve(modifier_extension["manifest"]),
                expected_types={"systematic-anchor"},
            )
            data = modifier.get("data")
            if not isinstance(data, Mapping):
                raise ValueError("Systematic modifier has no data object.")
            roles = {
                record.kind: (systematic_manifest_path.parent / record.path).resolve()
                for record in systematic_manifest.files
            }
            if set(roles) != {"up-ratio", "down-ratio"}:
                raise ValueError(
                    "Systematic manifest must contain exactly the up-ratio "
                    "and down-ratio roles."
                )
            for workspace_role, manifest_role in (
                ("hi_ratio", "up-ratio"),
                ("lo_ratio", "down-ratio"),
            ):
                if roles.get(manifest_role) != resolve(data[workspace_role]).resolve():
                    raise ValueError(
                        f"Systematic {name!r}/{modifier.get('name')!r} "
                        f"{workspace_role} does not match its manifest."
                    )
            hi_data = np.asarray(data.get("hi_data"), dtype=np.float64).reshape(-1)
            lo_data = np.asarray(data.get("lo_data"), dtype=np.float64).reshape(-1)
            if len(hi_data) != 1 or len(lo_data) != 1:
                raise ValueError("Systematic yield anchors must be scalar.")
            specification = SystematicSpecification(
                parameter=modifier["name"],
                component=name,
                yield_up=float(hi_data[0]),
                yield_down=float(lo_data[0]),
                interpolation=modifier_extension.get("interpolation", "nsbi_code4p"),
            )
            expected_systematic_metadata = {
                "component": name,
                "parameter": specification.parameter,
                "rows": int(extension.get("asimov_raw_count", -1)),
                "interpolation": specification.interpolation,
            }
            for metadata_key, expected in (
                *expected_systematic_metadata.items(),
                ("yield_up", specification.yield_up),
                ("yield_down", specification.yield_down),
            ):
                observed = systematic_manifest.metadata.get(metadata_key)
                matches = (
                    np.isclose(float(observed), float(expected))
                    if metadata_key.startswith("yield_") and observed is not None
                    else observed == expected
                )
                if not matches:
                    raise ValueError(
                        "Systematic manifest metadata "
                        f"{metadata_key!r} does not match the workspace."
                    )
            key = (name, specification.parameter)
            if key in systematics:
                raise ValueError(f"Workspace repeats systematic {key}.")
            systematics[key] = specification
    reference_manifest = (
        resolve(extension["reference_manifest"])
        if "reference_manifest" in extension
        else None
    )
    if reference_manifest is not None:
        reference_manifest, reference_artifact = _verified_manifest(
            reference_manifest,
            expected_types={
                "reference-flow-checkpoint",
                "reference-flow-onnx-bundle",
            },
        )
        if tuple(reference_artifact.metadata.get("features", ())) != features:
            raise ValueError("Reference-flow manifest has the wrong feature order.")
    intensity = IntensityModel(components, parameters)
    extension_normalizers = extension.get("ratio_normalization")
    if not isinstance(extension_normalizers, Mapping) or set(
        extension_normalizers
    ) != set(normalizers):
        raise ValueError("Workspace ratio-normalization keys do not match its samples.")
    for name, value in normalizers.items():
        if not np.isclose(float(extension_normalizers[name]), value):
            raise ValueError(f"Ratio normalizer for {name!r} is inconsistent.")
    expected_fingerprint = extension.get("intensity_fingerprint")
    if expected_fingerprint != intensity.fingerprint:
        raise ValueError(
            "Workspace intensity does not match its recorded scientific fingerprint."
        )
    return WorkspaceModel(
        intensity=intensity,
        features=features,
        ratio_normalizer=RatioNormalizer(normalizers),
        reference_manifest=reference_manifest,
        ratio_manifests=ratio_manifests,
        systematics=systematics,
        path=workspace_path,
    )
