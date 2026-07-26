"""Thin runtime adapter for upstream LHC workspaces, fits, and scans."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import ArtifactManifest
from ..expressions import Expression
from ..intensity import Component, IntensityModel, Parameter
from ..onnx import require_optional

_ARRAY_PATH_KEYS = frozenset({"weights", "ratios", "hi_ratio", "lo_ratio"})
_ASIMOV_FIXED_ROLES = frozenset(
    {
        "event-weights",
        "reference-integration-weights",
        "metadata",
    }
)
_SYSTEMATIC_ROLES = frozenset({"up-ratio", "down-ratio"})


def _workspace_payload(
    source: str | Path | Mapping[str, Any],
    base_directory: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(source, (str, Path)):
        workspace_path = Path(source)
        with workspace_path.open(encoding="utf-8") as stream:
            workspace = json.load(stream)
        base = workspace_path.parent if base_directory is None else Path(base_directory)
    elif isinstance(source, Mapping):
        workspace = copy.deepcopy(dict(source))
        base = Path.cwd() if base_directory is None else Path(base_directory)
    else:
        raise TypeError("source must be a workspace mapping or JSON path.")
    if not isinstance(workspace, dict):
        raise TypeError("The workspace JSON root must be an object.")
    if "channels" not in workspace or "measurements" not in workspace:
        raise ValueError(
            "An upstream workspace requires 'channels' and 'measurements'."
        )
    return workspace, base.resolve()


def _resolved_file(
    value: Any,
    *,
    base: Path,
    label: str,
) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{label} must be a filesystem path.")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _manifest_roles(
    manifest: ArtifactManifest,
    *,
    manifest_path: Path,
    expected: set[str] | frozenset[str],
    label: str,
) -> dict[str, Path]:
    roles: dict[str, Path] = {}
    for record in manifest.files:
        if record.kind in roles:
            raise ValueError(f"{label} repeats role {record.kind!r}.")
        roles[record.kind] = (manifest_path.parent / record.path).resolve()
    actual = set(roles)
    if actual != set(expected):
        missing = sorted(set(expected).difference(actual))
        unexpected = sorted(actual.difference(expected))
        raise ValueError(
            f"{label} roles do not match the workspace contract; "
            f"missing={missing}, unexpected={unexpected}."
        )
    return roles


def _load_vector(
    path: Path,
    *,
    role: str,
    rows: int,
    positive_measure: bool = False,
) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False))
    if values.ndim != 1 or len(values) != rows:
        raise ValueError(
            f"Workspace array role {role!r} must be one-dimensional with {rows} rows."
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(
            f"Workspace array role {role!r} must be finite and non-negative."
        )
    if positive_measure and not float(np.sum(values)) > 0:
        raise ValueError(
            f"Workspace array role {role!r} must have positive total weight."
        )
    return values


def _scalar_anchor(value: Any, *, label: str) -> float:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(values) != 1 or not np.isfinite(values).all() or values[0] < 0:
        raise ValueError(f"{label} must be one finite non-negative scalar.")
    return float(values[0])


def _workspace_intensity(
    extension: Mapping[str, Any],
    *,
    channel: Mapping[str, Any],
    measurement: Mapping[str, Any],
) -> IntensityModel:
    samples = channel.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("An hNSBI workspace requires at least one sample.")
    sample_names = [
        sample.get("name") if isinstance(sample, Mapping) else None
        for sample in samples
    ]
    if any(not isinstance(name, str) or not name for name in sample_names):
        raise ValueError("Every hNSBI sample requires a non-empty name.")
    if len(sample_names) != len(set(sample_names)):
        raise ValueError("hNSBI sample names must be unique.")

    multiplier_map = extension.get("sample_multipliers")
    if not isinstance(multiplier_map, Mapping) or set(multiplier_map) != set(
        sample_names
    ):
        raise ValueError("hNSBI sample_multipliers must exactly match channel samples.")
    components: list[Component] = []
    for sample in samples:
        assert isinstance(sample, Mapping)
        name = str(sample["name"])
        data = np.asarray(sample.get("data"), dtype=np.float64).reshape(-1)
        if len(data) != 1 or not np.isfinite(data).all() or data[0] < 0:
            raise ValueError(
                f"Sample {name!r} must contain one finite non-negative yield."
            )
        sample_extension = sample.get("hnsbi")
        if not isinstance(sample_extension, Mapping):
            raise ValueError(f"Sample {name!r} has no hNSBI extension.")
        multiplier = sample_extension.get("multiplier")
        if multiplier != multiplier_map[name]:
            raise ValueError(
                f"Sample {name!r} multiplier disagrees with hnsbi.sample_multipliers."
            )
        components.append(Component(name, float(data[0]), multiplier=multiplier))

    config = measurement.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("The hNSBI measurement requires a config object.")
    entries = config.get("parameters")
    if not isinstance(entries, list):
        raise ValueError("The hNSBI measurement requires a parameter list.")
    nominals = extension.get("parameter_nominals")
    if not isinstance(nominals, Mapping):
        raise ValueError("The hNSBI extension has no parameter_nominals map.")
    entry_names = [
        entry.get("name") if isinstance(entry, Mapping) else None for entry in entries
    ]
    if any(not isinstance(name, str) or not name for name in entry_names):
        raise ValueError("Every hNSBI parameter requires a non-empty name.")
    if len(entry_names) != len(set(entry_names)) or set(entry_names) != set(nominals):
        raise ValueError(
            "hNSBI parameter_nominals must exactly match measurement parameters."
        )
    parameters: list[Parameter] = []
    for entry in entries:
        assert isinstance(entry, Mapping)
        name = str(entry["name"])
        bounds_value = entry.get("bounds")
        bounds: tuple[float, float] | None = None
        if bounds_value is not None:
            if (
                not isinstance(bounds_value, list)
                or len(bounds_value) != 1
                or not isinstance(bounds_value[0], list)
                or len(bounds_value[0]) != 2
            ):
                raise ValueError(f"Parameter {name!r} has malformed upstream bounds.")
            bounds = tuple(map(float, bounds_value[0]))
        constraint = entry.get("hnsbi_constraint")
        if constraint is not None and not isinstance(constraint, Mapping):
            raise ValueError(f"Parameter {name!r} has a malformed hNSBI constraint.")
        parameters.append(
            Parameter(
                name=name,
                nominal=float(nominals[name]),
                bounds=bounds,
                constrained=constraint is not None,
                constraint_mean=float((constraint or {}).get("mean", 0.0)),
                constraint_sigma=float((constraint or {}).get("sigma", 1.0)),
            )
        )
    intensity = IntensityModel(components, parameters)
    specification = extension.get("intensity_specification")
    if specification != intensity.specification():
        raise ValueError(
            "Workspace intensity fields do not match the recorded hNSBI "
            "intensity specification."
        )
    fingerprint = extension.get("intensity_fingerprint")
    if fingerprint != intensity.fingerprint:
        raise ValueError(
            "Workspace intensity does not match its recorded scientific fingerprint."
        )
    return intensity


def _verify_systematic(
    modifier: Mapping[str, Any],
    *,
    sample_name: str,
    declared_parameters: set[str],
    rows: int,
    base: Path,
) -> None:
    name = modifier.get("name")
    if not isinstance(name, str) or name not in declared_parameters:
        raise ValueError(
            f"Systematic for sample {sample_name!r} references undeclared "
            f"parameter {name!r}."
        )
    if modifier.get("type") != "normplusshape":
        raise ValueError(f"Unsupported hNSBI modifier type {modifier.get('type')!r}.")
    data = modifier.get("data")
    if not isinstance(data, Mapping):
        raise ValueError(f"Systematic {sample_name}/{name} has no data object.")
    yield_up = _scalar_anchor(
        data.get("hi_data"), label=f"Systematic {sample_name}/{name} hi_data"
    )
    yield_down = _scalar_anchor(
        data.get("lo_data"), label=f"Systematic {sample_name}/{name} lo_data"
    )
    modifier_extension = modifier.get("hnsbi")
    if not isinstance(modifier_extension, Mapping):
        raise ValueError(f"Systematic {sample_name}/{name} has no hNSBI extension.")
    manifest_path = _resolved_file(
        modifier_extension.get("manifest"),
        base=base,
        label=f"Systematic {sample_name}/{name} manifest",
    )
    manifest = ArtifactManifest.load(manifest_path)
    if manifest.artifact_type != "systematic-anchor":
        raise ValueError(
            f"Systematic {sample_name}/{name} manifest has unexpected "
            f"artifact type {manifest.artifact_type!r}."
        )
    manifest.verify(manifest_path.parent)
    roles = _manifest_roles(
        manifest,
        manifest_path=manifest_path,
        expected=_SYSTEMATIC_ROLES,
        label=f"Systematic {sample_name}/{name} manifest",
    )
    paths = {
        "up-ratio": _resolved_file(
            data.get("hi_ratio"),
            base=base,
            label=f"Systematic {sample_name}/{name} hi_ratio",
        ),
        "down-ratio": _resolved_file(
            data.get("lo_ratio"),
            base=base,
            label=f"Systematic {sample_name}/{name} lo_ratio",
        ),
    }
    for role, path in paths.items():
        if path != roles[role]:
            workspace_role = "hi_ratio" if role == "up-ratio" else "lo_ratio"
            raise ValueError(
                f"Systematic {sample_name}/{name} {workspace_role} does not "
                "match its checksummed manifest role."
            )
        _load_vector(path, role=role, rows=rows)
    expected_metadata = {
        "component": sample_name,
        "parameter": name,
        "rows": rows,
        "yield_up": yield_up,
        "yield_down": yield_down,
        "interpolation": modifier_extension.get("interpolation"),
    }
    for key, expected in expected_metadata.items():
        if manifest.metadata.get(key) != expected:
            raise ValueError(
                f"Systematic {sample_name}/{name} manifest metadata "
                f"{key!r} does not match the workspace."
            )


def _verify_hnsbi_workspace(
    workspace: Mapping[str, Any],
    *,
    base: Path,
) -> None:
    extension = workspace.get("hnsbi")
    if extension is None:
        return
    if not isinstance(extension, Mapping):
        raise ValueError("The hnsbi workspace extension must be an object.")
    if extension.get("schema_version") != "1.0":
        raise ValueError("Unsupported or missing hNSBI workspace schema_version.")
    if not isinstance(extension.get("upstream_compatible"), bool):
        raise ValueError(
            "The hNSBI extension must declare upstream_compatible as a boolean."
        )
    channels = workspace.get("channels")
    measurements = workspace.get("measurements")
    if not isinstance(channels, list) or len(channels) != 1:
        raise ValueError(
            "An hNSBI upstream workspace must contain exactly one channel."
        )
    if not isinstance(measurements, list) or len(measurements) != 1:
        raise ValueError(
            "An hNSBI upstream workspace must contain exactly one measurement."
        )
    channel = channels[0]
    measurement = measurements[0]
    if not isinstance(channel, Mapping) or channel.get("type") != "unbinned":
        raise ValueError("The hNSBI channel must have type 'unbinned'.")
    if not isinstance(measurement, Mapping):
        raise ValueError("The hNSBI measurement must be an object.")
    intensity = _workspace_intensity(
        extension,
        channel=channel,
        measurement=measurement,
    )
    asimov_point = extension.get("asimov_point")
    parameter_names = {parameter.name for parameter in intensity.parameters}
    if not isinstance(asimov_point, Mapping) or set(asimov_point) != parameter_names:
        raise ValueError(
            "hNSBI asimov_point must exactly match the intensity parameters."
        )
    try:
        parsed_point = {name: float(value) for name, value in asimov_point.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("hNSBI asimov_point values must be numeric.") from exc
    if not all(np.isfinite(value) for value in parsed_point.values()):
        raise ValueError("hNSBI asimov_point values must be finite.")
    parameter_entries = measurement["config"]["parameters"]
    for entry in parameter_entries:
        name = entry["name"]
        initial = np.asarray(entry.get("inits"), dtype=np.float64).reshape(-1)
        if (
            len(initial) != 1
            or not np.isfinite(initial).all()
            or not np.isclose(initial[0], parsed_point[name], rtol=0.0, atol=0.0)
        ):
            raise ValueError(
                f"Upstream initial value for parameter {name!r} does not "
                "match the hNSBI Asimov generating point."
            )
    if extension["upstream_compatible"]:
        incompatible_constraints = [
            parameter.name
            for parameter in intensity.parameters
            if parameter.constrained
            and (
                not np.isclose(parameter.constraint_mean, 0.0, rtol=0.0, atol=0.0)
                or not np.isclose(parameter.constraint_sigma, 1.0, rtol=0.0, atol=0.0)
                or not np.isclose(
                    parsed_point[parameter.name],
                    parameter.constraint_mean,
                    rtol=0.0,
                    atol=0.0,
                )
            )
        ]
        if incompatible_constraints:
            raise ValueError(
                "Workspace is marked upstream-compatible but contains "
                "constraints the upstream likelihood cannot preserve for "
                f"parameters {incompatible_constraints}."
            )
    samples = channel["samples"]
    sample_names = [str(sample["name"]) for sample in samples]
    expected_roles = set(_ASIMOV_FIXED_ROLES)
    expected_roles.update(f"process-ratio:{name}" for name in sample_names)
    manifest_path = _resolved_file(
        extension.get("array_manifest"),
        base=base,
        label="hNSBI Asimov array manifest",
    )
    manifest = ArtifactManifest.load(manifest_path)
    if manifest.artifact_type != "asimov-array-bundle":
        raise ValueError(
            "Workspace array manifest has unexpected artifact type "
            f"{manifest.artifact_type!r}."
        )
    manifest.verify(manifest_path.parent)
    roles = _manifest_roles(
        manifest,
        manifest_path=manifest_path,
        expected=expected_roles,
        label="hNSBI Asimov array manifest",
    )

    rows_value = manifest.metadata.get("rows")
    if (
        isinstance(rows_value, bool)
        or not isinstance(rows_value, int)
        or rows_value < 1
    ):
        raise ValueError(
            "hNSBI Asimov array manifest metadata 'rows' must be positive."
        )
    rows = rows_value
    if extension.get("asimov_raw_count") != rows:
        raise ValueError(
            "hNSBI asimov_raw_count does not match the checksummed array manifest."
        )
    manifest_samples = manifest.metadata.get("samples")
    if (
        not isinstance(manifest_samples, list)
        or any(not isinstance(name, str) for name in manifest_samples)
        or len(manifest_samples) != len(set(manifest_samples))
        or set(manifest_samples) != set(sample_names)
    ):
        raise ValueError(
            "hNSBI Asimov array manifest samples do not match the workspace."
        )
    features = extension.get("features")
    if not isinstance(features, list) or manifest.metadata.get("features") != features:
        raise ValueError(
            "hNSBI Asimov array manifest feature order does not match the workspace."
        )
    if manifest.metadata.get("intensity_fingerprint") != intensity.fingerprint:
        raise ValueError(
            "hNSBI Asimov array manifest belongs to a different intensity."
        )
    metadata_path = roles["metadata"]
    try:
        with metadata_path.open(encoding="utf-8") as stream:
            asimov_metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "The checksummed Asimov metadata role is not valid JSON."
        ) from exc
    if not isinstance(asimov_metadata, Mapping):
        raise ValueError("The checksummed Asimov metadata must be an object.")
    if asimov_metadata.get("intensity_fingerprint") != intensity.fingerprint:
        raise ValueError(
            "The checksummed Asimov metadata belongs to a different intensity."
        )
    if asimov_metadata.get("point") != dict(asimov_point):
        raise ValueError(
            "The checksummed Asimov metadata point does not match the workspace."
        )

    ratio_normalization = extension.get("ratio_normalization")
    if (
        not isinstance(ratio_normalization, Mapping)
        or set(ratio_normalization) != set(sample_names)
        or asimov_metadata.get("ratio_normalizers") != dict(ratio_normalization)
    ):
        raise ValueError(
            "hNSBI ratio normalization metadata does not consistently match "
            "the workspace samples."
        )
    for sample in samples:
        sample_extension = sample["hnsbi"]
        if (
            sample_extension.get("ratio_normalizer")
            != ratio_normalization[sample["name"]]
        ):
            raise ValueError(
                f"Sample {sample['name']!r} ratio normalizer disagrees with "
                "the hNSBI extension."
            )

    workspace_paths = {
        "event-weights": _resolved_file(
            channel.get("weights"), base=base, label="channel weights"
        ),
        "reference-integration-weights": _resolved_file(
            extension.get("reference_weights"),
            base=base,
            label="hNSBI reference integration weights",
        ),
    }
    for sample in samples:
        workspace_paths[f"process-ratio:{sample['name']}"] = _resolved_file(
            sample.get("ratios"),
            base=base,
            label=f"Sample {sample['name']!r} ratios",
        )
    for role, path in workspace_paths.items():
        if path != roles[role]:
            raise ValueError(
                f"Workspace path for {role!r} does not match its "
                "checksummed manifest role."
            )
        _load_vector(
            path,
            role=role,
            rows=rows,
            positive_measure=role == "reference-integration-weights",
        )

    declared_parameters = {parameter.name for parameter in intensity.parameters}
    has_systematics = False
    for component, sample in zip(intensity.components, samples, strict=True):
        modifiers = sample.get("modifiers", [])
        if not isinstance(modifiers, list):
            raise ValueError(f"Sample {component.name!r} modifiers must be a list.")
        normfactors: list[str] = []
        systematic_names: list[str] = []
        for modifier in modifiers:
            if not isinstance(modifier, Mapping):
                raise ValueError(f"Sample {component.name!r} has a malformed modifier.")
            if modifier.get("type") == "normfactor":
                name = modifier.get("name")
                if not isinstance(name, str):
                    raise ValueError("A normfactor modifier requires a name.")
                normfactors.append(name)
            else:
                has_systematics = True
                _verify_systematic(
                    modifier,
                    sample_name=component.name,
                    declared_parameters=declared_parameters,
                    rows=rows,
                    base=base,
                )
                systematic_names.append(str(modifier.get("name")))
        if len(systematic_names) != len(set(systematic_names)):
            raise ValueError(
                f"Sample {component.name!r} repeats a systematic modifier."
            )
        expected_normfactors = component.multiplier.simple_normfactors()
        if expected_normfactors is not None and tuple(normfactors) != (
            expected_normfactors
        ):
            raise ValueError(
                f"Sample {component.name!r} normfactor modifiers do not "
                "implement its recorded multiplier."
            )
        if extension["upstream_compatible"] and expected_normfactors is None:
            raise ValueError(
                f"Sample {component.name!r} has a multiplier that the "
                "upstream backend cannot represent."
            )
    if extension["upstream_compatible"] and has_systematics:
        raise ValueError(
            "Workspace is marked upstream-compatible but contains hNSBI "
            "reference-normalized normplusshape systematics."
        )


def _upstream_incompatibility_message(
    workspace: Mapping[str, Any],
) -> str | None:
    extension = workspace.get("hnsbi")
    if (
        not isinstance(extension, Mapping)
        or extension.get("upstream_compatible") is not False
    ):
        return None
    reasons: list[str] = []
    details = extension.get("upstream_incompatibilities")
    if isinstance(details, Mapping):
        reasons.extend(
            f"{name}={value}"
            for name, value in details.items()
            if value not in (None, False, "", [], {})
        )
    multipliers = extension.get("sample_multipliers")
    if isinstance(multipliers, Mapping):
        unsupported: dict[str, Any] = {}
        for name, source in multipliers.items():
            try:
                representable = (
                    Expression.parse(source).simple_normfactors() is not None
                )
            except (TypeError, ValueError):
                representable = False
            if not representable:
                unsupported[str(name)] = source
        if unsupported:
            reasons.append(f"unsupported sample multipliers={unsupported}")
    for channel in workspace.get("channels", ()):
        if not isinstance(channel, Mapping):
            continue
        for sample in channel.get("samples", ()):
            if not isinstance(sample, Mapping):
                continue
            if any(
                isinstance(modifier, Mapping)
                and modifier.get("type") == "normplusshape"
                for modifier in sample.get("modifiers", ())
            ):
                reasons.append("reference-normalized normplusshape systematics")
                break
    reason_text = (
        "; ".join(dict.fromkeys(reasons))
        if reasons
        else "no structured incompatibility reason was recorded"
    )
    return (
        "This hNSBI workspace is explicitly marked "
        "upstream_compatible=false and cannot be represented safely by "
        "nsbi-common-utils. Incompatibilities can include nonlinear hNSBI "
        "sample multipliers, reference-normalized systematics, non-nominal "
        "constrained Asimov points, or nonstandard Gaussian constraints. "
        f"Recorded reason(s): {reason_text}. Load it with "
        "ExtendedUnbinnedLikelihood.from_workspace() instead."
    )


def resolve_workspace_array_paths(
    workspace: Mapping[str, Any],
    base_directory: str | Path,
    *,
    check_exists: bool = True,
) -> dict[str, Any]:
    """Deep-copy a workspace and make its relative ``.npy`` paths absolute.

    The upstream model loads four array fields lazily with ``numpy.load``:
    channel ``weights``, sample ``ratios``, and systematic ``hi_ratio`` /
    ``lo_ratio`` values.  Resolving only these keys avoids rewriting ordinary
    workspace labels that happen to look like paths.
    """

    base = Path(base_directory).resolve()

    def visit(value: Any, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            return {
                item_key: visit(item_value, str(item_key))
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [visit(item, key) for item in value]
        if isinstance(value, tuple):
            return tuple(visit(item, key) for item in value)
        if key in _ARRAY_PATH_KEYS and isinstance(value, (str, Path)):
            path = Path(value)
            resolved = path if path.is_absolute() else base / path
            resolved = resolved.resolve()
            if check_exists and not resolved.is_file():
                raise FileNotFoundError(
                    f"Workspace array {key!r} does not exist: {resolved}"
                )
            return str(resolved)
        return copy.deepcopy(value)

    resolved = visit(workspace)
    if not isinstance(resolved, dict):
        raise TypeError("workspace must be a mapping.")
    return resolved


def load_upstream_workspace(
    source: str | Path | Mapping[str, Any],
    *,
    base_directory: str | Path | None = None,
    check_arrays: bool = True,
) -> dict[str, Any]:
    """Load a workspace, verify hNSBI bundles, and resolve external arrays.

    Checksummed hNSBI extensions are always verified before their arrays can
    reach the upstream runtime. ``check_arrays`` only controls the historical
    existence check for legacy workspaces that have no hNSBI extension.
    """

    workspace, base = _workspace_payload(source, base_directory)
    _verify_hnsbi_workspace(workspace, base=base)
    return resolve_workspace_array_paths(workspace, base, check_exists=check_arrays)


def _resolve_measurement(workspace: Mapping[str, Any], requested: str | None) -> str:
    measurements = workspace.get("measurements", ())
    names = [
        measurement.get("name")
        for measurement in measurements
        if isinstance(measurement, Mapping)
    ]
    names = [name for name in names if isinstance(name, str) and name]
    if requested is not None:
        if requested not in names:
            raise ValueError(
                f"Unknown measurement {requested!r}; available measurements "
                f"are {names}."
            )
        return requested
    if len(names) != 1:
        raise ValueError(
            "measurement_to_fit is required unless the workspace contains "
            "exactly one named measurement."
        )
    return names[0]


@dataclass
class NsbiCommonUtilsInference:
    """An upstream model and inference engine with a relocatable workspace."""

    workspace: dict[str, Any]
    measurement_to_fit: str
    model: Any
    engine: Any

    @classmethod
    def from_workspace(
        cls,
        source: str | Path | Mapping[str, Any],
        *,
        measurement_to_fit: str | None = None,
        base_directory: str | Path | None = None,
        check_arrays: bool = True,
        model_factory: Callable[..., Any] | None = None,
        inference_factory: Callable[..., Any] | None = None,
    ) -> NsbiCommonUtilsInference:
        """Load a workspace and instantiate upstream model/inference objects."""

        raw_workspace, raw_base = _workspace_payload(source, base_directory)
        incompatibility = _upstream_incompatibility_message(raw_workspace)
        if incompatibility is not None:
            raise ValueError(incompatibility)
        workspace = load_upstream_workspace(
            raw_workspace,
            base_directory=raw_base,
            check_arrays=check_arrays,
        )
        measurement = _resolve_measurement(workspace, measurement_to_fit)
        if model_factory is None:
            models = require_optional(
                "nsbi_common_utils.models",
                extra="lhc",
                purpose="constructing an LHC likelihood model",
            )
            model_factory = models.sbi_parametric_model
        if inference_factory is None:
            inference_module = require_optional(
                "nsbi_common_utils.inference",
                extra="lhc",
                purpose="performing LHC fits and profile scans",
            )
            inference_factory = inference_module.inference
        model = model_factory(workspace=workspace, measurement_to_fit=measurement)
        parameter_names, initial_values = model.get_model_parameters()
        engine = inference_factory(
            model_nll=model.model,
            initial_values=initial_values,
            list_parameters=parameter_names,
            num_unconstrained_params=model.num_unconstrained_param,
            model_grad=getattr(model, "model_grad", None),
        )
        return cls(
            workspace=workspace,
            measurement_to_fit=measurement,
            model=model,
            engine=engine,
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.engine.list_parameters)

    @property
    def best_fit(self) -> np.ndarray | None:
        values = getattr(self.engine, "pulls_global_fit", None)
        return None if values is None else np.asarray(values, dtype=np.float64)

    def perform_fit(
        self,
        *,
        fit_strategy: int = 2,
        freeze_params: Sequence[str] = (),
    ) -> np.ndarray:
        """Delegate the global fit and return best-fit values in model order."""

        self.engine.perform_fit(
            fit_strategy=fit_strategy, freeze_params=list(freeze_params)
        )
        best_fit = self.best_fit
        if best_fit is None:
            raise RuntimeError(
                "The upstream inference engine did not expose best-fit values."
            )
        return best_fit

    def perform_profile_scan(
        self,
        parameter_name: str,
        *,
        bound_range: tuple[float, float] = (0.0, 3.0),
        fit_strategy: int = 2,
        freeze_params: Sequence[str] = (),
        stat_only: bool = False,
        constrained_nuisance: bool = False,
        size: int = 100,
    ) -> tuple[np.ndarray, ...]:
        """Delegate an upstream profile scan and return NumPy arrays."""

        if parameter_name not in self.parameter_names:
            raise ValueError(
                f"Unknown parameter {parameter_name!r}; available parameters "
                f"are {list(self.parameter_names)}."
            )
        result = self.engine.perform_profile_scan(
            parameter_name=parameter_name,
            bound_range=bound_range,
            fit_strategy=fit_strategy,
            freeze_params=list(freeze_params),
            doStatOnly=stat_only,
            isConstrainedNP=constrained_nuisance,
            size=size,
        )
        return tuple(np.asarray(value) for value in result)

    def profile_scan(
        self,
        parameter_name: str,
        **kwargs: Any,
    ) -> tuple[np.ndarray, ...]:
        """Alias for :meth:`perform_profile_scan`."""

        return self.perform_profile_scan(parameter_name, **kwargs)
