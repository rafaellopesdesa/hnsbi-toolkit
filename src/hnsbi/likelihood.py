"""Formula-aware extended unbinned likelihoods, fits, and profile scans."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactManifest
from .fnf_runtime import (
    evaluate_fnf_on_support,
    validate_fnf_systematics,
)
from .intensity import Component, IntensityModel, Parameter, RatioNormalizer
from .systematics import SystematicAnchor, SystematicRatioEvaluator


@dataclass(frozen=True)
class GaussianConstraint:
    """A Gaussian constraint on one scalar model parameter."""

    mean: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("Constraint mean must be finite.")
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("Constraint sigma must be finite and positive.")

    def nll(self, value: float) -> float:
        return 0.5 * ((float(value) - self.mean) / self.sigma) ** 2


@dataclass(frozen=True)
class FitResult:
    """Backend-neutral maximum-likelihood fit result."""

    point: Mapping[str, float]
    nll: float
    success: bool
    message: str
    evaluations: int
    covariance: np.ndarray | None = None
    parameter_names: tuple[str, ...] = ()
    errors: Mapping[str, float] | None = None
    correlation: np.ndarray | None = None
    edm: float | None = None
    valid: bool | None = None
    parameters_at_limit: tuple[str, ...] = ()
    backend: str = "scipy"


@dataclass(frozen=True)
class ProfileScanResult:
    """One-dimensional profile-likelihood scan."""

    parameter: str
    values: np.ndarray
    twice_delta_nll: np.ndarray
    profiled_points: tuple[Mapping[str, float], ...]
    global_fit: FitResult
    fits: tuple[FitResult, ...]


class ExtendedUnbinnedLikelihood:
    """Extended likelihood evaluated from process/reference ratios.

    Terms depending only on the fixed reference density are omitted. For
    weighted observations (including an Asimov quadrature),

    ``NLL(theta) = Lambda(theta) - sum_i w_i log[h(x_i; theta)] + constraints``

    where ``h = lambda / q_reference``. Ratio normalizers must therefore be
    frozen or estimated on an integration sample, never refit as a function
    of the evaluated parameter point.
    """

    def __init__(
        self,
        *,
        intensity: IntensityModel,
        ratios: Mapping[str, np.ndarray],
        event_weights: np.ndarray | None = None,
        normalizer: RatioNormalizer | None = None,
        constraints: Mapping[str, GaussianConstraint | Mapping[str, float]]
        | None = None,
        auxiliary_observations: Mapping[str, float] | None = None,
        systematics: Mapping[str, Sequence[SystematicAnchor]] | None = None,
        integration_weights: np.ndarray | None = None,
        event_values: np.ndarray | None = None,
        fnf_systematics: Mapping[str, Any] | None = None,
    ) -> None:
        self.intensity = intensity
        selected = {
            name: np.asarray(ratios[name], dtype=np.float64).reshape(-1)
            for name in intensity.component_names
            if name in ratios
        }
        missing = set(intensity.component_names).difference(selected)
        if missing:
            raise KeyError(f"Missing observed ratios {sorted(missing)}.")
        lengths = {len(values) for values in selected.values()}
        if len(lengths) != 1:
            raise ValueError("Observed ratio arrays must be aligned.")
        length = next(iter(lengths))
        if any(
            not np.isfinite(values).all() or np.any(values < 0)
            for values in selected.values()
        ):
            raise ValueError("Observed ratios must be finite and non-negative.")
        if normalizer is not None:
            selected = normalizer.normalize(selected)
        self.ratios = selected
        if event_weights is None:
            weights = np.ones(length, dtype=np.float64)
        else:
            weights = np.asarray(event_weights, dtype=np.float64).reshape(-1)
        if (
            len(weights) != length
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
        ):
            raise ValueError("event_weights must align and be finite and non-negative.")
        self.event_weights = weights
        if integration_weights is None:
            integration = np.full(length, 1.0 / length)
        else:
            integration = np.asarray(integration_weights, dtype=np.float64).reshape(-1)
            if (
                len(integration) != length
                or not np.isfinite(integration).all()
                or np.any(integration < 0)
                or not float(np.sum(integration)) > 0
            ):
                raise ValueError(
                    "integration_weights must align and define a finite "
                    "positive measure."
                )
            integration = integration / np.sum(integration)
        self.integration_weights = integration
        if event_values is None:
            parsed_event_values = None
        else:
            parsed_event_values = np.asarray(event_values, dtype=np.float64)
            if (
                parsed_event_values.ndim != 2
                or len(parsed_event_values) != length
                or not np.isfinite(parsed_event_values).all()
            ):
                raise ValueError(
                    "event_values must be a finite two-dimensional array "
                    "aligned with the likelihood support."
                )
        self.event_values = parsed_event_values
        parsed_systematics: dict[str, tuple[SystematicAnchor, ...]] = {}
        known_parameters = {parameter.name for parameter in intensity.parameters}
        for component, anchors in (systematics or {}).items():
            if component not in intensity.component_names:
                raise ValueError(
                    f"Systematics reference unknown component {component!r}."
                )
            parsed = tuple(anchors)
            for anchor in parsed:
                if anchor.component != component:
                    raise ValueError(
                        "Systematic anchor component does not match its map key."
                    )
                if anchor.parameter not in known_parameters:
                    raise ValueError(
                        f"Systematic anchor references unknown parameter "
                        f"{anchor.parameter!r}."
                    )
                if len(anchor.ratio_up) != length:
                    raise ValueError(
                        "Systematic anchor arrays must align with observations."
                    )
            names = [anchor.parameter for anchor in parsed]
            if len(names) != len(set(names)):
                raise ValueError(
                    f"Component {component!r} repeats a systematic parameter."
                )
            parsed_systematics[component] = parsed
        self.systematics = parsed_systematics
        self.systematic_evaluators = {
            component: SystematicRatioEvaluator(
                component=component,
                nominal_process_ratio=self.ratios[component],
                integration_weights=self.integration_weights,
                anchors=anchors,
            )
            for component, anchors in self.systematics.items()
        }
        parsed_fnf = validate_fnf_systematics(
            intensity,
            fnf_systematics,
            anchor_parameters={
                component: tuple(anchor.parameter for anchor in anchors)
                for component, anchors in self.systematics.items()
            },
        )
        if parsed_fnf and self.event_values is None:
            raise ValueError("FNF systematics require aligned event_values.")
        self.fnf_systematics = parsed_fnf
        parsed_constraints: dict[str, GaussianConstraint] = {}
        for name, value in (constraints or {}).items():
            if name not in {item.name for item in intensity.parameters}:
                raise ValueError(f"Constraint references unknown parameter {name!r}.")
            parsed_constraints[name] = (
                value
                if isinstance(value, GaussianConstraint)
                else GaussianConstraint(**value)
            )
        observations = {
            name: float(value) for name, value in (auxiliary_observations or {}).items()
        }
        unknown_observations = set(observations).difference(parsed_constraints)
        if unknown_observations:
            raise ValueError(
                "Auxiliary observations reference unconstrained parameters "
                f"{sorted(unknown_observations)}."
            )
        if not all(np.isfinite(value) for value in observations.values()):
            raise ValueError("Auxiliary observations must be finite.")
        for name, observed in observations.items():
            parsed_constraints[name] = GaussianConstraint(
                mean=observed,
                sigma=parsed_constraints[name].sigma,
            )
        self.constraints = parsed_constraints
        self.auxiliary_observations = {
            name: constraint.mean for name, constraint in parsed_constraints.items()
        }

    @property
    def observed_count(self) -> float:
        return float(np.sum(self.event_weights))

    def data_nll(self, point: Mapping[str, float]) -> float:
        """Return the extended data term without auxiliary constraints."""

        point = self.intensity.validate_point(point)
        nominal_yields = self.intensity.component_yields(point)
        component_yields: dict[str, float] = {}
        differential = np.zeros_like(self.event_weights, dtype=np.float64)
        for component in self.intensity.component_names:
            component_yield = nominal_yields[component]
            shape = self.ratios[component]
            evaluator = self.systematic_evaluators.get(component)
            if evaluator is not None:
                try:
                    evaluation = evaluator.evaluate(point)
                except (KeyError, ValueError):
                    return float("inf")
                component_yield *= evaluation.yield_factor
                shape = evaluation.shape_ratio
            fnf = self.fnf_systematics.get(component)
            if fnf is not None:
                try:
                    evaluation = evaluate_fnf_on_support(
                        fnf,
                        values=self.event_values,
                        point=point,
                        nominal_shape=shape,
                        integration_weights=self.integration_weights,
                    )
                    shape = evaluation.shape_ratio
                    component_yield *= evaluation.yield_factor
                except (KeyError, ValueError, FloatingPointError, OverflowError):
                    return float("inf")
            component_yields[component] = component_yield
            differential += component_yield * shape
        expected = float(sum(component_yields.values()))
        if not np.isfinite(expected) or expected < 0:
            return float("inf")
        if not np.isfinite(differential).all() or np.any(differential < 0):
            return float("inf")
        active = self.event_weights > 0
        if np.any(differential[active] <= 0):
            return float("inf")
        value = expected - float(
            np.sum(self.event_weights[active] * np.log(differential[active]))
        )
        return float(value)

    def constraint_nll(self, point: Mapping[str, float]) -> float:
        """Return the auxiliary-observation contribution only."""

        validated = self.intensity.validate_point(point)
        return float(
            sum(
                constraint.nll(validated[name])
                for name, constraint in self.constraints.items()
            )
        )

    def nll(self, point: Mapping[str, float]) -> float:
        """Return the full data-plus-constraint negative log likelihood."""

        data = self.data_nll(point)
        if not np.isfinite(data):
            return float("inf")
        return data + self.constraint_nll(point)

    def with_auxiliary_observations(
        self,
        observations: Mapping[str, float],
    ) -> ExtendedUnbinnedLikelihood:
        """Clone the model with shifted global observations.

        This is the primitive used by preferred impact fits. It changes only
        the auxiliary data; every nuisance and POI remains free in the refit.
        """

        unknown = set(observations).difference(self.constraints)
        if unknown:
            raise ValueError(
                f"Auxiliary observations reference unknown constraints "
                f"{sorted(unknown)}."
            )
        complete = dict(self.auxiliary_observations)
        complete.update({name: float(value) for name, value in observations.items()})
        return ExtendedUnbinnedLikelihood(
            intensity=self.intensity,
            ratios=self.ratios,
            event_weights=self.event_weights,
            constraints={
                name: GaussianConstraint(
                    mean=constraint.mean,
                    sigma=constraint.sigma,
                )
                for name, constraint in self.constraints.items()
            },
            auxiliary_observations=complete,
            systematics=self.systematics,
            integration_weights=self.integration_weights,
            event_values=self.event_values,
            fnf_systematics=self.fnf_systematics,
        )

    def fit(
        self,
        *,
        initial: Mapping[str, float] | None = None,
        fixed: Mapping[str, float] | None = None,
        backend: str = "scipy",
        use_jax: bool = True,
        method: str = "L-BFGS-B",
        options: Mapping[str, Any] | None = None,
    ) -> FitResult:
        """Minimize the NLL with native Minuit/JAX or the SciPy fallback."""

        if backend == "minuit":
            from .inference import MinuitInference

            if method != "L-BFGS-B" or options:
                raise ValueError(
                    "SciPy method/options cannot be combined with backend='minuit'."
                )
            return MinuitInference(self, use_jax=use_jax).fit(
                initial=initial,
                fixed=fixed,
            )
        if backend != "scipy":
            raise ValueError("backend must be 'minuit' or 'scipy'.")

        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise ImportError(
                "Fits require SciPy; install hnsbi-toolkit[plots] or "
                "hnsbi-toolkit[bayes]."
            ) from exc
        fixed_point = {name: float(value) for name, value in (fixed or {}).items()}
        known = {parameter.name for parameter in self.intensity.parameters}
        unknown = set(fixed_point).difference(known)
        if unknown:
            raise ValueError(f"Fixed point has unknown parameters {sorted(unknown)}.")
        seed = dict(self.intensity.nominal_point)
        seed.update(initial or {})
        seed.update(fixed_point)
        self.intensity.validate_point(seed)
        free = [
            parameter
            for parameter in self.intensity.parameters
            if parameter.name not in fixed_point
        ]
        if not free:
            value = self.nll(seed)
            return FitResult(
                point=seed,
                nll=value,
                success=np.isfinite(value),
                message="All parameters fixed.",
                evaluations=1,
            )
        x0 = np.asarray([seed[item.name] for item in free], dtype=np.float64)
        bounds = [item.bounds for item in free]

        def objective(vector: np.ndarray) -> float:
            point = dict(fixed_point)
            point.update(
                {
                    parameter.name: float(value)
                    for parameter, value in zip(free, vector, strict=True)
                }
            )
            return self.nll(point)

        result = minimize(
            objective,
            x0,
            method=method,
            bounds=bounds,
            options=dict(options or {}),
        )
        point = dict(fixed_point)
        point.update(
            {
                parameter.name: float(value)
                for parameter, value in zip(free, result.x, strict=True)
            }
        )
        covariance: np.ndarray | None = None
        inverse_hessian = getattr(result, "hess_inv", None)
        if inverse_hessian is not None:
            try:
                covariance = np.asarray(inverse_hessian.todense())
            except AttributeError:
                try:
                    covariance = np.asarray(inverse_hessian)
                except (TypeError, ValueError):
                    covariance = None
        return FitResult(
            point=point,
            nll=float(result.fun),
            success=bool(result.success),
            message=str(result.message),
            evaluations=int(result.nfev),
            covariance=covariance,
        )

    def profile_scan(
        self,
        parameter: str,
        values: Sequence[float] | np.ndarray,
        *,
        initial: Mapping[str, float] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> ProfileScanResult:
        """Profile every other parameter at each fixed scan value.

        The scan raises when the global or any fixed-point minimization fails,
        so unsuccessful optimizer output cannot masquerade as a valid
        likelihood-ratio point.
        """

        known = {item.name for item in self.intensity.parameters}
        if parameter not in known:
            raise ValueError(f"Unknown scan parameter {parameter!r}.")
        grid = np.asarray(values, dtype=np.float64).reshape(-1)
        if not len(grid) or not np.isfinite(grid).all():
            raise ValueError("Scan values must be a non-empty finite sequence.")
        parameter_spec = next(
            item for item in self.intensity.parameters if item.name == parameter
        )
        if parameter_spec.bounds is not None:
            low, high = parameter_spec.bounds
            if np.any((grid < low) | (grid > high)):
                raise ValueError(f"Scan values leave declared bounds [{low}, {high}].")
        global_fit = self.fit(initial=initial, options=options)
        if not global_fit.success or not np.isfinite(global_fit.nll):
            raise RuntimeError(
                "Global fit failed before profile scan: "
                f"{global_fit.message} (NLL={global_fit.nll!r})."
            )
        seed = dict(global_fit.point)
        fits: list[FitResult] = []
        for value in grid:
            fit = self.fit(
                initial=seed,
                fixed={parameter: float(value)},
                options=options,
            )
            if not fit.success or not np.isfinite(fit.nll):
                raise RuntimeError(
                    f"Profile fit failed for {parameter}={float(value)!r}: "
                    f"{fit.message} (NLL={fit.nll!r})."
                )
            fits.append(fit)
            seed.update(fit.point)
        delta = np.asarray(
            [2.0 * (fit.nll - global_fit.nll) for fit in fits],
            dtype=np.float64,
        )
        delta = np.maximum(delta, 0.0)
        return ProfileScanResult(
            parameter=parameter,
            values=grid,
            twice_delta_nll=delta,
            profiled_points=tuple(fit.point for fit in fits),
            global_fit=global_fit,
            fits=tuple(fits),
        )

    @classmethod
    def from_workspace(
        cls,
        path: str | Path,
        *,
        constraints: Mapping[str, GaussianConstraint | Mapping[str, float]]
        | None = None,
        verify_artifacts: bool = True,
        fnf_systematics: Mapping[str, Any] | None = None,
        fnf_device: str = "cpu",
        onnx_providers: tuple[str, ...] | None = None,
    ) -> ExtendedUnbinnedLikelihood:
        """Load a workspace written by :func:`hnsbi.workspace.write_workspace`.

        This loader covers hNSBI formulas, Gaussian constraints, normalized
        ``normplusshape`` interpolation, FNF morphs, and checked portable
        arrays. FNF artifacts are bound automatically from the reference-flow
        and ratio manifests. Explicit FNF overrides are rejected because their
        nominal process density cannot be authenticated against the workspace.
        """

        workspace_path = Path(path)
        with workspace_path.open(encoding="utf-8") as stream:
            workspace = json.load(stream)
        channels = workspace.get("channels", [])
        if len(channels) != 1 or channels[0].get("type") != "unbinned":
            raise ValueError(
                "Formula likelihood loading currently expects one unbinned channel."
            )
        measurements = workspace.get("measurements", [])
        if len(measurements) != 1:
            raise ValueError("Workspace must contain exactly one measurement.")
        extension = workspace.get("hnsbi", {})
        auxiliary_payload = (
            extension.get("auxiliary_observations", {})
            if isinstance(extension, Mapping)
            else {}
        )
        if not isinstance(auxiliary_payload, Mapping):
            raise ValueError("Workspace auxiliary observations must be an object.")
        auxiliary_observations = {
            str(name): float(value) for name, value in auxiliary_payload.items()
        }
        if not all(np.isfinite(value) for value in auxiliary_observations.values()):
            raise ValueError("Workspace auxiliary observations must be finite.")
        parameter_entries = measurements[0]["config"].get("parameters", [])
        parameters = []
        embedded_constraints: dict[str, GaussianConstraint] = {}
        for entry in parameter_entries:
            bounds = entry.get("bounds")
            if bounds is not None and len(bounds) == 1 and isinstance(bounds[0], list):
                bounds = bounds[0]
            constraint = entry.get("hnsbi_constraint")
            if constraint is not None:
                embedded_constraints[entry["name"]] = GaussianConstraint(
                    mean=float(constraint.get("mean", 0.0)),
                    sigma=float(constraint.get("sigma", 1.0)),
                )
            parameters.append(
                Parameter(
                    name=entry["name"],
                    nominal=float(
                        (
                            extension.get("parameter_nominals", {})
                            if isinstance(extension, Mapping)
                            else {}
                        ).get(
                            entry["name"],
                            entry.get(
                                "nominal",
                                entry.get("initial", entry.get("inits", [0.0])[0]),
                            ),
                        )
                    ),
                    bounds=(tuple(map(float, bounds)) if bounds is not None else None),
                    constrained=(
                        constraint is not None or entry["name"] in (constraints or {})
                    ),
                    constraint_mean=float((constraint or {}).get("mean", 0.0)),
                    constraint_sigma=float((constraint or {}).get("sigma", 1.0)),
                )
            )
        if isinstance(extension, Mapping) and "auxiliary_observations" in extension:
            constrained_names = set(embedded_constraints)
            if set(auxiliary_observations) != constrained_names:
                raise ValueError(
                    "Workspace auxiliary-observation keys must match every "
                    "constrained parameter exactly."
                )
            asimov_point = extension.get("asimov_point")
            if not isinstance(asimov_point, Mapping):
                raise ValueError(
                    "Workspace with auxiliary observations has no Asimov point."
                )
            for name, observed in auxiliary_observations.items():
                if name not in asimov_point or not np.isclose(
                    observed,
                    float(asimov_point[name]),
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise ValueError(
                        "Workspace auxiliary observations must equal the "
                        "constrained Asimov generating values."
                    )
        components = []
        ratios = {}
        base = workspace_path.parent
        manifest_value = (
            extension.get("array_manifest") if isinstance(extension, Mapping) else None
        )
        recorded_paths: dict[str, Path] = {}
        array_manifest_metadata: dict[str, Any] = {}
        if manifest_value is None:
            if verify_artifacts:
                raise ValueError(
                    "Workspace has no checksummed hnsbi array manifest. "
                    "Pass verify_artifacts=False only for a trusted legacy "
                    "workspace."
                )
        else:
            manifest_path = Path(manifest_value)
            if not manifest_path.is_absolute():
                manifest_path = base / manifest_path
            manifest = ArtifactManifest.load(manifest_path)
            if manifest.artifact_type != "asimov-array-bundle":
                raise ValueError(
                    "Workspace array manifest has unexpected artifact type "
                    f"{manifest.artifact_type!r}."
                )
            if verify_artifacts:
                manifest.verify(manifest_path.parent)
            array_manifest_metadata = dict(manifest.metadata)
            for record in manifest.files:
                if record.kind in recorded_paths:
                    raise ValueError(
                        f"Workspace array manifest repeats role {record.kind!r}."
                    )
                recorded_paths[record.kind] = (
                    manifest_path.parent / record.path
                ).resolve()

        def resolve_array(value: str, role: str) -> Path:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = base / candidate
            candidate = candidate.resolve()
            if recorded_paths:
                recorded = recorded_paths.get(role)
                if recorded is None:
                    raise ValueError(
                        f"Workspace array manifest is missing role {role!r}."
                    )
                if candidate != recorded:
                    raise ValueError(
                        f"Workspace path for {role!r} does not match its "
                        "checksummed manifest."
                    )
            return candidate

        systematics: dict[str, list[SystematicAnchor]] = {}
        for sample in channels[0].get("samples", []):
            multiplier = sample.get("hnsbi", {}).get("multiplier", "1")
            components.append(
                Component(
                    name=sample["name"],
                    nominal_yield=float(sample["data"][0]),
                    multiplier=multiplier,
                )
            )
            ratio_path = resolve_array(
                sample["ratios"], f"process-ratio:{sample['name']}"
            )
            ratios[sample["name"]] = np.load(ratio_path, allow_pickle=False)
            component_systematics: list[SystematicAnchor] = []
            for modifier in sample.get("modifiers", []):
                modifier_type = modifier.get("type")
                if modifier_type == "normfactor":
                    continue
                if modifier_type != "normplusshape":
                    raise ValueError(
                        "Formula likelihood cannot interpret modifier type "
                        f"{modifier_type!r}."
                    )
                data = modifier.get("data")
                if not isinstance(data, Mapping):
                    raise ValueError("normplusshape modifier has no data object.")
                extension_modifier = modifier.get("hnsbi")
                if verify_artifacts and (
                    not isinstance(extension_modifier, Mapping)
                    or "manifest" not in extension_modifier
                ):
                    raise ValueError("Systematic modifier has no integrity manifest.")
                systematic_paths: dict[str, Path] = {}
                for workspace_role, manifest_role in (
                    ("hi_ratio", "up-ratio"),
                    ("lo_ratio", "down-ratio"),
                ):
                    candidate = Path(data[workspace_role])
                    if not candidate.is_absolute():
                        candidate = base / candidate
                    systematic_paths[manifest_role] = candidate.resolve()
                if isinstance(extension_modifier, Mapping) and (
                    "manifest" in extension_modifier
                ):
                    systematic_manifest_path = Path(extension_modifier["manifest"])
                    if not systematic_manifest_path.is_absolute():
                        systematic_manifest_path = base / systematic_manifest_path
                    systematic_manifest = ArtifactManifest.load(
                        systematic_manifest_path
                    )
                    if systematic_manifest.artifact_type != "systematic-anchor":
                        raise ValueError(
                            "Unexpected systematic manifest artifact type."
                        )
                    if verify_artifacts:
                        systematic_manifest.verify(systematic_manifest_path.parent)
                    roles = {
                        record.kind: (
                            systematic_manifest_path.parent / record.path
                        ).resolve()
                        for record in systematic_manifest.files
                    }
                    if set(roles) != {"up-ratio", "down-ratio"}:
                        raise ValueError(
                            "Systematic manifest must contain exactly the "
                            "up-ratio and down-ratio roles."
                        )
                    if roles.get("up-ratio") != systematic_paths["up-ratio"]:
                        raise ValueError(
                            "Systematic hi_ratio does not match its manifest."
                        )
                    if roles.get("down-ratio") != systematic_paths["down-ratio"]:
                        raise ValueError(
                            "Systematic lo_ratio does not match its manifest."
                        )
                hi_data = np.asarray(data.get("hi_data"), dtype=np.float64).reshape(-1)
                lo_data = np.asarray(data.get("lo_data"), dtype=np.float64).reshape(-1)
                if len(hi_data) != 1 or len(lo_data) != 1:
                    raise ValueError("Systematic yield anchors must be scalar.")
                interpolation = (
                    extension_modifier.get("interpolation", "nsbi_code4p")
                    if isinstance(extension_modifier, Mapping)
                    else "nsbi_code4p"
                )
                if (
                    verify_artifacts
                    and isinstance(extension_modifier, Mapping)
                    and "manifest" in extension_modifier
                ):
                    expected_systematic_metadata = {
                        "component": sample["name"],
                        "parameter": modifier["name"],
                        "rows": len(ratios[sample["name"]]),
                        "interpolation": interpolation,
                    }
                    for key, expected in expected_systematic_metadata.items():
                        if systematic_manifest.metadata.get(key) != expected:
                            raise ValueError(
                                "Systematic manifest metadata "
                                f"{key!r} does not match the workspace."
                            )
                    for key, expected in (
                        ("yield_up", float(hi_data[0])),
                        ("yield_down", float(lo_data[0])),
                    ):
                        if not np.isclose(
                            float(systematic_manifest.metadata.get(key, np.nan)),
                            expected,
                        ):
                            raise ValueError(
                                "Systematic manifest metadata "
                                f"{key!r} does not match the workspace."
                            )
                component_systematics.append(
                    SystematicAnchor(
                        parameter=modifier["name"],
                        component=sample["name"],
                        ratio_up=np.load(
                            systematic_paths["up-ratio"],
                            allow_pickle=False,
                        ),
                        ratio_down=np.load(
                            systematic_paths["down-ratio"],
                            allow_pickle=False,
                        ),
                        yield_up=float(hi_data[0]),
                        yield_down=float(lo_data[0]),
                        interpolation=interpolation,
                    )
                )
            if component_systematics:
                systematics[sample["name"]] = component_systematics
        if recorded_paths:
            expected_roles = {
                "event-values",
                "event-weights",
                "reference-integration-weights",
                "metadata",
                *(f"process-ratio:{component.name}" for component in components),
            }
            if set(recorded_paths) != expected_roles:
                raise ValueError(
                    "Workspace array manifest roles do not match the model; "
                    f"expected={sorted(expected_roles)}, "
                    f"found={sorted(recorded_paths)}."
                )
            if tuple(array_manifest_metadata.get("features", ())) != tuple(
                extension.get("features", ())
            ):
                raise ValueError(
                    "Asimov array manifest feature order does not match the workspace."
                )
            if tuple(array_manifest_metadata.get("samples", ())) != tuple(
                component.name for component in components
            ):
                raise ValueError(
                    "Asimov array manifest samples do not match the workspace."
                )
            metadata_path = recorded_paths["metadata"]
            try:
                with metadata_path.open(encoding="utf-8") as stream:
                    asimov_metadata = json.load(stream)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Could not load the Asimov array metadata.") from exc
            if not isinstance(asimov_metadata, Mapping):
                raise ValueError("Asimov array metadata must be an object.")
            if asimov_metadata.get("intensity_fingerprint") != extension.get(
                "intensity_fingerprint"
            ):
                raise ValueError("Asimov metadata belongs to a different intensity.")
            if asimov_metadata.get("point") != extension.get("asimov_point"):
                raise ValueError("Asimov metadata point does not match the workspace.")
            if asimov_metadata.get("ratio_normalizers") != extension.get(
                "ratio_normalization"
            ):
                raise ValueError("Asimov ratio normalizers do not match the workspace.")
            if asimov_metadata.get("auxiliary_observations", {}) != (
                extension.get("auxiliary_observations", {})
            ):
                raise ValueError(
                    "Asimov auxiliary observations do not match the workspace."
                )
            if set(asimov_metadata.get("fnf_components", ())) != set(
                extension.get("asimov_fnf_components", ())
            ):
                raise ValueError("Asimov FNF provenance does not match the workspace.")
        extension_normalizers = extension.get("ratio_normalization", {})
        if not isinstance(extension_normalizers, Mapping) or {
            component.name for component in components
        } != set(extension_normalizers):
            raise ValueError(
                "Workspace ratio-normalization keys do not match its samples."
            )
        for component in components:
            sample_normalizer = next(
                sample["hnsbi"]["ratio_normalizer"]
                for sample in channels[0].get("samples", [])
                if sample["name"] == component.name
            )
            if not np.isclose(
                float(sample_normalizer),
                float(extension_normalizers[component.name]),
            ):
                raise ValueError(
                    f"Ratio normalizer for {component.name!r} is inconsistent."
                )
        weights_path = resolve_array(channels[0]["weights"], "event-weights")
        values_payload = channels[0].get("values")
        if values_payload is None:
            event_values = None
        else:
            values_path = resolve_array(values_payload, "event-values")
            event_values = np.load(values_path, allow_pickle=False)
        reference_weights_value = (
            extension.get("reference_weights")
            if isinstance(extension, Mapping)
            else None
        )
        if reference_weights_value is None:
            if verify_artifacts:
                raise ValueError("Workspace has no reference integration weights.")
            integration_weights = None
        else:
            reference_weights_path = resolve_array(
                reference_weights_value,
                "reference-integration-weights",
            )
            integration_weights = np.load(reference_weights_path, allow_pickle=False)
        embedded_constraints.update(constraints or {})
        intensity = IntensityModel(components, parameters)
        expected_fingerprint = (
            extension.get("intensity_fingerprint")
            if isinstance(extension, Mapping)
            else None
        )
        if expected_fingerprint != intensity.fingerprint:
            raise ValueError(
                "Workspace intensity does not match its recorded scientific "
                "fingerprint."
            )
        if array_manifest_metadata:
            if (
                array_manifest_metadata.get("intensity_fingerprint")
                != expected_fingerprint
            ):
                raise ValueError(
                    "Asimov array manifest belongs to a different intensity."
                )
            expected_rows = array_manifest_metadata.get("rows")
            if expected_rows is not None and int(expected_rows) != len(
                next(iter(ratios.values()))
            ):
                raise ValueError(
                    "Asimov array manifest row count does not match arrays."
                )
        configured_fnf = {
            sample["name"]
            for sample in channels[0].get("samples", ())
            if isinstance(sample.get("hnsbi"), Mapping)
            and "fnf_manifest" in sample["hnsbi"]
        }
        if configured_fnf != set(extension.get("fnf_components", ())):
            raise ValueError(
                "Workspace FNF component metadata does not match its samples."
            )
        if fnf_systematics is None and configured_fnf:
            from .fnf_runtime import load_workspace_fnf_systematics
            from .workspace import load_workspace_model

            fnf_systematics = load_workspace_fnf_systematics(
                load_workspace_model(workspace_path),
                device=fnf_device,
                providers=onnx_providers,
            )
        elif fnf_systematics is not None:
            if configured_fnf or fnf_systematics:
                raise ValueError(
                    "Explicit FNF runtime overrides are not accepted for "
                    "workspace loading; omit fnf_systematics so the checked "
                    "reference, ratio, and residual manifests are reconstructed."
                )
            fnf_systematics = {}
        return cls(
            intensity=intensity,
            ratios=ratios,
            event_weights=np.load(weights_path, allow_pickle=False),
            constraints=embedded_constraints,
            auxiliary_observations=auxiliary_observations,
            systematics=systematics,
            integration_weights=integration_weights,
            event_values=event_values,
            fnf_systematics=fnf_systematics,
        )
