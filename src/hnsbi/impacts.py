"""Pulls and uncertainty impacts for constrained likelihood fits.

The preferred refit-based impact changes an auxiliary (global) observation and
leaves every model parameter floating.  It is intentionally different from the
legacy procedure that shifts and fixes a nuisance parameter.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpy as np


def _json_compatible(value: Any) -> Any:
    if is_dataclass(value):
        return _json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class _JsonResult:
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return _json_compatible(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the result without backend-specific objects."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class PullEntry(_JsonResult):
    """One constrained-parameter pull."""

    name: str
    fitted_value: float
    prefit_mean: float
    prefit_sigma: float
    pull: float
    postfit_sigma: float
    postfit_over_prefit: float


@dataclass(frozen=True)
class PullResult(_JsonResult):
    """Pulls from one maximum-likelihood fit."""

    entries: tuple[PullEntry, ...]
    baseline_point: Mapping[str, float]
    baseline_nll: float
    fit_success: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImpactEntry(_JsonResult):
    """Signed POI response associated with one constrained nuisance."""

    name: str
    up: float | None
    down: float | None
    magnitude: float | None
    up_fit_success: bool | None = None
    down_fit_success: bool | None = None
    up_point: Mapping[str, float] | None = None
    down_point: Mapping[str, float] | None = None


@dataclass(frozen=True)
class ImpactResult(_JsonResult):
    """A complete POI impact decomposition."""

    poi: str
    method: str
    entries: tuple[ImpactEntry, ...]
    baseline_point: Mapping[str, float]
    baseline_nll: float
    total: float | None
    groups: Mapping[str, float | None]
    statistical: float | None = None
    step: float | None = None
    warnings: tuple[str, ...] = ()


def _fit_point(fit: Any) -> dict[str, float]:
    point = getattr(fit, "point", None)
    if point is None:
        names = getattr(fit, "parameter_names", None)
        values = getattr(fit, "values", None)
        if names is not None and values is not None:
            point = dict(zip(names, values, strict=True))
    if not isinstance(point, Mapping):
        raise TypeError("Fit result must expose a point mapping.")
    parsed = {str(name): float(value) for name, value in point.items()}
    if not all(np.isfinite(value) for value in parsed.values()):
        raise ValueError("Fit point contains non-finite values.")
    return parsed


def _fit_nll(fit: Any) -> float:
    if hasattr(fit, "nll"):
        return float(fit.nll)
    if hasattr(fit, "twice_nll"):
        return 0.5 * float(fit.twice_nll)
    return float("nan")


def _fit_success(fit: Any) -> bool:
    return bool(getattr(fit, "success", True))


def _constraint_mean_sigma(constraint: Any) -> tuple[float, float]:
    if isinstance(constraint, Mapping):
        mean = constraint.get("mean", constraint.get("constraint_mean", 0.0))
        sigma = constraint.get("sigma", constraint.get("constraint_sigma", 1.0))
    else:
        mean = getattr(
            constraint,
            "mean",
            getattr(constraint, "constraint_mean", 0.0),
        )
        sigma = getattr(
            constraint,
            "sigma",
            getattr(constraint, "constraint_sigma", 1.0),
        )
    mean = float(mean)
    sigma = float(sigma)
    if not np.isfinite(mean) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Constraint means must be finite and sigmas positive.")
    return mean, sigma


def _constraints(likelihood: Any) -> dict[str, Any]:
    constraints = getattr(likelihood, "constraints", None)
    if not isinstance(constraints, Mapping):
        raise TypeError("Likelihood must expose a constraints mapping.")
    return {str(name): constraint for name, constraint in constraints.items()}


def _parameter_order(likelihood: Any, fit: Any) -> tuple[str, ...]:
    for attribute in ("parameter_names", "covariance_order"):
        names = getattr(fit, attribute, None)
        if names is not None:
            parsed = tuple(str(name) for name in names)
            if parsed:
                return parsed
    intensity = getattr(likelihood, "intensity", None)
    parameters = getattr(intensity, "parameters", None)
    if parameters is not None:
        parsed = tuple(str(parameter.name) for parameter in parameters)
        if parsed:
            return parsed
    return tuple(_fit_point(fit))


def _covariance(likelihood: Any, fit: Any) -> tuple[tuple[str, ...], np.ndarray]:
    covariance = getattr(fit, "covariance", None)
    if covariance is None:
        correlation = getattr(fit, "correlation", None)
        errors = getattr(fit, "errors", None)
        if correlation is None or errors is None:
            raise ValueError(
                "The fit has no covariance matrix. Use a fit backend that runs HESSE."
            )
        names = _parameter_order(likelihood, fit)
        if isinstance(errors, Mapping):
            error_array = np.asarray(
                [errors[name] for name in names],
                dtype=np.float64,
            )
        else:
            error_array = np.asarray(errors, dtype=np.float64).reshape(-1)
        correlation_array = np.asarray(correlation, dtype=np.float64)
        if len(error_array) != len(names):
            raise ValueError("Fit errors do not match the parameter order.")
        covariance = correlation_array * np.outer(error_array, error_array)
    names = _parameter_order(likelihood, fit)
    if isinstance(covariance, Mapping):
        try:
            matrix = np.asarray(
                [[covariance[row][column] for column in names] for row in names],
                dtype=np.float64,
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Named covariance does not cover every parameter."
            ) from exc
    else:
        matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (len(names), len(names)):
        raise ValueError(
            "Covariance shape does not match the inferred parameter order: "
            f"{matrix.shape} versus {len(names)} parameters."
        )
    if len(set(names)) != len(names):
        raise ValueError("Covariance parameter names must be unique.")
    if not np.isfinite(matrix).all():
        raise ValueError("Covariance contains non-finite values.")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 1.0e-8 * scale
    if not np.allclose(matrix, matrix.T, rtol=1.0e-7, atol=tolerance):
        raise ValueError("Covariance matrix is not symmetric.")
    matrix = 0.5 * (matrix + matrix.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    if minimum_eigenvalue < -tolerance:
        raise ValueError("Covariance matrix is not positive semidefinite.")
    return names, matrix


def _selected_nuisances(
    constraints: Mapping[str, Any],
    *,
    poi: str | None,
    parameters: Sequence[str] | None,
) -> tuple[str, ...]:
    if parameters is None:
        selected = tuple(name for name in constraints if name != poi)
    else:
        selected = tuple(str(name) for name in parameters)
    if len(set(selected)) != len(selected):
        raise ValueError("Nuisance parameter names must be unique.")
    unknown = set(selected).difference(constraints)
    if unknown:
        raise ValueError(f"Unknown constrained parameters {sorted(unknown)}.")
    if poi is not None and poi in selected:
        raise ValueError("The POI cannot also be ranked as a nuisance impact.")
    return selected


def _group_magnitudes(
    entries: Sequence[ImpactEntry],
    groups: Mapping[str, Sequence[str]] | None,
) -> dict[str, float | None]:
    by_name = {entry.name: entry for entry in entries}
    output: dict[str, float | None] = {}
    for group, members in (groups or {}).items():
        names = tuple(str(name) for name in members)
        unknown = set(names).difference(by_name)
        if unknown:
            raise ValueError(
                f"Impact group {group!r} references unknown nuisances "
                f"{sorted(unknown)}."
            )
        magnitudes = [by_name[name].magnitude for name in names]
        output[str(group)] = (
            None
            if any(value is None for value in magnitudes)
            else math.sqrt(sum(float(value) ** 2 for value in magnitudes))
        )
    return output


def compute_pulls(
    likelihood: Any,
    fit: Any | None = None,
    *,
    parameters: Sequence[str] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> PullResult:
    """Compute normalized pulls for Gaussian-constrained nuisances.

    The pull is ``(theta_hat - t) / sigma_t`` and its post-fit error is
    ``sqrt(V_kk) / sigma_t``. Unconstrained parameters are not included.
    """

    constraints = _constraints(likelihood)
    selected = _selected_nuisances(
        constraints,
        poi=None,
        parameters=parameters,
    )
    if fit is None:
        fit = likelihood.fit(**dict(fit_kwargs or {}))
    point = _fit_point(fit)
    names, covariance = _covariance(likelihood, fit)
    index = {name: position for position, name in enumerate(names)}
    warnings: list[str] = []
    if not _fit_success(fit):
        warnings.append("The baseline fit did not report success.")
    entries: list[PullEntry] = []
    for name in selected:
        if name not in point or name not in index:
            raise ValueError(
                f"Fit result does not contain constrained parameter {name!r}."
            )
        mean, sigma = _constraint_mean_sigma(constraints[name])
        variance = float(covariance[index[name], index[name]])
        if variance < 0:
            variance = 0.0
            warnings.append(
                f"Post-fit variance for {name!r} was clipped to zero at tolerance."
            )
        postfit = math.sqrt(variance)
        entries.append(
            PullEntry(
                name=name,
                fitted_value=point[name],
                prefit_mean=mean,
                prefit_sigma=sigma,
                pull=(point[name] - mean) / sigma,
                postfit_sigma=postfit,
                postfit_over_prefit=postfit / sigma,
            )
        )
    return PullResult(
        entries=tuple(entries),
        baseline_point=point,
        baseline_nll=_fit_nll(fit),
        fit_success=_fit_success(fit),
        warnings=tuple(warnings),
    )


def _clone_with_auxiliary_observations(
    likelihood: Any,
    observations: Mapping[str, float],
) -> Any:
    method = getattr(likelihood, "with_auxiliary_observations", None)
    if callable(method):
        return method(dict(observations))

    from .likelihood import ExtendedUnbinnedLikelihood, GaussianConstraint

    if not isinstance(likelihood, ExtendedUnbinnedLikelihood):
        raise TypeError(
            "A likelihood without with_auxiliary_observations() must be an "
            "ExtendedUnbinnedLikelihood."
        )
    constraints = {
        name: GaussianConstraint(mean=mean, sigma=_constraint_mean_sigma(value)[1])
        for name, value in likelihood.constraints.items()
        for mean in [float(observations.get(name, _constraint_mean_sigma(value)[0]))]
    }
    return ExtendedUnbinnedLikelihood(
        intensity=likelihood.intensity,
        ratios=likelihood.ratios,
        event_weights=likelihood.event_weights,
        constraints=constraints,
        auxiliary_observations=observations,
        systematics=likelihood.systematics,
        integration_weights=likelihood.integration_weights,
    )


def global_observable_impacts(
    likelihood: Any,
    poi: str,
    *,
    fit: Any | None = None,
    parameters: Sequence[str] | None = None,
    step: float = 1.0,
    groups: Mapping[str, Sequence[str]] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> ImpactResult:
    """Rank impacts by shifting global observations and profiling all parameters.

    For each nuisance ``k``, the auxiliary observation is shifted to
    ``t_k +/- step * sigma_tk``. No nuisance parameter is fixed in either
    refit. Returned ``up`` and ``down`` values are signed POI displacements.
    """

    step = float(step)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be finite and positive.")
    kwargs = dict(fit_kwargs or {})
    if kwargs.get("fixed"):
        raise ValueError(
            "Global-observable impacts require all parameters to float; "
            "fit_kwargs cannot contain fixed parameters."
        )
    baseline_initial = kwargs.pop("initial", None)
    if fit is None:
        baseline_kwargs = dict(kwargs)
        if baseline_initial is not None:
            baseline_kwargs["initial"] = baseline_initial
        fit = likelihood.fit(**baseline_kwargs)
    baseline = _fit_point(fit)
    if poi not in baseline:
        raise ValueError(f"Fit result does not contain POI {poi!r}.")
    if not _fit_success(fit):
        raise RuntimeError("Cannot compute impacts from an unsuccessful baseline fit.")

    constraints = _constraints(likelihood)
    selected = _selected_nuisances(
        constraints,
        poi=poi,
        parameters=parameters,
    )
    auxiliary = {
        name: _constraint_mean_sigma(value)[0] for name, value in constraints.items()
    }
    auxiliary.update(
        {
            str(name): float(value)
            for name, value in getattr(
                likelihood,
                "auxiliary_observations",
                {},
            ).items()
        }
    )
    entries: list[ImpactEntry] = []
    warnings: list[str] = []
    for name in selected:
        _, sigma = _constraint_mean_sigma(constraints[name])
        fits: dict[str, Any] = {}
        for direction, sign in (("up", 1.0), ("down", -1.0)):
            shifted = dict(auxiliary)
            shifted[name] = auxiliary[name] + sign * step * sigma
            shifted_likelihood = _clone_with_auxiliary_observations(
                likelihood,
                shifted,
            )
            fits[direction] = shifted_likelihood.fit(
                initial=baseline,
                **kwargs,
            )
        up_fit = fits["up"]
        down_fit = fits["down"]
        up_success = _fit_success(up_fit)
        down_success = _fit_success(down_fit)
        up_point = _fit_point(up_fit) if up_success else None
        down_point = _fit_point(down_fit) if down_success else None
        up = None if up_point is None else up_point[poi] - baseline[poi]
        down = None if down_point is None else down_point[poi] - baseline[poi]
        magnitude = None if up is None or down is None else 0.5 * (abs(up) + abs(down))
        if not up_success or not down_success:
            warnings.append(f"At least one shifted fit failed for {name!r}.")
        entries.append(
            ImpactEntry(
                name=name,
                up=up,
                down=down,
                magnitude=magnitude,
                up_fit_success=up_success,
                down_fit_success=down_success,
                up_point=up_point,
                down_point=down_point,
            )
        )
    magnitudes = [entry.magnitude for entry in entries]
    total = (
        None
        if any(value is None for value in magnitudes)
        else math.sqrt(sum(float(value) ** 2 for value in magnitudes))
    )
    return ImpactResult(
        poi=poi,
        method="global_observable",
        entries=tuple(entries),
        baseline_point=baseline,
        baseline_nll=_fit_nll(fit),
        total=total,
        groups=_group_magnitudes(entries, groups),
        step=step,
        warnings=tuple(warnings),
    )


def covariance_impacts(
    likelihood: Any,
    poi: str,
    *,
    fit: Any | None = None,
    parameters: Sequence[str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> ImpactResult:
    """Compute Gaussian impacts from the post-fit covariance matrix.

    For an independent Gaussian auxiliary observation with width ``sigma_t``,
    a one-standard-deviation signed POI response is ``V[poi, k] / sigma_t``.
    """

    if fit is None:
        fit = likelihood.fit(**dict(fit_kwargs or {}))
    baseline = _fit_point(fit)
    if poi not in baseline:
        raise ValueError(f"Fit result does not contain POI {poi!r}.")
    names, covariance = _covariance(likelihood, fit)
    index = {name: position for position, name in enumerate(names)}
    if poi not in index:
        raise ValueError(f"Covariance does not contain POI {poi!r}.")
    constraints = _constraints(likelihood)
    selected = _selected_nuisances(
        constraints,
        poi=poi,
        parameters=parameters,
    )
    entries: list[ImpactEntry] = []
    for name in selected:
        if name not in index:
            raise ValueError(f"Covariance does not contain nuisance {name!r}.")
        _, sigma = _constraint_mean_sigma(constraints[name])
        signed = float(covariance[index[poi], index[name]]) / sigma
        entries.append(
            ImpactEntry(
                name=name,
                up=signed,
                down=-signed,
                magnitude=abs(signed),
            )
        )
    total = math.sqrt(sum(float(entry.magnitude) ** 2 for entry in entries))
    poi_variance = float(covariance[index[poi], index[poi]])
    statistical_variance = poi_variance - total**2
    scale = max(1.0, abs(poi_variance), total**2)
    tolerance = 1.0e-8 * scale
    warnings: list[str] = []
    parameter_specifications = getattr(
        getattr(likelihood, "intensity", None),
        "parameters",
        (),
    )
    for parameter in parameter_specifications:
        bounds = getattr(parameter, "bounds", None)
        if parameter.name != poi or bounds is None:
            continue
        low, high = bounds
        tolerance_to_bound = 1.0e-7 * max(1.0, abs(high - low))
        if (
            abs(baseline[poi] - low) <= tolerance_to_bound
            or abs(baseline[poi] - high) <= tolerance_to_bound
        ):
            warnings.append(
                "The POI is at a declared bound; the local Gaussian covariance "
                "impact approximation may be unreliable."
            )
        break
    if statistical_variance < -tolerance:
        statistical = None
        warnings.append(
            "Quadrature systematic variance exceeds the POI variance; the "
            "independent Gaussian decomposition is not valid for this fit."
        )
    else:
        statistical = math.sqrt(max(statistical_variance, 0.0))
    if not _fit_success(fit):
        warnings.append("The baseline fit did not report success.")
    return ImpactResult(
        poi=poi,
        method="covariance",
        entries=tuple(entries),
        baseline_point=baseline,
        baseline_nll=_fit_nll(fit),
        total=total,
        groups=_group_magnitudes(entries, groups),
        statistical=statistical,
        warnings=tuple(warnings),
    )


def plot_pulls(
    result: PullResult,
    *,
    ax: Any | None = None,
    sort: bool = False,
) -> tuple[Any, Any]:
    """Plot pulls with the pre-fit band and post-fit horizontal errors."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Pull plots require matplotlib; install hnsbi-toolkit[plots]."
        ) from exc
    entries = list(result.entries)
    if sort:
        entries.sort(key=lambda entry: abs(entry.pull), reverse=True)
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.0, max(2.5, 0.45 * len(entries) + 1.5)))
    else:
        figure = ax.figure
    positions = np.arange(len(entries))
    ax.axvspan(-1.0, 1.0, color="0.92", label="pre-fit $\\pm1\\sigma$")
    ax.errorbar(
        [entry.pull for entry in entries],
        positions,
        xerr=[entry.postfit_over_prefit for entry in entries],
        fmt="o",
        color="black",
        capsize=3,
        label="post-fit",
    )
    ax.axvline(0.0, color="0.4", linewidth=1.0)
    ax.set_yticks(positions, [entry.name for entry in entries])
    ax.set_xlabel("pull $(\\hat{\\theta}-t)/\\sigma_t$")
    ax.set_ylabel("nuisance parameter")
    ax.invert_yaxis()
    ax.legend()
    figure.tight_layout()
    return figure, ax


def plot_impacts(
    result: ImpactResult,
    *,
    ax: Any | None = None,
    sort: bool = True,
) -> tuple[Any, Any]:
    """Plot signed down/up POI responses for an :class:`ImpactResult`."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Impact plots require matplotlib; install hnsbi-toolkit[plots]."
        ) from exc
    entries = list(result.entries)
    if sort:
        entries.sort(
            key=lambda entry: -math.inf if entry.magnitude is None else entry.magnitude,
            reverse=True,
        )
    if ax is None:
        figure, ax = plt.subplots(figsize=(7.0, max(2.5, 0.55 * len(entries) + 1.5)))
    else:
        figure = ax.figure
    positions = np.arange(len(entries))
    offset = 0.18
    height = 0.34
    down = [np.nan if entry.down is None else entry.down for entry in entries]
    up = [np.nan if entry.up is None else entry.up for entry in entries]
    ax.barh(positions + offset, down, height=height, label="$t-\\sigma_t$")
    ax.barh(positions - offset, up, height=height, label="$t+\\sigma_t$")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(positions, [entry.name for entry in entries])
    ax.set_xlabel(f"signed impact on {result.poi}")
    ax.set_ylabel("nuisance parameter")
    ax.invert_yaxis()
    ax.legend()
    figure.tight_layout()
    return figure, ax
