"""Small, typed wrappers around the public :mod:`pyhf` inference API.

``pyhf`` is imported lazily so the core toolkit remains usable without the LHC
extra. These helpers operate on ordinary pyhf data tensors and models; they do
not translate an unbinned hnsbi likelihood into HistFactory.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np


def _json_compatible(value: Any) -> Any:
    if is_dataclass(value):
        return _json_compatible(asdict(value))
    if isinstance(value, dict):
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
        return _json_compatible(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class PyhfFitResult(_JsonResult):
    """Named pyhf maximum-likelihood fit and local covariance information."""

    parameter_names: tuple[str, ...]
    values: tuple[float, ...]
    errors: tuple[float, ...] | None
    twice_nll: float
    success: bool
    message: str
    correlation: np.ndarray | None = None
    covariance: np.ndarray | None = None

    @property
    def point(self) -> dict[str, float]:
        return dict(zip(self.parameter_names, self.values, strict=True))

    @property
    def nll(self) -> float:
        return 0.5 * self.twice_nll


@dataclass(frozen=True)
class PyhfHypotestResult(_JsonResult):
    """Observed and expected CLs at one tested POI value."""

    poi_value: float
    cls: float
    clsb: float
    clb: float
    expected: tuple[float, float, float, float, float]
    calctype: str
    test_stat: str


@dataclass(frozen=True)
class PyhfUpperLimitResult(_JsonResult):
    """Observed and expected pyhf upper limits plus the evaluated CLs curve."""

    observed: float
    expected: tuple[float, float, float, float, float]
    level: float
    calctype: str
    test_stat: str
    scan: tuple[float, ...]
    observed_cls: tuple[float, ...]
    expected_cls: tuple[tuple[float, float, float, float, float], ...]


@dataclass(frozen=True)
class _AuxiliaryConstraint:
    """One pyhf auxiliary observation expressed in nuisance coordinates."""

    mean: float
    sigma: float
    pdf_type: str
    auxiliary_index: int
    factor: float


class PyhfLikelihoodAdapter:
    """Expose a pyhf model through the pull/impact likelihood protocol.

    pyhf appends auxiliary observations to the channel data. The adapter maps
    normal and Poisson constraint observations into the corresponding
    nuisance-parameter coordinates. This makes the same pull, covariance
    impact, and global-observable refit machinery usable for native hNSBI and
    HistFactory models without shifting or fixing nuisance parameters.
    """

    def __init__(self, data: Any, model: Any) -> None:
        pyhf = _require_pyhf()
        values = np.asarray(_tolist(pyhf, data), dtype=np.float64).reshape(-1)
        n_auxiliary = int(model.config.nauxdata)
        if len(values) < n_auxiliary or not np.isfinite(values).all():
            raise ValueError("pyhf data must be finite and include all auxdata.")
        main_count = len(values) - n_auxiliary
        self.data = values
        self.model = model
        self._main_count = main_count
        self._constraint_metadata: dict[str, _AuxiliaryConstraint] = {}

        offset = 0
        for set_name in model.config.auxdata_order:
            parameter_set = model.config.param_set(set_name)
            count = int(parameter_set.n_parameters)
            parameter_slice = model.config.par_map[set_name]["slice"]
            names = tuple(model.config.par_names[parameter_slice])
            if len(names) != count:
                raise ValueError(
                    f"pyhf parameter set {set_name!r} has inconsistent names."
                )
            pdf_type = str(getattr(parameter_set, "pdf_type", ""))
            widths = np.asarray(parameter_set.width(), dtype=np.float64).reshape(-1)
            if len(widths) != count or np.any(widths <= 0):
                raise ValueError(f"pyhf parameter set {set_name!r} has invalid widths.")
            auxiliary = values[main_count + offset : main_count + offset + count]
            if len(auxiliary) != count:
                raise ValueError(f"pyhf data are missing auxdata for {set_name!r}.")
            if pdf_type == "normal":
                factors = np.ones(count, dtype=np.float64)
                means = auxiliary
            elif pdf_type == "poisson":
                factors = np.asarray(
                    getattr(parameter_set, "factors", ()),
                    dtype=np.float64,
                ).reshape(-1)
                if (
                    len(factors) != count
                    or not np.isfinite(factors).all()
                    or np.any(factors <= 0)
                ):
                    raise ValueError(
                        f"pyhf Poisson constraint {set_name!r} has invalid factors."
                    )
                means = auxiliary / factors
            else:
                raise ValueError(
                    f"Unsupported pyhf constraint distribution {pdf_type!r} "
                    f"for {set_name!r}."
                )
            for local_index, name in enumerate(names):
                self._constraint_metadata[name] = _AuxiliaryConstraint(
                    mean=float(means[local_index]),
                    sigma=float(widths[local_index]),
                    pdf_type=pdf_type,
                    auxiliary_index=main_count + offset + local_index,
                    factor=float(factors[local_index]),
                )
            offset += count
        if offset != n_auxiliary:
            raise ValueError("pyhf auxdata ordering does not match nauxdata.")

        self.constraints = {
            name: SimpleNamespace(mean=value.mean, sigma=value.sigma)
            for name, value in self._constraint_metadata.items()
        }
        self.auxiliary_observations = {
            name: value.mean for name, value in self._constraint_metadata.items()
        }
        bounds = tuple(
            tuple(map(float, item)) for item in model.config.suggested_bounds()
        )
        self.intensity = SimpleNamespace(
            parameters=tuple(
                SimpleNamespace(name=name, bounds=bounds[index])
                for index, name in enumerate(model.config.par_names)
            )
        )

    def fit(
        self,
        *,
        initial: dict[str, float] | None = None,
        fixed: dict[str, float] | None = None,
        **optimizer_kwargs: Any,
    ) -> PyhfFitResult:
        """Fit with named initialization/fixing translated to pyhf arrays."""

        names = tuple(self.model.config.par_names)
        index = {name: position for position, name in enumerate(names)}
        initial_values = list(self.model.config.suggested_init())
        fixed_values = list(self.model.config.suggested_fixed())
        for name, value in (initial or {}).items():
            if name not in index:
                raise ValueError(f"Unknown pyhf parameter {name!r}.")
            initial_values[index[name]] = float(value)
        for name, value in (fixed or {}).items():
            if name not in index:
                raise ValueError(f"Unknown pyhf parameter {name!r}.")
            initial_values[index[name]] = float(value)
            fixed_values[index[name]] = True
        return fit(
            self.data,
            self.model,
            init_pars=initial_values,
            fixed_params=fixed_values,
            **optimizer_kwargs,
        )

    def with_auxiliary_observations(
        self,
        observations: dict[str, float],
    ) -> PyhfLikelihoodAdapter:
        """Clone the adapter after changing nuisance-coordinate auxdata."""

        unknown = set(observations).difference(self._constraint_metadata)
        if unknown:
            raise ValueError(f"Unknown pyhf auxiliary observations {sorted(unknown)}.")
        shifted = self.data.copy()
        for name, observed in observations.items():
            metadata = self._constraint_metadata[name]
            value = float(observed)
            if not np.isfinite(value):
                raise ValueError("pyhf auxiliary observations must be finite.")
            raw = value if metadata.pdf_type == "normal" else value * metadata.factor
            if metadata.pdf_type == "poisson" and raw < 0:
                raise ValueError("Poisson auxiliary observations cannot be negative.")
            shifted[metadata.auxiliary_index] = raw
        return PyhfLikelihoodAdapter(shifted, self.model)


def _require_pyhf() -> Any:
    try:
        return importlib.import_module("pyhf")
    except ImportError as exc:
        raise ImportError(
            "pyhf tooling requires pyhf; install hnsbi-toolkit[lhc]."
        ) from exc


def _tolist(pyhf: Any, value: Any) -> Any:
    try:
        return pyhf.tensorlib.tolist(value)
    except (AttributeError, TypeError):
        return np.asarray(value).tolist()


def _scalar(pyhf: Any, value: Any) -> float:
    converted = _tolist(pyhf, value)
    array = np.asarray(converted, dtype=np.float64)
    if array.size != 1:
        raise ValueError(
            f"Expected a scalar pyhf result, received shape {array.shape}."
        )
    return float(array.reshape(-1)[0])


def _five_values(pyhf: Any, values: Any, *, name: str) -> tuple[float, ...]:
    parsed = tuple(
        float(value)
        for value in np.asarray(_tolist(pyhf, values), dtype=np.float64).reshape(-1)
    )
    if len(parsed) != 5:
        raise ValueError(f"{name} must contain the -2,-1,0,+1,+2 sigma values.")
    return parsed


def _has_downward_crossing(values: Sequence[float], level: float) -> bool:
    curve = np.asarray(tuple(values), dtype=np.float64).reshape(-1)
    if len(curve) < 2 or not np.isfinite(curve).all():
        return False
    return bool(np.any((curve[:-1] >= level) & (curve[1:] <= level)))


def fit(
    data: Any,
    model: Any,
    *,
    init_pars: Sequence[float] | None = None,
    par_bounds: Sequence[Sequence[float]] | None = None,
    fixed_params: Sequence[bool] | None = None,
    **optimizer_kwargs: Any,
) -> PyhfFitResult:
    """Fit a pyhf model and request Minuit uncertainties and correlations.

    The active pyhf optimizer must support ``return_uncertainties`` and
    ``return_correlations``. In practice this means configuring pyhf with
    ``pyhf.optimize.minuit_optimizer()`` before calling this function.
    """

    pyhf = _require_pyhf()
    options = dict(optimizer_kwargs)
    options.update(
        {
            "return_uncertainties": True,
            "return_correlations": True,
            "return_fitted_val": True,
            "return_result_obj": True,
        }
    )
    try:
        fitted, correlation, twice_nll, raw = pyhf.infer.mle.fit(
            data,
            model,
            init_pars=(
                list(init_pars)
                if init_pars is not None
                else model.config.suggested_init()
            ),
            par_bounds=(
                [list(bound) for bound in par_bounds]
                if par_bounds is not None
                else model.config.suggested_bounds()
            ),
            fixed_params=(
                list(fixed_params)
                if fixed_params is not None
                else model.config.suggested_fixed()
            ),
            **options,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "The active pyhf optimizer did not provide uncertainties and "
            "correlations. Configure pyhf with "
            "pyhf.set_backend(pyhf.tensorlib, pyhf.optimize.minuit_optimizer())."
        ) from exc

    fitted_array = np.asarray(_tolist(pyhf, fitted), dtype=np.float64)
    if fitted_array.ndim == 2 and fitted_array.shape[1] == 2:
        values = fitted_array[:, 0]
        errors_array: np.ndarray | None = fitted_array[:, 1]
    elif fitted_array.ndim == 1:
        values = fitted_array
        raw_uncertainties = getattr(raw, "unc", None)
        errors_array = (
            None
            if raw_uncertainties is None
            else np.asarray(_tolist(pyhf, raw_uncertainties), dtype=np.float64)
        )
    else:
        raise ValueError(
            "Unexpected pyhf fit output shape. Expected (npars,) or (npars, 2)."
        )
    names = tuple(str(name) for name in model.config.par_names)
    if len(names) != len(values):
        raise ValueError("pyhf parameter names do not match the fitted values.")
    if correlation is None:
        raise RuntimeError(
            "The active pyhf optimizer did not return a correlation matrix. "
            "Configure pyhf with pyhf.optimize.minuit_optimizer()."
        )
    correlation_array = np.asarray(_tolist(pyhf, correlation), dtype=np.float64)
    if correlation_array.shape != (len(names), len(names)):
        raise ValueError("pyhf returned a correlation matrix with the wrong shape.")

    raw_covariance = getattr(raw, "hess_inv", None)
    if raw_covariance is not None:
        covariance = np.asarray(_tolist(pyhf, raw_covariance), dtype=np.float64)
    elif errors_array is not None:
        covariance = correlation_array * np.outer(errors_array, errors_array)
    else:
        covariance = None
    if covariance is not None and covariance.shape != (len(names), len(names)):
        raise ValueError("pyhf returned a covariance matrix with the wrong shape.")
    return PyhfFitResult(
        parameter_names=names,
        values=tuple(float(value) for value in values),
        errors=(
            None
            if errors_array is None
            else tuple(float(value) for value in errors_array)
        ),
        twice_nll=_scalar(pyhf, twice_nll),
        success=bool(getattr(raw, "success", True)),
        message=str(getattr(raw, "message", "")),
        correlation=correlation_array,
        covariance=covariance,
    )


def hypotest(
    poi_value: float,
    data: Any,
    model: Any,
    *,
    calctype: str = "asymptotics",
    test_stat: str = "qtilde",
    **calculator_kwargs: Any,
) -> PyhfHypotestResult:
    """Evaluate observed and expected CLs using pyhf."""

    if calctype not in {"asymptotics", "toybased"}:
        raise ValueError("calctype must be 'asymptotics' or 'toybased'.")
    pyhf = _require_pyhf()
    try:
        observed, tails, expected = pyhf.infer.hypotest(
            float(poi_value),
            data,
            model,
            calctype=calctype,
            test_stat=test_stat,
            return_tail_probs=True,
            return_expected_set=True,
            **calculator_kwargs,
        )
    except TypeError as exc:
        if calctype == "toybased" and "interpolation" in str(exc):
            raise RuntimeError(
                "This pyhf release requires NumPy < 2 for toy-based "
                "percentiles. Install a compatible NumPy version."
            ) from exc
        raise
    tail_values = tuple(
        float(value)
        for value in np.asarray(_tolist(pyhf, tails), dtype=np.float64).reshape(-1)
    )
    if len(tail_values) != 2:
        raise ValueError("pyhf tail probabilities must be [CLs+b, CLb].")
    return PyhfHypotestResult(
        poi_value=float(poi_value),
        cls=_scalar(pyhf, observed),
        clsb=tail_values[0],
        clb=tail_values[1],
        expected=_five_values(pyhf, expected, name="Expected CLs"),
        calctype=calctype,
        test_stat=test_stat,
    )


def upper_limit(
    data: Any,
    model: Any,
    *,
    scan: Sequence[float] | None = None,
    level: float = 0.05,
    calctype: str = "asymptotics",
    test_stat: str = "qtilde",
    **calculator_kwargs: Any,
) -> PyhfUpperLimitResult:
    """Return observed and expected pyhf CLs upper limits.

    A fixed scan is required for toy-based limits because stochastic CLs
    evaluations are unsuitable for deterministic root finding.
    """

    level = float(level)
    if not 0 < level < 1:
        raise ValueError("level must lie strictly between zero and one.")
    if calctype not in {"asymptotics", "toybased"}:
        raise ValueError("calctype must be 'asymptotics' or 'toybased'.")
    if calctype == "toybased" and scan is None:
        raise ValueError("Toy-based upper limits require an explicit fixed scan.")
    pyhf = _require_pyhf()
    scan_values = (
        None if scan is None else np.asarray(tuple(scan), dtype=np.float64).reshape(-1)
    )
    if scan_values is not None and (
        len(scan_values) < 2
        or not np.isfinite(scan_values).all()
        or np.any(np.diff(scan_values) <= 0)
    ):
        raise ValueError("scan must contain at least two increasing finite values.")
    try:
        observed, expected, details = pyhf.infer.intervals.upper_limits.upper_limit(
            data,
            model,
            scan=scan_values,
            level=level,
            return_results=True,
            calctype=calctype,
            test_stat=test_stat,
            **calculator_kwargs,
        )
    except TypeError as exc:
        if calctype == "toybased" and "interpolation" in str(exc):
            raise RuntimeError(
                "This pyhf release requires NumPy < 2 for toy-based "
                "percentiles. Install a compatible NumPy version."
            ) from exc
        raise
    evaluated_scan, results = details
    parsed_scan = tuple(
        float(value)
        for value in np.asarray(
            _tolist(pyhf, evaluated_scan),
            dtype=np.float64,
        ).reshape(-1)
    )
    observed_cls: list[float] = []
    expected_cls: list[tuple[float, float, float, float, float]] = []
    for item in results:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError("Unexpected pyhf upper-limit scan result.")
        observed_cls.append(_scalar(pyhf, item[0]))
        expected_cls.append(_five_values(pyhf, item[1], name="Expected scan CLs"))
    if scan_values is not None:
        curves = {
            "observed": tuple(observed_cls),
            **{
                label: tuple(values[index] for values in expected_cls)
                for index, label in enumerate(
                    (
                        "expected -2 sigma",
                        "expected -1 sigma",
                        "expected median",
                        "expected +1 sigma",
                        "expected +2 sigma",
                    )
                )
            },
        }
        unbracketed = {
            name: values
            for name, values in curves.items()
            if not _has_downward_crossing(values, level)
        }
        if unbracketed:
            endpoints = ", ".join(
                f"{name}=[{values[0]:.6g}, {values[-1]:.6g}]"
                for name, values in unbracketed.items()
            )
            raise ValueError(
                "The fixed pyhf CLs scan does not bracket a downward crossing "
                f"at level {level:g} for {', '.join(unbracketed)}. Extend the "
                f"upper scan edge or lower its first point. Endpoint CLs: {endpoints}."
            )
    return PyhfUpperLimitResult(
        observed=_scalar(pyhf, observed),
        expected=_five_values(pyhf, expected, name="Expected upper limits"),
        level=level,
        calctype=calctype,
        test_stat=test_stat,
        scan=parsed_scan,
        observed_cls=tuple(observed_cls),
        expected_cls=tuple(expected_cls),
    )


def pulls(
    data: Any,
    model: Any,
    *,
    fit_result: PyhfFitResult | None = None,
    parameters: Sequence[str] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Compute pulls and post-fit constraint widths for a pyhf model."""

    from .impacts import compute_pulls

    likelihood = PyhfLikelihoodAdapter(data, model)
    return compute_pulls(
        likelihood,
        fit=fit_result,
        parameters=parameters,
        fit_kwargs=fit_kwargs,
    )


def global_observable_impacts(
    data: Any,
    model: Any,
    *,
    poi: str | None = None,
    fit_result: PyhfFitResult | None = None,
    parameters: Sequence[str] | None = None,
    step: float = 1.0,
    groups: dict[str, Sequence[str]] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Profile pyhf impacts after shifting auxdata, never nuisances.

    Normal auxiliary observations are shifted directly. Poisson auxiliary
    counts are shifted by the corresponding nuisance-coordinate pre-fit width
    and mapped back through the parameter-set factor.
    """

    from .impacts import global_observable_impacts as calculate

    likelihood = PyhfLikelihoodAdapter(data, model)
    selected_poi = model.config.poi_name if poi is None else str(poi)
    return calculate(
        likelihood,
        selected_poi,
        fit=fit_result,
        parameters=parameters,
        step=step,
        groups=groups,
        fit_kwargs=fit_kwargs,
    )


def covariance_impacts(
    data: Any,
    model: Any,
    *,
    poi: str | None = None,
    fit_result: PyhfFitResult | None = None,
    parameters: Sequence[str] | None = None,
    groups: dict[str, Sequence[str]] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Compute local pyhf impacts from the Minuit covariance matrix."""

    from .impacts import covariance_impacts as calculate

    likelihood = PyhfLikelihoodAdapter(data, model)
    selected_poi = model.config.poi_name if poi is None else str(poi)
    return calculate(
        likelihood,
        selected_poi,
        fit=fit_result,
        parameters=parameters,
        groups=groups,
        fit_kwargs=fit_kwargs,
    )


__all__ = [
    "PyhfFitResult",
    "PyhfHypotestResult",
    "PyhfLikelihoodAdapter",
    "PyhfUpperLimitResult",
    "covariance_impacts",
    "fit",
    "global_observable_impacts",
    "hypotest",
    "pulls",
    "upper_limit",
]
