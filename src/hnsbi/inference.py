"""Native JAX likelihood evaluation, Minuit fitting, and test-statistic scans."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .likelihood import FitResult, ProfileScanResult
from .onnx import require_optional


def _jax_functions(jnp: Any) -> dict[str, Any]:
    return {
        "abs": jnp.abs,
        "clip": jnp.clip,
        "exp": jnp.exp,
        "log": jnp.log,
        "maximum": jnp.maximum,
        "minimum": jnp.minimum,
        "sqrt": jnp.sqrt,
    }


def _code4p_jax(alpha: Any, up: Any, down: Any, jnp: Any) -> Any:
    log_up = jnp.log(up)
    log_down = jnp.log(down)
    up_log = up * log_up
    down_log = -(down * log_down)
    up_log2 = up_log * log_up
    down_log2 = -(down_log * log_down)
    s0 = (up + down) / 2.0
    a0 = (up - down) / 2.0
    s1 = (up_log + down_log) / 2.0
    a1 = (up_log - down_log) / 2.0
    s2 = (up_log2 + down_log2) / 2.0
    a2 = (up_log2 - down_log2) / 2.0
    coefficients = (
        (15.0 * a0 - 7.0 * s1 + a2) / 8.0,
        (-24.0 + 24.0 * s0 - 9.0 * a1 + s2) / 8.0,
        (-5.0 * a0 + 5.0 * s1 - a2) / 4.0,
        (12.0 - 12.0 * s0 + 7.0 * a1 - s2) / 4.0,
        (3.0 * a0 - 3.0 * s1 + a2) / 8.0,
        (-8.0 + 8.0 * s0 - 5.0 * a1 + s2) / 8.0,
    )
    polynomial = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        polynomial = coefficient + alpha * polynomial
    interior = 1.0 + alpha * polynomial
    extrapolated = jnp.where(alpha > 1.0, up**alpha, down ** (-alpha))
    return jnp.where(jnp.abs(alpha) <= 1.0, interior, extrapolated)


def _interpolate_jax(
    alpha: Any,
    up: Any,
    down: Any,
    interpolation: str,
    jnp: Any,
) -> Any:
    if interpolation == "nsbi_code4p":
        return _code4p_jax(alpha, up, down, jnp)
    if interpolation == "linear":
        return jnp.where(
            alpha >= 0,
            1.0 + alpha * (up - 1.0),
            1.0 + (-alpha) * (down - 1.0),
        )
    raise ValueError(f"Unsupported systematic interpolation {interpolation!r}.")


class JaxLikelihood:
    """JIT-ready differentiable view of :class:`ExtendedUnbinnedLikelihood`."""

    def __init__(
        self,
        likelihood: Any,
        *,
        enable_x64: bool = True,
    ) -> None:
        jax = require_optional(
            "jax",
            extra="lhc",
            purpose="autodifferentiable likelihood evaluation",
        )
        if enable_x64:
            jax.config.update("jax_enable_x64", True)
        self.jax = jax
        self.jnp = jax.numpy
        self.likelihood = likelihood
        if getattr(likelihood, "fnf_systematics", None):
            raise ValueError(
                "FNF residuals execute with Torch/ONNX and cannot be "
                "autodifferentiated by JAX. Use MinuitInference(..., "
                "use_jax=False) for FNF workspaces."
            )
        self.parameter_names = tuple(
            parameter.name for parameter in likelihood.intensity.parameters
        )
        self._parameter_index = {
            name: index for index, name in enumerate(self.parameter_names)
        }
        self._functions = _jax_functions(self.jnp)
        channels = getattr(likelihood, "channels", None)
        self._channel_views = (
            tuple(JaxLikelihood(channel, enable_x64=False) for channel in channels)
            if channels is not None
            else ()
        )
        if self._channel_views:
            for view in self._channel_views:
                if view.parameter_names != self.parameter_names:
                    raise ValueError(
                        "Combined JAX channels must share an ordered parameter vector."
                    )
            self._ratios = {}
            self._event_weights = None
            self._integration_weights = None
            self._active = None
        else:
            self._ratios = {
                name: self.jnp.asarray(values)
                for name, values in likelihood.ratios.items()
            }
            self._event_weights = self.jnp.asarray(likelihood.event_weights)
            self._integration_weights = self.jnp.asarray(likelihood.integration_weights)
            self._active = self._event_weights > 0
        self._compiled_nll = jax.jit(self._nll_vector)
        self._compiled_value_grad = jax.jit(jax.value_and_grad(self._nll_vector))
        self._compiled_hessian = jax.jit(jax.hessian(self._nll_vector))

    def vector(self, point: Mapping[str, float]) -> np.ndarray:
        validated = self.likelihood.intensity.validate_point(point)
        return np.asarray(
            [validated[name] for name in self.parameter_names],
            dtype=np.float64,
        )

    def point(self, vector: Sequence[float]) -> dict[str, float]:
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if len(values) != len(self.parameter_names):
            raise ValueError("Parameter vector has the wrong length.")
        return {
            name: float(value)
            for name, value in zip(self.parameter_names, values, strict=True)
        }

    def _data_nll_vector(self, vector: Any) -> Any:
        jnp = self.jnp
        if self._channel_views:
            value = jnp.asarray(0.0)
            for view in self._channel_views:
                value = value + view._data_nll_vector(vector)
            return value
        point = {name: vector[index] for name, index in self._parameter_index.items()}
        differential = jnp.zeros_like(self._event_weights)
        expected = jnp.asarray(0.0, dtype=self._event_weights.dtype)
        valid = jnp.asarray(True)
        for component in self.likelihood.intensity.components:
            multiplier = component.multiplier.evaluate_with(point, self._functions)
            component_yield = component.nominal_yield * multiplier
            shape = self._ratios[component.name]
            evaluator = self.likelihood.systematic_evaluators.get(component.name)
            if evaluator is not None:
                combined = jnp.ones_like(shape)
                yield_factor = jnp.asarray(1.0, dtype=shape.dtype)
                for anchor in evaluator.anchors:
                    alpha = point[anchor.parameter]
                    up = jnp.asarray(anchor.ratio_up)
                    down = jnp.asarray(anchor.ratio_down)
                    combined = combined * _interpolate_jax(
                        alpha,
                        up,
                        down,
                        anchor.interpolation,
                        jnp,
                    )
                    yield_factor = yield_factor * _interpolate_jax(
                        alpha,
                        jnp.asarray(anchor.yield_up),
                        jnp.asarray(anchor.yield_down),
                        anchor.interpolation,
                        jnp,
                    )
                partition = jnp.sum(self._integration_weights * shape * combined)
                valid = valid & jnp.isfinite(partition) & (partition > 0)
                shape = shape * combined / partition
                component_yield = component_yield * yield_factor
            valid = valid & jnp.isfinite(component_yield) & (component_yield >= 0)
            expected = expected + component_yield
            differential = differential + component_yield * shape
        safe_differential = jnp.where(
            self._active,
            differential,
            jnp.asarray(1.0, dtype=differential.dtype),
        )
        valid = (
            valid
            & jnp.isfinite(expected)
            & (expected >= 0)
            & jnp.all(jnp.isfinite(differential))
            & jnp.all(differential >= 0)
            & jnp.all(jnp.where(self._active, differential > 0, True))
        )
        value = expected - jnp.sum(self._event_weights * jnp.log(safe_differential))
        return jnp.where(valid, value, jnp.inf)

    def _nll_vector(self, vector: Any) -> Any:
        value = self._data_nll_vector(vector)
        point = {name: vector[index] for name, index in self._parameter_index.items()}
        for name, constraint in self.likelihood.constraints.items():
            value = (
                value + 0.5 * ((point[name] - constraint.mean) / constraint.sigma) ** 2
            )
        return value

    def nll(self, vector_or_point: Sequence[float] | Mapping[str, float]) -> float:
        vector = (
            self.vector(vector_or_point)
            if isinstance(vector_or_point, Mapping)
            else np.asarray(vector_or_point, dtype=np.float64)
        )
        return float(self._compiled_nll(vector))

    def value_and_grad(
        self,
        vector_or_point: Sequence[float] | Mapping[str, float],
    ) -> tuple[float, np.ndarray]:
        vector = (
            self.vector(vector_or_point)
            if isinstance(vector_or_point, Mapping)
            else np.asarray(vector_or_point, dtype=np.float64)
        )
        value, gradient = self._compiled_value_grad(vector)
        return float(value), np.asarray(gradient, dtype=np.float64)

    def hessian(
        self,
        vector_or_point: Sequence[float] | Mapping[str, float],
    ) -> np.ndarray:
        vector = (
            self.vector(vector_or_point)
            if isinstance(vector_or_point, Mapping)
            else np.asarray(vector_or_point, dtype=np.float64)
        )
        return np.asarray(self._compiled_hessian(vector), dtype=np.float64)


@dataclass(frozen=True)
class TestStatisticPoint:
    poi: float
    q: float
    fixed_fit: FitResult


@dataclass(frozen=True)
class TestStatisticScan:
    """One-sided profile-likelihood test statistic over a POI grid."""

    parameter: str
    points: tuple[TestStatisticPoint, ...]
    global_fit: FitResult

    @property
    def values(self) -> np.ndarray:
        return np.asarray([point.poi for point in self.points])

    @property
    def q(self) -> np.ndarray:
        return np.asarray([point.q for point in self.points])


class MinuitInference:
    """MIGRAD/HESSE minimization with JAX gradients and named covariance."""

    def __init__(
        self,
        likelihood: Any,
        *,
        use_jax: bool = True,
        enable_x64: bool = True,
    ) -> None:
        self.likelihood = likelihood
        self.parameter_names = tuple(
            parameter.name for parameter in likelihood.intensity.parameters
        )
        self._jax = (
            JaxLikelihood(likelihood, enable_x64=enable_x64) if use_jax else None
        )

    def fit(
        self,
        *,
        initial: Mapping[str, float] | None = None,
        fixed: Mapping[str, float] | None = None,
        run_hesse: bool = True,
        strategy: int = 1,
        tolerance: float | None = None,
    ) -> FitResult:
        iminuit = require_optional(
            "iminuit", extra="lhc", purpose="Minuit likelihood fitting"
        )
        fixed_point = {name: float(value) for name, value in (fixed or {}).items()}
        unknown = set(fixed_point).difference(self.parameter_names)
        if unknown:
            raise ValueError(f"Fixed point has unknown parameters {sorted(unknown)}.")
        seed = dict(self.likelihood.intensity.nominal_point)
        seed.update(initial or {})
        seed.update(fixed_point)
        self.likelihood.intensity.validate_point(seed)

        def objective(*values: float) -> float:
            vector = np.asarray(values, dtype=np.float64)
            if self._jax is not None:
                return self._jax.nll(vector)
            return self.likelihood.nll(
                {
                    name: float(value)
                    for name, value in zip(
                        self.parameter_names,
                        vector,
                        strict=True,
                    )
                }
            )

        gradient = None
        if self._jax is not None:

            def gradient(*values: float) -> np.ndarray:
                return self._jax.value_and_grad(np.asarray(values, dtype=np.float64))[1]

        start = [seed[name] for name in self.parameter_names]
        minimizer = iminuit.Minuit(
            objective,
            *start,
            name=self.parameter_names,
            grad=gradient,
        )
        # The objective is NLL, not -2 log L. This scaling is essential for
        # HESSE errors and covariance.
        minimizer.errordef = 0.5
        minimizer.strategy = int(strategy)
        if tolerance is not None:
            minimizer.tol = float(tolerance)
        parameters = {
            parameter.name: parameter
            for parameter in self.likelihood.intensity.parameters
        }
        for name, parameter in parameters.items():
            if parameter.bounds is not None:
                minimizer.limits[name] = parameter.bounds
        for name, value in fixed_point.items():
            minimizer.values[name] = value
            minimizer.fixed[name] = True
        minimizer.migrad()
        if run_hesse and minimizer.valid:
            minimizer.hesse()
        point = {name: float(minimizer.values[name]) for name in self.parameter_names}
        errors = {name: float(minimizer.errors[name]) for name in self.parameter_names}
        covariance: np.ndarray | None = None
        if minimizer.covariance is not None:
            covariance = np.zeros(
                (len(self.parameter_names), len(self.parameter_names)),
                dtype=np.float64,
            )
            for row, left in enumerate(self.parameter_names):
                for column, right in enumerate(self.parameter_names):
                    covariance[row, column] = float(minimizer.covariance[left, right])
        correlation: np.ndarray | None = None
        if covariance is not None:
            scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            correlation = np.divide(
                covariance,
                np.outer(scale, scale),
                out=np.zeros_like(covariance),
                where=np.outer(scale, scale) > 0,
            )
        at_limit = tuple(
            name
            for name, parameter in parameters.items()
            if parameter.bounds is not None
            and (
                math.isclose(point[name], parameter.bounds[0], rel_tol=0, abs_tol=1e-7)
                or math.isclose(
                    point[name],
                    parameter.bounds[1],
                    rel_tol=0,
                    abs_tol=1e-7,
                )
            )
        )
        fmin = minimizer.fmin
        message = (
            "MIGRAD converged."
            if minimizer.valid
            else "MIGRAD did not produce a valid minimum."
        )
        return FitResult(
            point=point,
            nll=float(minimizer.fval),
            success=bool(minimizer.valid),
            message=message,
            evaluations=int(minimizer.nfcn),
            covariance=covariance,
            parameter_names=self.parameter_names,
            errors=errors,
            correlation=correlation,
            edm=float(fmin.edm),
            valid=bool(minimizer.valid),
            parameters_at_limit=at_limit,
            backend="jax+iminuit" if self._jax is not None else "iminuit",
        )

    def profile_scan(
        self,
        parameter: str,
        values: Sequence[float],
        *,
        initial: Mapping[str, float] | None = None,
    ) -> ProfileScanResult:
        """Profile a parameter, raising if any minimization is unsuccessful.

        Returning a finite-looking test statistic for a failed optimizer point
        is unsafe. Call :meth:`fit` directly when custom retry or masking
        policy is required.
        """

        if parameter not in self.parameter_names:
            raise ValueError(f"Unknown scan parameter {parameter!r}.")
        grid = np.asarray(values, dtype=np.float64).reshape(-1)
        if not len(grid) or not np.isfinite(grid).all():
            raise ValueError("Scan values must be a non-empty finite sequence.")
        parameter_spec = next(
            item
            for item in self.likelihood.intensity.parameters
            if item.name == parameter
        )
        if parameter_spec.bounds is not None:
            low, high = parameter_spec.bounds
            if np.any((grid < low) | (grid > high)):
                raise ValueError(f"Scan values leave declared bounds [{low}, {high}].")
        global_fit = self.fit(initial=initial)
        if not global_fit.success or not np.isfinite(global_fit.nll):
            raise RuntimeError(
                "Global fit failed before profile scan: "
                f"{global_fit.message} (NLL={global_fit.nll!r})."
            )
        seed = dict(global_fit.point)
        fits: list[FitResult] = []
        for value in grid:
            fit = self.fit(initial=seed, fixed={parameter: float(value)})
            if not fit.success or not np.isfinite(fit.nll):
                raise RuntimeError(
                    f"Profile fit failed for {parameter}={float(value)!r}: "
                    f"{fit.message} (NLL={fit.nll!r})."
                )
            fits.append(fit)
            seed.update(fit.point)
        twice_delta = np.maximum(
            0.0,
            np.asarray([2.0 * (fit.nll - global_fit.nll) for fit in fits]),
        )
        return ProfileScanResult(
            parameter=parameter,
            values=grid,
            twice_delta_nll=twice_delta,
            profiled_points=tuple(fit.point for fit in fits),
            global_fit=global_fit,
            fits=tuple(fits),
        )

    def test_statistic_scan(
        self,
        parameter: str,
        values: Sequence[float],
        *,
        one_sided: bool = True,
        initial: Mapping[str, float] | None = None,
    ) -> TestStatisticScan:
        profile = self.profile_scan(parameter, values, initial=initial)
        best = float(profile.global_fit.point[parameter])
        points = []
        for value, delta, fit in zip(
            profile.values,
            profile.twice_delta_nll,
            profile.fits,
            strict=True,
        ):
            q = 0.0 if one_sided and best > value else float(delta)
            points.append(
                TestStatisticPoint(
                    poi=float(value),
                    q=q,
                    fixed_fit=fit,
                )
            )
        return TestStatisticScan(
            parameter=parameter,
            points=tuple(points),
            global_fit=profile.global_fit,
        )
