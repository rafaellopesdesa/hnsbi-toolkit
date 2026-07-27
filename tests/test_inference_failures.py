from __future__ import annotations

from types import SimpleNamespace

import pytest

from hnsbi.inference import MinuitInference
from hnsbi.likelihood import ExtendedUnbinnedLikelihood, FitResult


def _fit(*, success: bool, nll: float, message: str) -> FitResult:
    return FitResult(
        point={"mu": 1.0},
        nll=nll,
        success=success,
        message=message,
        evaluations=1,
    )


def _stub_inference(results: list[FitResult]) -> MinuitInference:
    inference = object.__new__(MinuitInference)
    inference.parameter_names = ("mu",)
    inference.likelihood = SimpleNamespace(
        intensity=SimpleNamespace(
            parameters=(SimpleNamespace(name="mu", bounds=(0.0, 5.0)),)
        )
    )
    remaining = iter(results)

    def fit(**kwargs):
        return next(remaining)

    inference.fit = fit  # type: ignore[method-assign]
    return inference


def test_profile_scan_rejects_failed_global_fit() -> None:
    inference = _stub_inference(
        [_fit(success=False, nll=3.0, message="MIGRAD invalid")]
    )

    with pytest.raises(RuntimeError, match="Global fit failed") as error:
        inference.profile_scan("mu", [0.0, 1.0])
    assert "MIGRAD invalid" in str(error.value)


def test_profile_and_test_statistic_scans_reject_failed_fixed_fit() -> None:
    inference = _stub_inference(
        [
            _fit(success=True, nll=1.0, message="ok"),
            _fit(success=False, nll=2.0, message="EDM too large"),
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="Profile fit failed for mu=0.5",
    ) as error:
        inference.test_statistic_scan("mu", [0.5])
    assert "EDM too large" in str(error.value)


def test_profile_scan_rejects_nonfinite_nll_even_when_fit_claims_success() -> None:
    inference = _stub_inference(
        [_fit(success=True, nll=float("nan"), message="claimed success")]
    )

    with pytest.raises(RuntimeError, match="Global fit failed") as error:
        inference.profile_scan("mu", [1.0])
    assert "nan" in str(error.value)


@pytest.mark.parametrize("scan_value", [-0.1, 5.1])
def test_minuit_profile_scan_rejects_values_outside_declared_bounds(
    scan_value: float,
) -> None:
    inference = _stub_inference([])

    with pytest.raises(
        ValueError,
        match=r"Scan values leave declared bounds \[0.0, 5.0\]",
    ):
        inference.profile_scan("mu", [scan_value])


def test_scipy_profile_scan_uses_the_same_strict_failure_contract() -> None:
    likelihood = object.__new__(ExtendedUnbinnedLikelihood)
    likelihood.intensity = SimpleNamespace(
        parameters=(SimpleNamespace(name="mu", bounds=(0.0, 5.0)),)
    )
    results = iter(
        [
            _fit(success=True, nll=1.0, message="ok"),
            _fit(success=False, nll=2.0, message="line search failed"),
        ]
    )

    def fit(**kwargs):
        return next(results)

    likelihood.fit = fit  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="Profile fit failed") as error:
        likelihood.profile_scan("mu", [1.0])
    assert "line search failed" in str(error.value)
