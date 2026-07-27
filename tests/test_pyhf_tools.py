from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hnsbi import pyhf_tools


class FakeTensorlib:
    @staticmethod
    def tolist(value: object) -> object:
        return np.asarray(value).tolist()


class FakeConfig:
    par_names = ["mu", "theta"]

    @staticmethod
    def suggested_init() -> list[float]:
        return [1.0, 0.0]

    @staticmethod
    def suggested_bounds() -> list[list[float]]:
        return [[0.0, 5.0], [-5.0, 5.0]]

    @staticmethod
    def suggested_fixed() -> list[bool]:
        return [False, False]


class FakeNormalParamset:
    n_parameters = 1
    pdf_type = "normal"

    @staticmethod
    def width() -> list[float]:
        return [1.0]


class FakeAdapterConfig(FakeConfig):
    par_names = ["theta", "mu"]
    par_order = ["theta", "mu"]
    poi_name = "mu"
    nauxdata = 1
    auxdata_order = ["theta"]
    par_map = {"theta": {"slice": slice(0, 1)}}

    @staticmethod
    def suggested_bounds() -> list[list[float]]:
        return [[-5.0, 5.0], [0.0, 5.0]]

    @staticmethod
    def param_set(name: str) -> FakeNormalParamset:
        assert name == "theta"
        return FakeNormalParamset()


class FakeMle:
    calls: list[dict[str, object]] = []

    @classmethod
    def fit(cls, data: object, model: object, **kwargs: object) -> tuple[object, ...]:
        cls.calls.append(kwargs)
        raw = SimpleNamespace(
            success=True,
            message="ok",
            hess_inv=np.asarray([[0.04, 0.01], [0.01, 0.09]]),
        )
        fitted = np.asarray([[1.2, 0.2], [-0.1, 0.3]])
        correlation = np.asarray([[1.0, 1.0 / 6.0], [1.0 / 6.0, 1.0]])
        return fitted, correlation, np.asarray(5.0), raw


class FakeUpperLimits:
    calls: list[dict[str, object]] = []

    @classmethod
    def upper_limit(
        cls,
        data: object,
        model: object,
        **kwargs: object,
    ) -> tuple[object, ...]:
        cls.calls.append(kwargs)
        scan = np.asarray([0.0, 1.0, 2.0])
        results = [
            (0.9, [0.8, 0.85, 0.9, 0.94, 0.97]),
            (0.2, [0.1, 0.15, 0.2, 0.3, 0.4]),
            (0.01, [0.005, 0.008, 0.01, 0.02, 0.03]),
        ]
        return (
            np.asarray(1.5),
            np.asarray([1.1, 1.3, 1.6, 1.9, 2.2]),
            (scan, results),
        )


class FakeInfer:
    mle = FakeMle()
    intervals = SimpleNamespace(
        upper_limits=SimpleNamespace(upper_limit=FakeUpperLimits.upper_limit)
    )
    hypotest_calls: list[dict[str, object]] = []

    @classmethod
    def hypotest(
        cls,
        poi_value: float,
        data: object,
        model: object,
        **kwargs: object,
    ) -> tuple[object, ...]:
        cls.hypotest_calls.append(kwargs)
        return (
            np.asarray(0.04),
            np.asarray([0.02, 0.5]),
            np.asarray([0.01, 0.02, 0.04, 0.08, 0.14]),
        )


@pytest.fixture
def fake_pyhf(monkeypatch: pytest.MonkeyPatch) -> object:
    module = SimpleNamespace(tensorlib=FakeTensorlib(), infer=FakeInfer())
    monkeypatch.setitem(sys.modules, "pyhf", module)
    return module


def test_pyhf_fit_wrapper_names_covariance_and_json(fake_pyhf: object) -> None:
    model = SimpleNamespace(config=FakeConfig())
    result = pyhf_tools.fit([12.0], model)
    assert result.point == pytest.approx({"mu": 1.2, "theta": -0.1})
    assert result.errors == pytest.approx((0.2, 0.3))
    assert result.covariance == pytest.approx(np.asarray([[0.04, 0.01], [0.01, 0.09]]))
    assert result.nll == pytest.approx(2.5)
    payload = json.loads(result.to_json())
    assert payload["parameter_names"] == ["mu", "theta"]
    assert FakeMle.calls[-1]["return_uncertainties"] is True
    assert FakeMle.calls[-1]["return_correlations"] is True


def test_pyhf_hypotest_wrapper_exposes_cls_tails_and_bands(
    fake_pyhf: object,
) -> None:
    model = SimpleNamespace(config=FakeConfig())
    result = pyhf_tools.hypotest(
        1.0,
        [12.0],
        model,
        calctype="toybased",
        ntoys=200,
        track_progress=False,
    )
    assert result.cls == pytest.approx(0.04)
    assert result.clsb == pytest.approx(0.02)
    assert result.clb == pytest.approx(0.5)
    assert result.expected[2] == pytest.approx(0.04)
    assert FakeInfer.hypotest_calls[-1]["ntoys"] == 200
    assert FakeInfer.hypotest_calls[-1]["return_expected_set"] is True


def test_pyhf_upper_limit_wrapper_keeps_scan_curves(fake_pyhf: object) -> None:
    model = SimpleNamespace(config=FakeConfig())
    result = pyhf_tools.upper_limit(
        [12.0],
        model,
        scan=[0.0, 1.0, 2.0],
    )
    assert result.observed == pytest.approx(1.5)
    assert result.expected[2] == pytest.approx(1.6)
    assert result.scan == pytest.approx((0.0, 1.0, 2.0))
    assert result.observed_cls == pytest.approx((0.9, 0.2, 0.01))
    assert result.expected_cls[1][2] == pytest.approx(0.2)


def test_fixed_scan_rejects_unbracketed_observed_limit(
    fake_pyhf: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unbracketed(*args: object, **kwargs: object) -> tuple[object, ...]:
        scan = np.asarray([0.0, 1.0, 2.0])
        results = [
            (0.9, [0.8, 0.85, 0.9, 0.94, 0.97]),
            (0.2, [0.1, 0.15, 0.2, 0.3, 0.4]),
            (0.1, [0.005, 0.008, 0.01, 0.02, 0.03]),
        ]
        return np.asarray(2.0), np.ones(5), (scan, results)

    monkeypatch.setattr(
        fake_pyhf.infer.intervals.upper_limits,
        "upper_limit",
        unbracketed,
    )
    with pytest.raises(ValueError, match="observed"):
        pyhf_tools.upper_limit(
            [12.0], SimpleNamespace(config=FakeConfig()), scan=[0, 1, 2]
        )


def test_fixed_scan_rejects_unbracketed_expected_band(
    fake_pyhf: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unbracketed(*args: object, **kwargs: object) -> tuple[object, ...]:
        scan = np.asarray([0.0, 1.0, 2.0])
        results = [
            (0.9, [0.8, 0.85, 0.9, 0.94, 0.97]),
            (0.2, [0.1, 0.15, 0.2, 0.3, 0.4]),
            (0.01, [0.005, 0.008, 0.01, 0.02, 0.3]),
        ]
        return np.asarray(1.5), np.ones(5), (scan, results)

    monkeypatch.setattr(
        fake_pyhf.infer.intervals.upper_limits,
        "upper_limit",
        unbracketed,
    )
    with pytest.raises(ValueError, match=r"expected \+2 sigma"):
        pyhf_tools.upper_limit(
            [12.0], SimpleNamespace(config=FakeConfig()), scan=[0, 1, 2]
        )


def test_toy_upper_limit_requires_fixed_scan() -> None:
    with pytest.raises(ValueError, match="explicit fixed scan"):
        pyhf_tools.upper_limit(
            [12.0],
            SimpleNamespace(config=FakeConfig()),
            calctype="toybased",
        )


def test_pyhf_adapter_maps_auxdata_to_pull_coordinates(fake_pyhf: object) -> None:
    model = SimpleNamespace(config=FakeAdapterConfig())
    adapter = pyhf_tools.PyhfLikelihoodAdapter([12.0, 0.25], model)
    assert adapter.auxiliary_observations == pytest.approx({"theta": 0.25})
    shifted = adapter.with_auxiliary_observations({"theta": 1.25})
    assert shifted.data == pytest.approx([12.0, 1.25])
    fit_result = pyhf_tools.PyhfFitResult(
        parameter_names=("theta", "mu"),
        values=(0.5, 1.2),
        errors=(0.4, 0.2),
        twice_nll=4.0,
        success=True,
        message="",
        covariance=np.asarray([[0.16, 0.02], [0.02, 0.04]]),
        correlation=np.asarray([[1.0, 0.25], [0.25, 1.0]]),
    )
    result = pyhf_tools.pulls(
        [12.0, 0.25],
        model,
        fit_result=fit_result,
    )
    assert result.entries[0].name == "theta"
    assert result.entries[0].pull == pytest.approx(0.25)
    assert result.entries[0].postfit_over_prefit == pytest.approx(0.4)


def test_missing_pyhf_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = pyhf_tools.importlib.import_module

    def fail(name: str) -> object:
        if name == "pyhf":
            raise ImportError("missing")
        return real_import(name)

    monkeypatch.setattr(pyhf_tools.importlib, "import_module", fail)
    with pytest.raises(ImportError, match=r"hnsbi-toolkit\[lhc\]"):
        pyhf_tools.fit([1.0], SimpleNamespace(config=FakeConfig()))


def test_real_pyhf_minuit_smoke_when_available() -> None:
    pyhf = pytest.importorskip("pyhf")
    pytest.importorskip("iminuit")
    previous_backend, previous_optimizer = pyhf.get_backend()
    try:
        pyhf.set_backend(
            "numpy",
            pyhf.optimize.minuit_optimizer(verbose=0),
        )
        model = pyhf.simplemodels.uncorrelated_background(
            signal=[5.0],
            bkg=[20.0],
            bkg_uncertainty=[2.0],
        )
        data = [23.0, *model.config.auxdata]
        fit_result = pyhf_tools.fit(data, model)
        assert fit_result.success
        assert fit_result.parameter_names == (
            "mu",
            "uncorr_bkguncrt[0]",
        )
        assert fit_result.covariance is not None
        assert fit_result.covariance.shape == (2, 2)

        pull_result = pyhf_tools.pulls(
            data,
            model,
            fit_result=fit_result,
        )
        assert pull_result.entries[0].name == "uncorr_bkguncrt[0]"
        covariance_result = pyhf_tools.covariance_impacts(
            data,
            model,
            fit_result=fit_result,
        )
        assert covariance_result.poi == "mu"
        assert covariance_result.entries[0].name == "uncorr_bkguncrt[0]"

        normal_model = pyhf.simplemodels.correlated_background(
            signal=[5.0, 4.0],
            bkg=[20.0, 30.0],
            bkg_up=[22.0, 33.0],
            bkg_down=[18.0, 27.0],
        )
        normal_data = [24.0, 35.0, *normal_model.config.auxdata]
        normal_fit = pyhf_tools.fit(normal_data, normal_model)
        refit_impacts = pyhf_tools.global_observable_impacts(
            normal_data,
            normal_model,
            fit_result=normal_fit,
        )
        assert refit_impacts.method == "global_observable"
        assert refit_impacts.entries[0].name == "correlated_bkg_uncertainty"
        assert refit_impacts.entries[0].up_fit_success
        assert refit_impacts.entries[0].down_fit_success

        test_result = pyhf_tools.hypotest(1.0, data, model)
        assert 0.0 <= test_result.cls <= 1.0
        assert len(test_result.expected) == 5

        limit_result = pyhf_tools.upper_limit(
            data,
            model,
            scan=np.linspace(0.0, 5.0, 11),
        )
        assert np.isfinite(limit_result.observed)
        assert len(limit_result.observed_cls) == 11

        np.random.seed(20260727)
        toy_test = pyhf_tools.hypotest(
            1.0,
            data,
            model,
            calctype="toybased",
            ntoys=5,
            track_progress=False,
        )
        assert toy_test.calctype == "toybased"
        assert 0.0 <= toy_test.cls <= 1.0
        toy_limit = pyhf_tools.upper_limit(
            data,
            model,
            scan=[0.0, 1.5, 3.0],
            calctype="toybased",
            ntoys=5,
            track_progress=False,
        )
        assert toy_limit.calctype == "toybased"
        assert np.isfinite(toy_limit.observed)
        assert len(toy_limit.observed_cls) == 3
    finally:
        pyhf.set_backend(previous_backend, previous_optimizer)
