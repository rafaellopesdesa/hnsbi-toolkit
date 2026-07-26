"""End-to-end smoke test against nsbi-common-utils from upstream ``main``.

This module is deliberately skip-safe: the default test matrix does not
install the heavyweight LHC stack.  The dedicated CI job installs that stack
and therefore exercises the real upstream model and inference engine.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("iminuit")
pytest.importorskip("nsbi_common_utils.models")

from hnsbi.asimov import AsimovBuilder
from hnsbi.integrations.nsbi_inference import NsbiCommonUtilsInference
from hnsbi.intensity import Component, IntensityModel, Parameter
from hnsbi.workspace import write_nsbi_workspace


class _GaussianReference:
    def sample(
        self,
        n: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        generator = np.random.default_rng() if rng is None else rng
        return generator.normal(size=(n, 1))


def test_real_upstream_model_and_inference_consume_hnsbi_workspace(
    tmp_path,
) -> None:
    """Write, verify, fit, and scan a portable one-parameter workspace."""

    intensity = IntensityModel(
        [Component("signal", 12.0, "mu")],
        [Parameter("mu", 1.0, (0.2, 2.0))],
    )
    asimov = AsimovBuilder(
        reference=_GaussianReference(),
        ratios={"signal": lambda values: np.ones(len(values))},
        intensity=intensity,
        features=["x"],
    ).build({"mu": 1.0}, n_events=64, seed=19)
    export = write_nsbi_workspace(
        result=asimov,
        intensity=intensity,
        output_dir=tmp_path,
        measurement="measurement",
        poi="mu",
        require_upstream_compatible=True,
    )

    runtime = NsbiCommonUtilsInference.from_workspace(export.path)
    assert runtime.parameter_names == ("mu",)
    assert runtime.model.__class__.__module__.startswith("nsbi_common_utils.")
    assert runtime.engine.__class__.__module__ == "nsbi_common_utils.inference"

    _, initial = runtime.model.get_model_parameters()
    initial = np.asarray(initial, dtype=np.float64)
    nll = float(runtime.model.model(initial))
    gradient = np.asarray(runtime.model.model_grad(initial), dtype=np.float64)
    assert np.isfinite(nll)
    assert gradient.shape == (1,)
    assert np.all(np.isfinite(gradient))
    assert abs(gradient[0]) < 1.0e-8

    best_fit = runtime.perform_fit(fit_strategy=0)
    assert best_fit == pytest.approx([1.0], abs=2.0e-3)

    scan_points, delta_nll = runtime.perform_profile_scan(
        "mu",
        bound_range=(0.6, 1.4),
        fit_strategy=0,
        size=5,
    )
    assert scan_points.shape == delta_nll.shape == (5,)
    assert np.all(np.isfinite(delta_nll))
    assert np.min(delta_nll) == pytest.approx(0.0, abs=1.0e-8)
