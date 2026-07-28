from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


VALIDATION = _load_module(
    ROOT / "examples" / "notebooks" / "utils_lhc_validation.py",
    "hnsbi_notebook_lhc_validation",
)
GENERATOR = _load_module(
    ROOT / "examples" / "lhc_analysis" / "generate_distributions.py",
    "hnsbi_notebook_lhc_generator",
)


def test_reuse_provenance_binds_configuration_data_and_artifacts(tmp_path) -> None:
    configuration = tmp_path / "analysis.yaml"
    data = tmp_path / "sample.parquet"
    artifact = tmp_path / "model.manifest.json"
    configuration.write_text("schema_version: '2.0'\n", encoding="utf-8")
    data.write_bytes(b"sample-a")
    artifact.write_text('{"artifact": "a"}\n', encoding="utf-8")
    provenance = VALIDATION.write_reuse_provenance(
        tmp_path / "reuse.json",
        configuration_path=configuration,
        data_paths={"sample": data},
        artifact_paths={"model": artifact},
        metadata={"features": ["x"]},
    )

    valid, reasons = VALIDATION.verify_reuse_provenance(
        provenance,
        configuration_path=configuration,
        data_paths={"sample": data},
        artifact_paths={"model": artifact},
    )
    assert valid
    assert reasons == ()

    data.write_bytes(b"sample-b")
    valid, reasons = VALIDATION.verify_reuse_provenance(
        provenance,
        configuration_path=configuration,
        data_paths={"sample": data},
        artifact_paths={"model": artifact},
    )
    assert not valid
    assert reasons == ("data file 'sample' changed",)


def test_reconstructed_mixture_matches_generator_moments() -> None:
    components = GENERATOR.signal_components()
    reconstructed = VALIDATION.reconstructed_components(
        components,
        scale=GENERATOR.SCALE,
        resolution=GENERATOR.RESOLUTION,
    )
    fraction, latent_mean, latent_covariance = components[0]
    observed_fraction, mean, covariance = reconstructed[0]
    transform = np.diag(GENERATOR.SCALE)

    assert observed_fraction == fraction
    np.testing.assert_allclose(mean, GENERATOR.SCALE * latent_mean)
    np.testing.assert_allclose(
        covariance,
        transform @ latent_covariance @ transform.T + np.diag(GENERATOR.RESOLUTION**2),
    )


def test_truth_density_and_binned_calibration_close_exactly() -> None:
    components = VALIDATION.reconstructed_components(
        GENERATOR.background_components(),
        scale=GENERATOR.SCALE,
        resolution=GENERATOR.RESOLUTION,
    )
    values = GENERATOR.sample_mixture(
        components,
        500,
        np.random.default_rng(12),
    )
    truth = VALIDATION.mixture_log_density(values, components)
    reference = VALIDATION.reference_log_density(
        {"signal": truth, "background": truth},
        fractions={"signal": 0.5, "background": 0.5},
    )
    calibration, edges = VALIDATION.binned_log_density_calibration(
        truth,
        reference,
        bins=10,
    )

    np.testing.assert_allclose(reference, truth)
    np.testing.assert_allclose(calibration["delta_mean"], 0.0, atol=1.0e-12)
    assert len(edges) == 11


def test_conditional_density_uses_manifest_selection_efficiency() -> None:
    inclusive = np.asarray([-3.0, -2.0, -1.0])
    selected = VALIDATION.conditional_log_density(
        inclusive,
        selection_efficiency=0.25,
    )
    np.testing.assert_allclose(selected, inclusive - np.log(0.25))
    with pytest.raises(ValueError, match="selection_efficiency"):
        VALIDATION.conditional_log_density(
            inclusive,
            selection_efficiency=0.0,
        )


def test_extended_and_asimov_mu_scans_minimize_at_truth() -> None:
    rng = np.random.default_rng(4)
    rows = 2000
    signal_ratio = rng.lognormal(0.0, 0.3, rows)
    background_ratio = rng.lognormal(0.0, 0.2, rows)
    signal_ratio /= np.mean(signal_ratio)
    background_ratio /= np.mean(background_ratio)
    signal_yield = 40.0
    background_yield = 100.0
    truth_mu = 1.0
    weights = (
        truth_mu * signal_yield * signal_ratio + background_yield * background_ratio
    ) / rows
    result = SimpleNamespace(
        normalized_ratios={
            "signal": signal_ratio,
            "background": background_ratio,
        },
        events=SimpleNamespace(weights=weights),
    )
    grid = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])

    asimov_curve = VALIDATION.asimov_mu_scan(
        result,
        grid,
        signal_yield=signal_yield,
        background_yield=background_yield,
        truth_mu=truth_mu,
    )
    direct_curve = VALIDATION.extended_mu_scan(
        signal_log_density=np.log(signal_ratio),
        background_log_density=np.log(background_ratio),
        event_weights=weights,
        signal_yield=signal_yield,
        background_yield=background_yield,
        mu_values=grid,
    )

    assert int(np.argmin(asimov_curve)) == 2
    assert int(np.argmin(direct_curve)) == 2
    assert asimov_curve[2] == pytest.approx(0.0, abs=1.0e-12)
    assert direct_curve[2] == pytest.approx(0.0, abs=1.0e-12)


def test_compressed_toy_model_closes_and_returns_plotting_columns() -> None:
    rng = np.random.default_rng(18)
    support = rng.normal(size=20_000)
    signal_ratio = np.exp(0.8 * support - 0.5 * 0.8**2)
    background_ratio = np.ones_like(signal_ratio)
    model = VALIDATION.build_compressed_toy_model(
        signal_ratio,
        background_ratio,
        signal_yield=40.0,
        background_yield=1_000.0,
        bins=64,
    )

    assert model.bins == 64
    assert np.sum(model.signal_probability) == pytest.approx(1.0)
    assert np.sum(model.background_probability) == pytest.approx(1.0)
    scan = model.asimov_scan(np.asarray([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert int(np.argmin(scan)) == 2
    assert scan[2] == pytest.approx(0.0, abs=1.0e-12)

    toys = VALIDATION.run_compressed_toys(
        model,
        hypotheses=(0.0, 1.0),
        n_toys=200,
        batch_size=50,
        seed=19,
    )
    assert len(toys) == 400
    assert set(
        (
            "mu_true",
            "toy",
            "n_events",
            "n_signal",
            "n_background",
            "mu_hat",
            "t_mu",
            "q_zero",
            "information",
        )
    ).issubset(toys)
    assert np.isfinite(toys.to_numpy()).all()
    assert (toys[["mu_hat", "t_mu", "q_zero", "information"]] >= 0).all().all()
    assert set(toys["mu_true"]) == {0.0, 1.0}
