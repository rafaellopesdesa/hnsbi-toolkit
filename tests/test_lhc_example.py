from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from hnsbi import Project

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "lhc_analysis"


def _generator():
    specification = importlib.util.spec_from_file_location(
        "hnsbi_lhc_generator",
        EXAMPLE / "generate_distributions.py",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    example_path = str(EXAMPLE)
    inserted = example_path not in sys.path
    if inserted:
        sys.path.insert(0, example_path)
    try:
        specification.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(example_path)
    return module


def _runner():
    specification = importlib.util.spec_from_file_location(
        "hnsbi_lhc_runner",
        EXAMPLE / "run_analysis.py",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_lhc_yaml_exposes_all_systematics_and_inference() -> None:
    project = Project.load(EXAMPLE / "analysis.yaml")
    frequentist = project.config.frequentist
    assert frequentist is not None
    assert frequentist["reference"]["path"].endswith("_presel.parquet")
    assert all(
        sample["source"]["path"].endswith("_presel.parquet")
        and sample["source"]["split_column"] == "preselection_split"
        and sample["nominal_yield"] == {"kind": "source_weight_sum"}
        for sample in frequentist["samples"]
    )
    assert all(
        variation[direction]["path"].endswith("_presel.parquet")
        for systematic in frequentist["systematics"]
        for variation in systematic["variations"]
        for direction in ("up", "down")
    )
    assert frequentist["asimov"]["normalization_source"]["path"].endswith(
        "_presel.parquet"
    )
    assert frequentist["ratios"]["backend"] == "native"
    assert {item["parameter"] for item in frequentist["systematics"]} == {
        "response",
        "resolution",
        "theory",
    }
    assert frequentist["inference"]["backend"] == "jax_iminuit"
    assert frequentist["inference"]["pyhf_projection"]["scan"][0] == 0.0


def test_lhc_runner_exposes_full_optional_campaigns() -> None:
    runner = _runner()
    parameters = inspect.signature(runner.run).parameters
    assert {"pyhf_toys", "native_toys", "nis"}.issubset(parameters)
    source = (EXAMPLE / "run_analysis.py").read_text(encoding="utf-8")
    assert "project.train_nis_asimov(" in source
    assert "project.generate_configured_toys(" in source
    assert 'calctype="toybased"' in source
    assert "systematic_modifiers=" not in source


def test_lhc_generator_response_resolution_and_theory_anchors(tmp_path) -> None:
    generator = _generator()
    paths = generator.generate(
        tmp_path,
        signal_events=3_000,
        background_events=9_000,
        reference_events=2_400,
        seed=17,
    )
    assert len(paths) == 27
    assert "preselection_manifest" in paths
    assert sum(name.endswith("_presel") for name in paths) == 13
    nominal = pd.read_parquet(paths["signal"])
    response = pd.read_parquet(paths["signal_response_up"])
    resolution = pd.read_parquet(paths["signal_resolution_up"])
    theory_up = pd.read_parquet(paths["signal_theory_up"])
    theory_down = pd.read_parquet(paths["signal_theory_down"])
    latent = nominal.loc[:, list(generator.LATENT)].to_numpy()
    reconstructed = nominal.loc[:, list(generator.FEATURES)].to_numpy()
    expected_response = reconstructed + 0.10 * latent * generator.SCALE
    np.testing.assert_allclose(
        response.loc[:, list(generator.FEATURES)].to_numpy(),
        expected_response,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    nominal_residual = reconstructed - latent * generator.SCALE
    varied_residual = (
        resolution.loc[:, list(generator.FEATURES)].to_numpy()
        - latent * generator.SCALE
    )
    np.testing.assert_allclose(varied_residual, 1.25 * nominal_residual)
    latent_shift = (
        theory_up.loc[:, list(generator.LATENT)].mean().to_numpy()
        - theory_down.loc[:, list(generator.LATENT)].mean().to_numpy()
    )
    assert float(np.dot(latent_shift, generator.THEORY_SHIFT)) > 0
