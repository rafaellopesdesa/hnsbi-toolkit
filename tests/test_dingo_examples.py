from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from hnsbi import Project
from hnsbi.config import ToolkitConfig

pq = pytest.importorskip("pyarrow.parquet")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "dingo_bbh": ROOT / "examples" / "dingo_bbh",
    "dingo_bns": ROOT / "examples" / "dingo_bns",
}
TRAINING_DESIGNS = ("rho", "rho_residual", "nu", "kappa")
ALL_DESIGNS = TRAINING_DESIGNS + ("validation",)
NOTEBOOK_REQUIREMENTS = (
    "project.train_dual()",
    "load_dual_model(",
    "hnpe_log_weights(",
    "hnde_log_weights(",
    "geometric_consensus(",
    "posterior_normalization_diagnostic(",
    "conditional_normalization_diagnostic(",
    "route_diagnostic(",
    "estimate_evidence(",
    "bridge_diagnostic(",
    "exact_log_likelihood(",
    ".ess",
)


def _load_generator(example_name: str) -> ModuleType:
    path = EXAMPLES[example_name] / "generate_data.py"
    specification = importlib.util.spec_from_file_location(
        f"_hnsbi_{example_name}_generator",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _configured_project(
    example_name: str,
    paths: dict[str, Path],
) -> Project:
    configuration = deepcopy(
        ToolkitConfig.load(EXAMPLES[example_name] / "dual.yaml").raw
    )
    for design in ALL_DESIGNS:
        configuration["bayesian"]["datasets"][design]["path"] = str(
            paths[design].resolve()
        )
    return Project.load(configuration)


@pytest.mark.parametrize("example_name", tuple(EXAMPLES))
def test_small_generation_is_finite_disjoint_and_deterministic(
    example_name: str,
    tmp_path: Path,
) -> None:
    generator = _load_generator(example_name)
    first = generator.generate_datasets(
        tmp_path / "first",
        rows_per_design=32,
        validation_rows=12,
        seed=8675309,
    )
    second = generator.generate_datasets(
        tmp_path / "second",
        rows_per_design=32,
        validation_rows=12,
        seed=8675309,
    )

    all_ids: set[int] = set()
    for design in ALL_DESIGNS:
        table = pq.read_table(first[design])
        repeated = pq.read_table(second[design])
        assert table.equals(repeated)

        expected_rows = 12 if design == "validation" else 32
        assert table.num_rows == expected_rows
        dense = np.column_stack(
            [
                table[name].to_numpy(zero_copy_only=False)
                for name in (
                    tuple(generator.THETA_FEATURES)
                    + tuple(generator.OBSERVATION_FEATURES)
                )
            ]
        )
        assert dense.shape == (
            expected_rows,
            len(generator.THETA_FEATURES) + len(generator.OBSERVATION_FEATURES),
        )
        assert np.isfinite(dense).all()

        ids = {int(value) for value in table["simulation_id"].to_pylist()}
        assert len(ids) == expected_rows
        assert all_ids.isdisjoint(ids)
        all_ids.update(ids)

        if design == "validation":
            assert "split" not in table.column_names
        else:
            assert set(table["split"].to_pylist()) == {
                "train",
                "validation",
                "holdout",
            }

    theta = np.asarray(
        [
            pq.read_table(first["validation"], columns=[name])[name][0].as_py()
            for name in generator.THETA_FEATURES
        ],
        dtype=np.float64,
    )
    observation = np.asarray(
        [
            pq.read_table(first["validation"], columns=[name])[name][0].as_py()
            for name in generator.OBSERVATION_FEATURES
        ],
        dtype=np.float64,
    )
    mean = generator.waveform_mean(theta)
    log_likelihood = generator.exact_log_likelihood(theta, observation)
    assert mean.shape == (1, len(generator.OBSERVATION_FEATURES))
    assert log_likelihood.shape == (1,)
    assert np.isfinite(mean).all()
    assert np.isfinite(log_likelihood).all()


@pytest.mark.parametrize("example_name", tuple(EXAMPLES))
def test_yaml_handoff_materializes_dual_data_without_training(
    example_name: str,
    tmp_path: Path,
) -> None:
    generator = _load_generator(example_name)
    paths = generator.generate_datasets(
        tmp_path / "data",
        rows_per_design=32,
        validation_rows=12,
        seed=314159,
    )
    project = _configured_project(example_name, paths)
    data = project.dual_training_data()

    assert project.config.raw["schema_version"] == "2.0"
    assert project.config.features == tuple(generator.OBSERVATION_FEATURES)
    for dataset in (
        data.rho_flow,
        data.rho_ratio,
        data.nu_flow,
        data.kappa_ratio,
    ):
        assert dataset.theta.shape == (32, len(generator.THETA_FEATURES))
        assert dataset.observation.shape == (
            32,
            len(generator.OBSERVATION_FEATURES),
        )
        assert set(dataset.split_values) == {
            "train",
            "validation",
            "holdout",
        }
        assert np.isfinite(dataset.theta).all()
        assert np.isfinite(dataset.observation).all()

    assert data.validation is not None
    assert data.validation.theta.shape == (12, len(generator.THETA_FEATURES))
    assert data.validation.observation.shape == (
        12,
        len(generator.OBSERVATION_FEATURES),
    )
    assert data.validation.split_values is None

    distribution = project.design_distribution("rho")
    np.testing.assert_allclose(distribution.low, generator.PRIOR_LOW)
    np.testing.assert_allclose(distribution.high, generator.PRIOR_HIGH)


@pytest.mark.parametrize("example_name", tuple(EXAMPLES))
def test_notebook_is_complete_opt_in_dual_workflow(example_name: str) -> None:
    notebook_path = EXAMPLES[example_name] / f"{example_name}_dual.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    cells = notebook["cells"]
    identifiers = [cell["id"] for cell in cells]
    assert len(identifiers) == len(set(identifiers))

    source = "\n".join("".join(cell.get("source", ())) for cell in cells)
    assert source.count("RUN_FULL_WORKFLOW = False") == 1
    assert 'PROFILE = "quick" if RUN_FULL_WORKFLOW else "smoke"' in source
    assert "RUN_TRAINING" not in source
    for requirement in NOTEBOOK_REQUIREMENTS:
        assert requirement in source

    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile(
            "".join(cell.get("source", ())),
            f"{notebook_path.name}:cell-{index}",
            "exec",
        )
