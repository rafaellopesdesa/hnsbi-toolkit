from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hnsbi import DataSource, Project
from hnsbi.bayes import ProposalDataset


def test_pandas_auxiliary_columns_survive_batches_and_materialization() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "x": np.arange(6, dtype=np.float64),
            "y": np.arange(10, 16, dtype=np.float64),
            "weight": np.linspace(1.0, 2.0, 6),
            "simulation_id": np.arange(100, 106),
            "split": ["train", "train", "validation"] * 2,
            "log_rho": np.linspace(-3.0, -0.5, 6),
        }
    )
    source = DataSource(
        frame,
        features=("x", "y"),
        weight="weight",
        row_id="simulation_id",
        auxiliary=("split", "log_rho"),
    )

    batches = list(source.iter_batches(batch_size=2))
    assert [len(batch.values) for batch in batches] == [2, 2, 2]
    assert all(tuple(batch.columns) == ("split", "log_rho") for batch in batches)
    np.testing.assert_array_equal(
        np.concatenate([batch.columns["split"] for batch in batches]),
        frame["split"].to_numpy(),
    )

    materialized = source.materialize(batch_size=2, max_events=5)
    np.testing.assert_array_equal(
        materialized.row_ids, frame["simulation_id"].to_numpy()[:5]
    )
    np.testing.assert_array_equal(
        materialized.columns["split"], frame["split"].to_numpy()[:5]
    )
    np.testing.assert_allclose(
        materialized.columns["log_rho"], frame["log_rho"].to_numpy()[:5]
    )


def test_data_source_uses_configured_batch_size_by_default() -> None:
    values = np.arange(14, dtype=np.float32).reshape(7, 2)
    source = DataSource(
        values,
        features=("x", "y"),
        weight=None,
        batch_size=3,
    )

    assert [len(batch.values) for batch in source.iter_batches()] == [3, 3, 1]
    assert len(source.materialize().values) == 7
    assert [len(batch.values) for batch in source.iter_batches(4)] == [4, 3]


def test_project_passes_source_batch_size_to_data_source() -> None:
    values = np.arange(10, dtype=np.float32).reshape(5, 2)
    project = Project(
        SimpleNamespace(features=("x", "y")),
        registry={"events": values},
    )
    source = project.data_source(
        {
            "kind": "pyarrow",
            "registry_key": "events",
            "batch_size": 2,
        }
    )

    assert source.batch_size == 2
    assert [len(batch.values) for batch in source.iter_batches()] == [2, 2, 1]


def test_parquet_sequence_preserves_auxiliary_columns_and_global_row_ids(
    tmp_path,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    paths = []
    for index, offset in enumerate((0, 3)):
        path = tmp_path / f"part-{index}.parquet"
        pq.write_table(
            pa.table(
                {
                    "x": np.arange(offset, offset + 3, dtype=np.float64),
                    "weight": np.ones(3),
                    "split": [f"part-{index}"] * 3,
                    "log_rho": np.arange(offset, offset + 3) / 10.0,
                }
            ),
            path,
        )
        paths.append(path)

    batch = DataSource(
        paths,
        features=("x",),
        weight="weight",
        auxiliary=("split", "log_rho"),
    ).materialize(batch_size=2)

    np.testing.assert_array_equal(batch.row_ids, np.arange(6))
    np.testing.assert_array_equal(
        batch.columns["split"],
        np.asarray(["part-0"] * 3 + ["part-1"] * 3),
    )
    np.testing.assert_allclose(batch.columns["log_rho"], np.arange(6) / 10.0)


def test_project_proposal_dataset_preserves_split_and_log_density() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "theta": [-1.0, -0.5, 0.5, 1.0],
            "x": [1.0, 0.25, 0.25, 1.0],
            "simulation_id": [11, 12, 13, 14],
            "split": ["train", "validation", "train", "validation"],
            "log_rho": [-2.0, -1.0, -1.5, -2.5],
        }
    )
    dataset = {
        "kind": "pyarrow",
        "registry_key": "proposal",
        "event_id_column": "simulation_id",
        "split_column": "split",
        "log_density_column": "log_rho",
        "batch_size": 2,
    }
    project = Project(
        SimpleNamespace(
            features=("x",),
            bayesian={
                "theta_features": ["theta"],
                "datasets": {
                    "rho": dataset,
                    "nu": dataset,
                    "kappa": dataset,
                },
            },
        ),
        registry={"proposal": frame},
    )

    proposal = project.proposal_dataset("rho")
    np.testing.assert_array_equal(proposal.simulation_ids, [11, 12, 13, 14])
    np.testing.assert_array_equal(proposal.split_values, frame["split"].to_numpy())
    np.testing.assert_allclose(proposal.log_density, frame["log_rho"].to_numpy())

    subset = proposal.subset(np.asarray([3, 1]))
    np.testing.assert_array_equal(subset.simulation_ids, [14, 12])
    np.testing.assert_array_equal(subset.split_values, ["validation", "validation"])
    np.testing.assert_allclose(subset.log_density, [-2.5, -1.0])


@pytest.mark.parametrize(
    ("split_values", "log_density", "message"),
    [
        (np.asarray(["train"]), None, "split_values"),
        (None, np.asarray([0.0, np.nan]), "log_density"),
    ],
)
def test_proposal_dataset_validates_preserved_columns(
    split_values,
    log_density,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ProposalDataset(
            theta=np.zeros((2, 1)),
            observation=np.zeros((2, 1)),
            simulation_ids=np.asarray([0, 1]),
            design="rho",
            split_values=split_values,
            log_density=log_density,
        )
