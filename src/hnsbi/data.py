"""Memory-bounded event input for Parquet, PyArrow, Awkward and pandas."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EventBatch:
    """One dense event batch with optional event weights and row identifiers."""

    values: np.ndarray
    weights: np.ndarray
    row_ids: np.ndarray
    features: tuple[str, ...]
    columns: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        weights = np.asarray(self.weights)
        row_ids = np.asarray(self.row_ids)
        if values.ndim != 2:
            raise ValueError("values must be two-dimensional.")
        if values.shape[1] != len(self.features):
            raise ValueError("values columns do not match features.")
        if len(weights) != len(values) or len(row_ids) != len(values):
            raise ValueError("weights and row_ids must align with values.")
        if not np.isfinite(values).all() or not np.isfinite(weights).all():
            raise ValueError("Event values and weights must be finite.")
        columns = {name: np.asarray(column) for name, column in self.columns.items()}
        for name, column in columns.items():
            if len(column) != len(values):
                raise ValueError(
                    f"Auxiliary column {name!r} does not align with values."
                )
        object.__setattr__(self, "columns", columns)


@dataclass
class WeightedEvents:
    """A weighted event measure returned by Asimov and Bayesian operations."""

    values: np.ndarray
    weights: np.ndarray
    features: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    columns: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        self.features = tuple(self.features)
        if self.values.ndim != 2:
            raise ValueError("values must be two-dimensional.")
        if self.values.shape[1] != len(self.features):
            raise ValueError("values columns do not match features.")
        if len(self.values) != len(self.weights):
            raise ValueError("weights must contain one value per event.")
        if not np.isfinite(self.values).all():
            raise ValueError("Event values must be finite.")
        if not np.isfinite(self.weights).all():
            raise ValueError("Event weights must be finite.")
        for name, values in list(self.columns.items()):
            array = np.asarray(values)
            if len(array) != len(self.values):
                raise ValueError(f"Column {name!r} has the wrong row count.")
            self.columns[name] = array

    @property
    def raw_count(self) -> int:
        return len(self.values)

    @property
    def expected_count(self) -> float:
        return float(np.sum(self.weights, dtype=np.float64))

    @property
    def ess(self) -> float:
        from .diagnostics import effective_sample_size

        return effective_sample_size(self.weights)

    def probability_weights(self) -> np.ndarray:
        if np.any(self.weights < 0):
            raise ValueError("Probability weights cannot be negative.")
        total = float(np.sum(self.weights))
        if not total > 0:
            raise ValueError("Probability weights must have positive sum.")
        return self.weights / total

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Install hnsbi-toolkit[data] for pandas output.") from exc
        data = {name: self.values[:, index] for index, name in enumerate(self.features)}
        data["weight"] = self.weights
        data.update(self.columns)
        return pd.DataFrame(data)

    def write_parquet(self, path: str | Path) -> Path:
        from .diagnostics import json_safe

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_pandas().to_parquet(path, index=False)
        sidecar = path.with_suffix(path.suffix + ".metadata.json")
        sidecar.write_text(
            json.dumps(
                json_safe(self.metadata),
                indent=2,
                sort_keys=True,
                default=str,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return path


class DataSource:
    """Uniform, streaming access to supported event containers.

    A path is interpreted as Parquet and read with ``ParquetFile.iter_batches``.
    In-memory NumPy, pandas, PyArrow, and Awkward values are sliced without
    first converting the complete dataset to a pandas DataFrame.
    """

    def __init__(
        self,
        source: str | Path | Any,
        *,
        features: Sequence[str],
        weight: str | None = "weight",
        row_id: str | None = None,
        auxiliary: Sequence[str] = (),
        format: str | None = None,
        batch_size: int = 65_536,
    ) -> None:
        self.source = source
        self.features = tuple(features)
        self.weight = weight
        self.row_id = row_id
        self.auxiliary = tuple(auxiliary)
        self.format = format
        self.batch_size = int(batch_size)
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("features must be non-empty and unique.")
        if self.format not in {None, "parquet", "arrow_ipc"}:
            raise ValueError("format must be 'parquet', 'arrow_ipc', or None.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if len(set(self.auxiliary)) != len(self.auxiliary):
            raise ValueError("auxiliary column names must be unique.")
        overlap = set(self.auxiliary).intersection(self.features)
        if overlap:
            raise ValueError(
                f"Auxiliary columns overlap model features {sorted(overlap)}."
            )

    def iter_batches(self, batch_size: int | None = None) -> Iterator[EventBatch]:
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if isinstance(self.source, (str, Path)):
            path = Path(self.source)
            if self.format == "arrow_ipc":
                yield from self._iter_arrow_ipc(path, batch_size)
            else:
                yield from self._iter_parquet(path, batch_size)
            return
        if isinstance(self.source, (list, tuple)):
            offset = 0
            for item in self.source:
                if not isinstance(item, (str, Path)):
                    raise TypeError(
                        "A sequence data source must contain only file paths."
                    )
                nested = DataSource(
                    item,
                    features=self.features,
                    weight=self.weight,
                    row_id=self.row_id,
                    auxiliary=self.auxiliary,
                    format=self.format,
                    batch_size=self.batch_size,
                )
                rows = 0
                for batch in nested.iter_batches(batch_size):
                    yield EventBatch(
                        batch.values,
                        batch.weights,
                        (
                            batch.row_ids + offset
                            if self.row_id is None
                            else batch.row_ids
                        ),
                        batch.features,
                        batch.columns,
                    )
                    rows += len(batch.values)
                offset += rows
            return
        yield from self._iter_memory(self.source, batch_size)

    def _iter_parquet(self, path: Path, batch_size: int) -> Iterator[EventBatch]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Parquet streaming requires hnsbi-toolkit[data]."
            ) from exc
        parquet = pq.ParquetFile(path)
        columns = list(self.features)
        if self.weight is not None:
            columns.append(self.weight)
        if self.row_id is not None:
            columns.append(self.row_id)
        columns.extend(self.auxiliary)
        columns = list(dict.fromkeys(columns))
        missing = set(columns).difference(parquet.schema.names)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}.")
        offset = 0
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=columns, use_threads=True
        ):
            arrays = [
                np.asarray(batch.column(name).to_numpy(zero_copy_only=False))
                for name in self.features
            ]
            values = np.column_stack(arrays).astype(np.float32, copy=False)
            if self.weight is None:
                weights = np.ones(len(values), dtype=np.float64)
            else:
                weights = np.asarray(
                    batch.column(self.weight).to_numpy(zero_copy_only=False),
                    dtype=np.float64,
                )
            row_ids = (
                np.arange(offset, offset + len(values), dtype=np.int64)
                if self.row_id is None
                else np.asarray(
                    batch.column(self.row_id).to_numpy(zero_copy_only=False)
                )
            )
            offset += len(values)
            auxiliary = {
                name: np.asarray(batch.column(name).to_numpy(zero_copy_only=False))
                for name in self.auxiliary
            }
            yield EventBatch(
                values,
                weights,
                row_ids,
                self.features,
                auxiliary,
            )

    def _iter_arrow_ipc(self, path: Path, batch_size: int) -> Iterator[EventBatch]:
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise ImportError(
                "Arrow IPC streaming requires hnsbi-toolkit[data]."
            ) from exc
        source = None
        try:
            source = pa.memory_map(path, "r")
            try:
                file_reader = ipc.open_file(source)
                record_batches = (
                    file_reader.get_batch(index)
                    for index in range(file_reader.num_record_batches)
                )
            except pa.ArrowInvalid:
                source.close()
                source = pa.input_stream(path)
                record_batches = iter(ipc.open_stream(source))
            offset = 0
            for record_batch in record_batches:
                for start in range(0, len(record_batch), batch_size):
                    sliced = record_batch.slice(start, batch_size)
                    table = pa.Table.from_batches([sliced])
                    nested = DataSource(
                        table,
                        features=self.features,
                        weight=self.weight,
                        row_id=self.row_id,
                        auxiliary=self.auxiliary,
                    )
                    batch = next(nested.iter_batches(batch_size))
                    row_ids = (
                        batch.row_ids + offset if self.row_id is None else batch.row_ids
                    )
                    yield EventBatch(
                        batch.values,
                        batch.weights,
                        row_ids,
                        batch.features,
                        batch.columns,
                    )
                    offset += len(batch.values)
        except (OSError, pa.ArrowInvalid) as exc:
            raise ValueError(f"Could not read Arrow IPC data from {path}.") from exc
        finally:
            if source is not None:
                source.close()

    def _iter_memory(self, source: Any, batch_size: int) -> Iterator[EventBatch]:
        if isinstance(source, np.ndarray):
            if self.auxiliary:
                raise ValueError(
                    "A NumPy source cannot resolve named auxiliary columns."
                )
            if self.row_id is not None:
                raise ValueError("A NumPy source cannot resolve a named row_id column.")
            if source.ndim != 2 or source.shape[1] != len(self.features):
                raise ValueError("NumPy input must have shape (n_events, n_features).")
            n_rows = len(source)

            def take(start: int, stop: int):
                return (
                    np.asarray(source[start:stop], dtype=np.float32),
                    np.ones(stop - start, dtype=np.float64),
                )

        elif source.__class__.__module__.startswith("pandas"):
            missing = set(self.features).difference(source.columns)
            if self.weight is not None and self.weight not in source.columns:
                missing.add(self.weight)
            if self.row_id is not None and self.row_id not in source.columns:
                missing.add(self.row_id)
            missing.update(set(self.auxiliary).difference(source.columns))
            if missing:
                raise ValueError(f"Input is missing columns {sorted(missing)}.")
            n_rows = len(source)

            def take(start: int, stop: int):
                frame = source.iloc[start:stop]
                values = frame.loc[:, self.features].to_numpy(dtype=np.float32)
                weights = (
                    np.ones(len(frame), dtype=np.float64)
                    if self.weight is None
                    else frame.loc[:, self.weight].to_numpy(dtype=np.float64)
                )
                row_ids = (
                    np.arange(start, stop, dtype=np.int64)
                    if self.row_id is None
                    else frame.loc[:, self.row_id].to_numpy()
                )
                auxiliary = {
                    name: frame.loc[:, name].to_numpy() for name in self.auxiliary
                }
                return values, weights, row_ids, auxiliary

        elif source.__class__.__module__.startswith("pyarrow"):
            names = set(source.schema.names)
            required = set(self.features)
            if self.weight is not None:
                required.add(self.weight)
            if self.row_id is not None:
                required.add(self.row_id)
            required.update(self.auxiliary)
            missing = required.difference(names)
            if missing:
                raise ValueError(f"Input is missing columns {sorted(missing)}.")
            n_rows = len(source)

            def take(start: int, stop: int):
                table = source.slice(start, stop - start)
                values = np.column_stack(
                    [
                        np.asarray(
                            table[name].to_numpy(zero_copy_only=False),
                            dtype=np.float32,
                        )
                        for name in self.features
                    ]
                )
                weights = (
                    np.ones(len(values), dtype=np.float64)
                    if self.weight is None
                    else np.asarray(
                        table[self.weight].to_numpy(zero_copy_only=False),
                        dtype=np.float64,
                    )
                )
                row_ids = (
                    np.arange(start, stop, dtype=np.int64)
                    if self.row_id is None
                    else np.asarray(table[self.row_id].to_numpy(zero_copy_only=False))
                )
                auxiliary = {
                    name: np.asarray(table[name].to_numpy(zero_copy_only=False))
                    for name in self.auxiliary
                }
                return values, weights, row_ids, auxiliary

        elif source.__class__.__module__.startswith("awkward"):
            try:
                import awkward as ak
            except ImportError as exc:
                raise ImportError(
                    "Awkward input requires hnsbi-toolkit[data]."
                ) from exc
            fields = set(ak.fields(source))
            required = set(self.features)
            if self.weight is not None:
                required.add(self.weight)
            if self.row_id is not None:
                required.add(self.row_id)
            required.update(self.auxiliary)
            missing = required.difference(fields)
            if missing:
                raise ValueError(f"Input is missing fields {sorted(missing)}.")
            n_rows = len(source)

            def take(start: int, stop: int):
                array = source[start:stop]
                values = np.column_stack(
                    [
                        np.asarray(ak.to_numpy(array[name]), dtype=np.float32)
                        for name in self.features
                    ]
                )
                weights = (
                    np.ones(len(values), dtype=np.float64)
                    if self.weight is None
                    else np.asarray(ak.to_numpy(array[self.weight]), dtype=np.float64)
                )
                row_ids = (
                    np.arange(start, stop, dtype=np.int64)
                    if self.row_id is None
                    else np.asarray(ak.to_numpy(array[self.row_id]))
                )
                auxiliary = {
                    name: np.asarray(ak.to_numpy(array[name]))
                    for name in self.auxiliary
                }
                return values, weights, row_ids, auxiliary

        else:
            raise TypeError(
                "Unsupported data source. Use Parquet, NumPy, pandas, "
                "PyArrow, or Awkward."
            )

        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            taken = take(start, stop)
            if len(taken) == 2:
                values, weights = taken
                row_ids = np.arange(start, stop, dtype=np.int64)
                auxiliary = {}
            elif len(taken) == 3:
                values, weights, row_ids = taken
                auxiliary = {}
            else:
                values, weights, row_ids, auxiliary = taken
            yield EventBatch(
                values,
                weights,
                row_ids,
                self.features,
                auxiliary,
            )

    def materialize(
        self, *, batch_size: int | None = None, max_events: int | None = None
    ) -> EventBatch:
        values: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        row_ids: list[np.ndarray] = []
        columns: dict[str, list[np.ndarray]] = {name: [] for name in self.auxiliary}
        retained = 0
        for batch in self.iter_batches(batch_size):
            take = len(batch.values)
            if max_events is not None:
                take = min(take, int(max_events) - retained)
            if take <= 0:
                break
            values.append(batch.values[:take])
            weights.append(batch.weights[:take])
            row_ids.append(batch.row_ids[:take])
            for name in columns:
                columns[name].append(batch.columns[name][:take])
            retained += take
        if not values:
            return EventBatch(
                np.empty((0, len(self.features)), dtype=np.float32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
                self.features,
                {name: np.empty(0, dtype=np.float64) for name in self.auxiliary},
            )
        return EventBatch(
            np.concatenate(values),
            np.concatenate(weights),
            np.concatenate(row_ids),
            self.features,
            {name: np.concatenate(chunks) for name, chunks in columns.items()},
        )
