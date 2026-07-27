# Data contract

The ordered top-level `features` list is the observation-space signature for
all models. Parameter columns used by the dual workflow are listed separately
in `bayesian.theta_features`.

Supported source kinds are:

- `parquet`: one path or a list of Parquet paths;
- `arrow_ipc`: Arrow IPC file input;
- `pyarrow`: an in-memory object registered from Python;
- `awkward`: an in-memory Awkward array registered from Python.

File configurations contain exactly one of `path`, `paths`, or `registry_key`.
Streaming readers request only required columns and emit bounded-size Arrow
record batches.

Optional source metadata includes:

- `weight_column`;
- stable `event_id_column` for reproducible hashing and splits;
- `split_column` for externally prepared independent splits;
- `log_density_column` for supplied design-density values.

## Weights

The data layer preserves finite signed weights. Algorithms that require a
probability measure—maximum-likelihood flow training, Poisson sampling, and
ordinary importance resampling—require nonnegative input weights and reject
incompatible data. They never take an absolute value silently.

## Bayesian scientific splits

For Bayesian proposal sources, a configured `split_column` accepts exactly the
case-sensitive string labels `train`, `validation`, and `holdout`.

- `train` rows may update parameters or fitted preprocessing.
- `validation` rows may select an early-stopping checkpoint.
- `holdout` rows are reserved for final diagnostics and ONNX parity.

The split is resolved on simulator rows before positive/negative classifier
pairs are constructed. Both members of a matched pair therefore inherit one
label, and no simulation identifier can cross partitions. Every proposal
source used for fitting must contain at least one `train` row.

`bayesian.datasets.validation` is independent of the four proposal-training
sources. If it has no split column, it supplies `validation` when the stage
source has none; otherwise it becomes an additional `holdout`. If it has a
split column, it may contain only `validation` and `holdout` rows. This
independent dataset is evaluated by both conditional flows, both residual
ratios, and the conditional normalizer; its parameter design must therefore
have support appropriate for those checks.

## In-memory registration

YAML and JSON cannot embed a live PyArrow or Awkward object. Use a logical
registry key:

```json
{
  "kind": "pyarrow",
  "registry_key": "reference"
}
```

and associate `reference` with the object through the Python data registry.
The same validated configuration can therefore be used with local files or a
notebook-created table.
