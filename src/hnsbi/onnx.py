"""Optional ONNX export, inference, and parity helpers.

Heavy dependencies are imported only when the corresponding operation is
requested.  Importing :mod:`hnsbi.onnx` therefore remains safe in lightweight
analysis environments.
"""

from __future__ import annotations

import copy
import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_artifact_manifest

ArrayLike = np.ndarray | Sequence[float]


class OptionalDependencyError(ImportError):
    """An optional toolkit backend is required for the requested operation."""


def require_optional(
    module: str,
    *,
    extra: str,
    purpose: str | None = None,
) -> Any:
    """Import an optional dependency or raise an actionable error."""

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        detail = f" for {purpose}" if purpose else ""
        raise OptionalDependencyError(
            f"Optional dependency {module!r} is required{detail}. "
            f"Install it with `pip install 'hnsbi-toolkit[{extra}]'`."
        ) from exc


def manifest_path_for(model_path: str | Path) -> Path:
    """Return the conventional manifest sidecar path for an exported model."""

    path = Path(model_path)
    return path.with_suffix(path.suffix + ".manifest.json")


@dataclass(frozen=True)
class OnnxParityReport:
    """Numerical comparison between a native model and its ONNX export."""

    passed: bool
    max_absolute_error: float
    mean_absolute_error: float
    max_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_close(self) -> None:
        """Raise ``AssertionError`` when the parity tolerance was exceeded."""

        if not self.passed:
            raise AssertionError(
                "ONNX parity check failed: "
                f"max abs={self.max_absolute_error:.3g}, "
                f"max rel={self.max_relative_error:.3g}, "
                f"atol={self.absolute_tolerance:.3g}, "
                f"rtol={self.relative_tolerance:.3g}."
            )


def compare_outputs(
    reference: ArrayLike,
    candidate: ArrayLike,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> OnnxParityReport:
    """Compare native and ONNX outputs and return detailed error metrics."""

    expected = np.asarray(reference)
    observed = np.asarray(candidate)
    if expected.shape != observed.shape:
        raise ValueError(
            f"Output shape mismatch: expected {expected.shape}, found {observed.shape}."
        )
    if not np.isfinite(expected).all() or not np.isfinite(observed).all():
        raise ValueError("Parity outputs must contain only finite values.")
    absolute = np.abs(expected.astype(np.float64) - observed.astype(np.float64))
    scale = np.maximum(np.abs(expected.astype(np.float64)), np.finfo(float).tiny)
    relative = absolute / scale
    return OnnxParityReport(
        passed=bool(np.allclose(expected, observed, atol=atol, rtol=rtol)),
        max_absolute_error=float(np.max(absolute)) if absolute.size else 0.0,
        mean_absolute_error=float(np.mean(absolute)) if absolute.size else 0.0,
        max_relative_error=float(np.max(relative)) if relative.size else 0.0,
        absolute_tolerance=float(atol),
        relative_tolerance=float(rtol),
        shape=expected.shape,
    )


class OnnxRunner:
    """A lazily initialized ONNX Runtime session."""

    def __init__(
        self,
        path: str | Path,
        *,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.providers = tuple(providers) if providers is not None else None
        self._session: Any | None = None

    @property
    def session(self) -> Any:
        if self._session is None:
            runtime = require_optional(
                "onnxruntime", extra="flows", purpose="ONNX inference"
            )
            options: dict[str, Any] = {}
            if self.providers is not None:
                options["providers"] = list(self.providers)
            self._session = runtime.InferenceSession(str(self.path), **options)
        return self._session

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(value.name for value in self.session.get_inputs())

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(value.name for value in self.session.get_outputs())

    def run(
        self,
        inputs: Mapping[str, ArrayLike] | ArrayLike,
        *,
        output_names: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Run inference and return outputs keyed by graph output name."""

        if isinstance(inputs, Mapping):
            feed = {
                name: np.asarray(value, dtype=np.float32)
                for name, value in inputs.items()
            }
        else:
            names = self.input_names
            if len(names) != 1:
                raise ValueError(
                    "A mapping is required for an ONNX graph with multiple inputs."
                )
            feed = {names[0]: np.asarray(inputs, dtype=np.float32)}
        unknown = set(feed).difference(self.input_names)
        missing = set(self.input_names).difference(feed)
        if unknown or missing:
            raise ValueError(
                f"ONNX input mismatch; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}."
            )
        requested = list(output_names) if output_names is not None else None
        values = self.session.run(requested, feed)
        names = tuple(output_names) if output_names is not None else self.output_names
        return {
            name: np.asarray(value) for name, value in zip(names, values, strict=True)
        }


def check_onnx_parity(
    model_path: str | Path,
    reference: Callable[[Mapping[str, np.ndarray]], ArrayLike],
    inputs: Mapping[str, ArrayLike],
    *,
    output_name: str | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    providers: Sequence[str] | None = None,
) -> OnnxParityReport:
    """Compare a callable reference with one output from an ONNX graph."""

    feed = {name: np.asarray(value, dtype=np.float32) for name, value in inputs.items()}
    runner = OnnxRunner(model_path, providers=providers)
    outputs = runner.run(feed)
    selected = output_name or next(iter(outputs))
    if selected not in outputs:
        raise ValueError(
            f"Unknown output {selected!r}; available outputs are {sorted(outputs)}."
        )
    return compare_outputs(reference(feed), outputs[selected], atol=atol, rtol=rtol)


def export_torch_onnx(
    module: Any,
    example_inputs: Any | tuple[Any, ...],
    path: str | Path,
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    artifact_type: str,
    metadata: dict[str, Any] | None = None,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> tuple[Path, Path]:
    """Export a PyTorch module and write its checksum manifest.

    The caller owns semantic parity testing because only it knows the native
    model API.  :func:`check_onnx_parity` provides the shared comparison hook.
    """

    torch = require_optional("torch", extra="flows", purpose="PyTorch ONNX export")
    require_optional("onnx", extra="flows", purpose="PyTorch ONNX export")
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    inputs = example_inputs if isinstance(example_inputs, tuple) else (example_inputs,)
    if len(inputs) != len(input_names):
        raise ValueError("input_names must contain one name per example input.")
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            name: {0: "batch"} for name in (*tuple(input_names), *tuple(output_names))
        }
    export_options = {
        "export_params": True,
        "do_constant_folding": True,
        "input_names": list(input_names),
        "output_names": list(output_names),
        "dynamic_axes": dynamic_axes,
        "opset_version": opset_version,
    }
    # PyTorch 2.9 switched ``torch.onnx.export`` to its dynamo exporter by
    # default. That exporter may silently raise the requested opset and treats
    # ``dynamic_axes`` as a compatibility hint. The established tracer emits
    # the requested opset exactly and preserves the symbolic batch axes used
    # by the portable inference contract. PyTorch 2.1 predates this keyword,
    # hence the signature guard.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    torch.onnx.export(module, inputs, str(model_path), **export_options)
    manifest_path = manifest_path_for(model_path)
    write_artifact_manifest(
        manifest_path,
        artifact_type=artifact_type,
        files={"onnx-model": model_path},
        metadata={
            "input_names": list(input_names),
            "opset_version": opset_version,
            "output_names": list(output_names),
            **dict(metadata or {}),
        },
    )
    return model_path, manifest_path


def export_sklearn_transformer_to_onnx(
    transformer: Any,
    path: str | Path,
    *,
    n_features: int,
    feature_names: Sequence[str] | None = None,
    input_name: str = "features",
    output_name: str = "transformed",
    metadata: dict[str, Any] | None = None,
    target_opset: int = 17,
) -> tuple[Path, Path]:
    """Export a fitted scikit-learn transformer and checksum its graph."""

    if n_features < 1:
        raise ValueError("n_features must be positive.")
    names = (
        tuple(feature_names)
        if feature_names is not None
        else tuple(getattr(transformer, "feature_names_in_", ()))
    )
    if names and (len(names) != n_features or len(set(names)) != len(names)):
        raise ValueError(
            "feature_names must be unique and contain exactly n_features names."
        )
    portable_transformer = _integerize_column_transformer(
        transformer, feature_names=names
    )
    skl2onnx = require_optional(
        "skl2onnx", extra="lhc", purpose="scikit-learn scaler ONNX export"
    )
    data_types = require_optional(
        "skl2onnx.common.data_types",
        extra="lhc",
        purpose="scikit-learn scaler ONNX export",
    )
    model = skl2onnx.convert_sklearn(
        portable_transformer,
        initial_types=[(input_name, data_types.FloatTensorType([None, n_features]))],
        target_opset=target_opset,
        final_types=[(output_name, data_types.FloatTensorType([None, n_features]))],
    )
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(model.SerializeToString())
    manifest_path = manifest_path_for(model_path)
    write_artifact_manifest(
        manifest_path,
        artifact_type="sklearn-transformer-onnx",
        files={"onnx-scaler": model_path},
        metadata={
            "input_name": input_name,
            "feature_names": list(names),
            "n_features": n_features,
            "output_name": output_name,
            "target_opset": target_opset,
            **dict(metadata or {}),
        },
    )
    return model_path, manifest_path


def convert_joblib_scaler_to_onnx(
    joblib_path: str | Path,
    onnx_path: str | Path,
    *,
    n_features: int | None = None,
    feature_names: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
    allow_unsafe_pickle: bool = False,
) -> tuple[Path, Path]:
    """Load a trusted legacy joblib scaler and export it to ONNX.

    Joblib artifacts are pickle programs and may execute arbitrary code while
    loading. Conversion is therefore denied by default. Set
    ``allow_unsafe_pickle=True`` only after establishing the provenance and
    integrity of ``joblib_path``.
    """

    if allow_unsafe_pickle is not True:
        raise ValueError(
            "Refusing to load a pickle-backed joblib scaler. Pass "
            "allow_unsafe_pickle=True only for a trusted artifact."
        )
    joblib = require_optional("joblib", extra="lhc", purpose="reading a legacy scaler")
    scaler = joblib.load(joblib_path)
    inferred = getattr(scaler, "n_features_in_", None)
    count = n_features if n_features is not None else inferred
    if count is None:
        mean = getattr(scaler, "mean_", None)
        count = len(mean) if mean is not None else None
    if count is None:
        raise ValueError(
            "Could not infer the scaler feature count; pass n_features explicitly."
        )
    return export_sklearn_transformer_to_onnx(
        scaler,
        onnx_path,
        n_features=int(count),
        feature_names=feature_names,
        metadata={
            "source_joblib": Path(joblib_path).name,
            **dict(metadata or {}),
        },
    )


def _integerize_column_transformer(
    transformer: Any,
    *,
    feature_names: Sequence[str],
) -> Any:
    """Adapt fitted named ColumnTransformer selectors to one tensor input."""

    if transformer.__class__.__name__ != "ColumnTransformer":
        return transformer
    names = tuple(feature_names)
    lookup = {name: index for index, name in enumerate(names)}

    def convert(columns: Any) -> Any:
        if isinstance(columns, str):
            if columns not in lookup:
                raise ValueError(f"Scaler selects unknown feature {columns!r}.")
            return lookup[columns]
        if isinstance(columns, slice) or callable(columns):
            return columns
        try:
            values = list(columns)
        except TypeError:
            return columns
        string_flags = [isinstance(value, str) for value in values]
        if not any(string_flags):
            return columns
        if not all(string_flags):
            raise ValueError(
                "Mixed named and positional ColumnTransformer selectors "
                "cannot be exported as one ONNX tensor."
            )
        missing = set(values).difference(lookup)
        if missing:
            raise ValueError(f"Scaler selects unknown features {sorted(missing)}.")
        return [lookup[value] for value in values]

    portable = copy.deepcopy(transformer)
    for attribute in ("transformers", "transformers_"):
        entries = getattr(portable, attribute, None)
        if entries is not None:
            setattr(
                portable,
                attribute,
                [(name, nested, convert(columns)) for name, nested, columns in entries],
            )
    return portable
