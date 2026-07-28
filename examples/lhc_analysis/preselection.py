"""Deterministic preparation of the selected synthetic LHC samples.

The original ML4HEP exercises used a learned signal/background density ratio
to define a fixed preselection region.  The historical network checkpoint was
not published, but this controlled Gaussian-mixture example has the exact
reconstructed signal and background densities.  We therefore use their
analytical ratio as a deterministic surrogate while preserving the statistical
contract of the exercise:

* the same ``p_signal / p_background`` observable and ``B/S <= 250`` target;
* the same SplitMix64 50% / 46% / 4% row partition;
* one cut derived from nominal samples and applied unchanged to every
  systematic variation; and
* a post-selection reference with equal conditional signal/background
  components.

The module is example-local because arbitrary user samples do not have
analytical densities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_UINT64_MASK = (1 << 64) - 1
MANIFEST_NAME = "preselection.manifest.json"
SAMPLE_NAMES = (
    "background",
    "background_resolution_down",
    "background_resolution_up",
    "background_response_down",
    "background_response_up",
    "reference",
    "signal",
    "signal_resolution_down",
    "signal_resolution_up",
    "signal_response_down",
    "signal_response_up",
    "signal_theory_down",
    "signal_theory_up",
)
_EXPECTED_SAMPLE_NAMES = frozenset(SAMPLE_NAMES)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric constant {value!r}.")


@dataclass(frozen=True)
class PreselectionConfig:
    """Legacy-compatible preselection and row-partition settings."""

    target_background_to_signal: float = 250.0
    selector_training_fraction: float = 0.50
    downstream_training_fraction: float = 0.92
    split_seed: int = 0
    histogram_bins: int = 4_000
    log_ratio_range: tuple[float, float] = (-20.0, 20.0)
    reference_batch_size: int = 100_000

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.target_background_to_signal)
            or self.target_background_to_signal <= 0
        ):
            raise ValueError("target_background_to_signal must be positive.")
        if not 0.0 < self.selector_training_fraction < 1.0:
            raise ValueError("selector_training_fraction must lie in (0, 1).")
        if not 0.0 < self.downstream_training_fraction < 1.0:
            raise ValueError("downstream_training_fraction must lie in (0, 1).")
        if int(self.histogram_bins) < 2:
            raise ValueError("histogram_bins must be at least two.")
        low, high = map(float, self.log_ratio_range)
        if not math.isfinite(low) or not math.isfinite(high) or not low < high:
            raise ValueError("log_ratio_range must contain two increasing values.")
        if int(self.reference_batch_size) < 1:
            raise ValueError("reference_batch_size must be positive.")


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 hash used by the historical exercise."""

    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return values ^ (values >> np.uint64(31))


def _hashed_uniform(row_indices: np.ndarray, seed: int) -> np.ndarray:
    seed_value = np.uint64(int(seed) & _UINT64_MASK)
    with np.errstate(over="ignore"):
        values = np.asarray(row_indices, dtype=np.uint64) + seed_value
    hashed = _splitmix64(values)
    return (hashed >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)


def legacy_partition_labels(
    rows: int | np.ndarray,
    *,
    config: PreselectionConfig,
) -> np.ndarray:
    """Return ``preselection``, ``flow_train``, or ``evaluation`` per row."""

    indices = (
        np.arange(int(rows), dtype=np.uint64)
        if np.isscalar(rows)
        else np.asarray(rows, dtype=np.uint64)
    )
    uniform = _hashed_uniform(indices, config.split_seed)
    flow_boundary = config.selector_training_fraction + (
        (1.0 - config.selector_training_fraction)
        * config.downstream_training_fraction
    )
    return np.where(
        uniform < config.selector_training_fraction,
        "preselection",
        np.where(uniform < flow_boundary, "flow_train", "evaluation"),
    )


def _component_payload(
    components: Sequence[tuple[float, np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    fractions = np.asarray([item[0] for item in components], dtype=np.float64)
    fractions /= np.sum(fractions)
    return [
        {
            "fraction": float(fraction),
            "mean": np.asarray(mean, dtype=np.float64).tolist(),
            "covariance": np.asarray(covariance, dtype=np.float64).tolist(),
        }
        for fraction, (_, mean, covariance) in zip(
            fractions,
            components,
            strict=True,
        )
    ]


class GaussianMixtureRatioSelector:
    """Exact nominal reconstructed ``p_signal(x) / p_background(x)`` selector."""

    def __init__(
        self,
        *,
        features: Sequence[str],
        signal_components: Sequence[tuple[float, np.ndarray, np.ndarray]],
        background_components: Sequence[tuple[float, np.ndarray, np.ndarray]],
    ) -> None:
        self.features = tuple(features)
        if not self.features:
            raise ValueError("features must be non-empty.")
        self._signal = self._prepare(signal_components)
        self._background = self._prepare(background_components)
        payload = {
            "algorithm": "analytical-gaussian-mixture-log-ratio",
            "features": list(self.features),
            "signal": _component_payload(signal_components),
            "background": _component_payload(background_components),
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        self.specification = payload
        self.fingerprint = hashlib.sha256(encoded).hexdigest()

    def _prepare(
        self,
        components: Sequence[tuple[float, np.ndarray, np.ndarray]],
    ) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
        if not components:
            raise ValueError("A selector density needs at least one component.")
        fractions = np.asarray([item[0] for item in components], dtype=np.float64)
        if (
            not np.isfinite(fractions).all()
            or np.any(fractions <= 0)
            or not float(np.sum(fractions)) > 0
        ):
            raise ValueError("Mixture fractions must be finite and positive.")
        fractions /= np.sum(fractions)
        prepared = []
        dimension = len(self.features)
        for fraction, (_, raw_mean, raw_covariance) in zip(
            fractions,
            components,
            strict=True,
        ):
            mean = np.asarray(raw_mean, dtype=np.float64)
            covariance = np.asarray(raw_covariance, dtype=np.float64)
            if mean.shape != (dimension,) or covariance.shape != (
                dimension,
                dimension,
            ):
                raise ValueError("Selector component dimensions do not match features.")
            sign, log_determinant = np.linalg.slogdet(covariance)
            if sign <= 0 or not math.isfinite(float(log_determinant)):
                raise ValueError("Selector covariance must be positive definite.")
            inverse = np.linalg.inv(covariance)
            log_normalization = (
                math.log(float(fraction))
                - 0.5 * dimension * math.log(2.0 * math.pi)
                - 0.5 * float(log_determinant)
            )
            prepared.append((log_normalization, mean, inverse))
        return tuple(prepared)

    @staticmethod
    def _log_density(
        values: np.ndarray,
        components: tuple[tuple[float, np.ndarray, np.ndarray], ...],
    ) -> np.ndarray:
        terms = []
        for log_normalization, mean, inverse in components:
            delta = values - mean
            exponent = np.einsum(
                "ni,ij,nj->n",
                delta,
                inverse,
                delta,
                optimize=True,
            )
            terms.append(log_normalization - 0.5 * exponent)
        stacked = np.stack(terms, axis=0)
        maximum = np.max(stacked, axis=0)
        return maximum + np.log(np.sum(np.exp(stacked - maximum), axis=0))

    def log_ratio(self, values: Any) -> np.ndarray:
        if isinstance(values, pd.DataFrame):
            array = values.loc[:, list(self.features)].to_numpy(dtype=np.float64)
        else:
            array = np.asarray(values, dtype=np.float64)
        if (
            array.ndim != 2
            or array.shape[1] != len(self.features)
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                f"Selector values must have shape (n, {len(self.features)}) "
                "and be finite."
            )
        return self._log_density(array, self._signal) - self._log_density(
            array,
            self._background,
        )

    def __call__(self, values: Any) -> np.ndarray:
        return np.exp(np.clip(self.log_ratio(values), -80.0, 80.0))


def reconstructed_components(
    components: Sequence[tuple[float, np.ndarray, np.ndarray]],
    *,
    scale: np.ndarray,
    resolution: np.ndarray,
) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    """Map the latent Gaussian mixture to the nominal reconstructed density."""

    scale_array = np.asarray(scale, dtype=np.float64)
    resolution_array = np.asarray(resolution, dtype=np.float64)
    if (
        scale_array.ndim != 1
        or resolution_array.shape != scale_array.shape
        or not np.isfinite(scale_array).all()
        or not np.isfinite(resolution_array).all()
        or np.any(resolution_array <= 0)
    ):
        raise ValueError("scale and resolution must be aligned finite vectors.")
    transform = np.diag(scale_array)
    noise = np.diag(resolution_array**2)
    return tuple(
        (
            float(fraction),
            scale_array * np.asarray(mean, dtype=np.float64),
            transform
            @ np.asarray(covariance, dtype=np.float64)
            @ transform.T
            + noise,
        )
        for fraction, mean, covariance in components
    )


def _validated_weights(frame: pd.DataFrame) -> np.ndarray:
    if "weight" not in frame:
        raise ValueError("Preselection inputs require a weight column.")
    weights = frame["weight"].to_numpy(dtype=np.float64)
    if (
        not len(weights)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not float(np.sum(weights)) > 0
    ):
        raise ValueError("Preselection weights must be finite and non-negative.")
    return weights


def derive_ratio_cut(
    signal: pd.DataFrame,
    background: pd.DataFrame,
    *,
    selector: GaussianMixtureRatioSelector,
    config: PreselectionConfig,
) -> tuple[float, dict[str, float]]:
    """Choose the loosest legacy histogram cut satisfying the target ``B/S``."""

    edges = np.linspace(
        float(config.log_ratio_range[0]),
        float(config.log_ratio_range[1]),
        int(config.histogram_bins) + 1,
    )
    histograms: dict[str, np.ndarray] = {}
    inclusive_yields: dict[str, float] = {}
    partition_weights: dict[str, float] = {}
    for name, frame in (("signal", signal), ("background", background)):
        weights = _validated_weights(frame)
        labels = legacy_partition_labels(len(frame), config=config)
        selected = labels == "flow_train"
        if not np.any(selected):
            raise ValueError(f"No {name} rows are available to derive the cut.")
        log_ratio = np.clip(
            selector.log_ratio(frame.loc[selected, list(selector.features)]),
            edges[0],
            edges[-1],
        )
        histograms[name] = np.histogram(
            log_ratio,
            bins=edges,
            weights=weights[selected],
        )[0]
        inclusive_yields[name] = float(np.sum(weights))
        partition_weights[name] = float(np.sum(weights[selected]))

    signal_yield = (
        inclusive_yields["signal"]
        * np.cumsum(histograms["signal"][::-1])
        / partition_weights["signal"]
    )
    background_yield = (
        inclusive_yields["background"]
        * np.cumsum(histograms["background"][::-1])
        / partition_weights["background"]
    )
    background_to_signal = np.divide(
        background_yield,
        signal_yield,
        out=np.full_like(background_yield, np.inf),
        where=signal_yield > 0,
    )
    valid = np.flatnonzero(
        (signal_yield > 0)
        & (background_to_signal <= config.target_background_to_signal)
    )
    if not len(valid):
        raise RuntimeError(
            "The nominal selector cannot reach the requested B/S target. "
            "Increase the generated statistics or relax the target."
        )
    best = valid[np.argmax(signal_yield[valid])]
    lower_edges_descending = edges[:-1][::-1]
    log_ratio_cut = float(lower_edges_descending[best])
    ratio_cut = float(math.exp(log_ratio_cut))
    return ratio_cut, {
        "background_to_signal": float(background_to_signal[best]),
        "background_yield": float(background_yield[best]),
        "log_ratio_cut": log_ratio_cut,
        "signal_yield": float(signal_yield[best]),
    }


def select_process_frame(
    frame: pd.DataFrame,
    *,
    selector: GaussianMixtureRatioSelector,
    ratio_cut: float,
    config: PreselectionConfig,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Apply one fixed cut and retain only the downstream 46%/4% partitions."""

    if not math.isfinite(float(ratio_cut)) or float(ratio_cut) <= 0:
        raise ValueError("ratio_cut must be finite and positive.")
    weights = _validated_weights(frame)
    labels = legacy_partition_labels(len(frame), config=config)
    downstream = labels != "preselection"
    if not np.any(downstream):
        raise ValueError("No downstream rows are available after partitioning.")
    positions = np.flatnonzero(downstream)
    passes = np.zeros(len(frame), dtype=bool)
    passes[positions] = (
        selector(frame.iloc[positions].loc[:, list(selector.features)])
        >= float(ratio_cut)
    )
    retained = downstream & passes
    if not np.any(retained):
        raise RuntimeError("Preselection retained no events.")

    selected = frame.loc[retained].copy()
    selected_labels = labels[retained]
    selected["preselection_partition"] = selected_labels
    # ``evaluation`` is the independent 4% legacy partition.  The 46%
    # flow-training pool uses the generator's original 80/10/10 labels to
    # provide explicit train/validation partitions to the native toolkit.
    original_split = selected["split"].astype(str).to_numpy()
    selected["preselection_split"] = np.where(
        selected_labels == "evaluation",
        "holdout",
        np.where(original_split == "train", "train", "validation"),
    )

    inclusive_weight = float(np.sum(weights))
    flow_training = labels == "flow_train"
    flow_training_weight = float(np.sum(weights[flow_training]))
    selected_flow_training_weight = float(
        np.sum(weights[flow_training & passes])
    )
    if flow_training_weight <= 0 or selected_flow_training_weight <= 0:
        raise RuntimeError(
            "The legacy flow-training partition has no positive selected weight."
        )
    selected_weight_before_rescale = float(np.sum(weights[retained]))
    efficiency = selected_flow_training_weight / flow_training_weight
    selected_yield_target = inclusive_weight * efficiency
    selected["weight"] = (
        selected["weight"].to_numpy(dtype=np.float64)
        * selected_yield_target
        / selected_weight_before_rescale
    )
    selected_yield = float(selected["weight"].sum())
    selected.reset_index(drop=True, inplace=True)
    return selected, {
        "analysis_partition_events": int(np.sum(downstream)),
        "analysis_partition_weight": float(np.sum(weights[downstream])),
        "efficiency": float(efficiency),
        "flow_training_events": int(np.sum(flow_training)),
        "flow_training_selected_events": int(np.sum(flow_training & passes)),
        "flow_training_selected_weight": selected_flow_training_weight,
        "flow_training_weight": flow_training_weight,
        "inclusive_events": int(len(frame)),
        "inclusive_weight": inclusive_weight,
        "selected_events": int(np.sum(retained)),
        "selected_weight": selected_yield,
        "selector_training_events": int(np.sum(labels == "preselection")),
    }


def select_reference_frame(
    frame: pd.DataFrame,
    *,
    selector: GaussianMixtureRatioSelector,
    ratio_cut: float,
) -> pd.DataFrame:
    """Apply the fixed selector to an independent reference candidate batch."""

    if not math.isfinite(float(ratio_cut)) or float(ratio_cut) <= 0:
        raise ValueError("ratio_cut must be finite and positive.")
    passes = selector(frame.loc[:, list(selector.features)]) >= float(ratio_cut)
    return frame.loc[passes].copy()


def write_parquet_atomic(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write one Parquet through a same-directory temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_preselection_manifest(
    output_directory: str | Path,
    *,
    raw_paths: Mapping[str, Path],
    selected_paths: Mapping[str, Path],
    selector: GaussianMixtureRatioSelector,
    ratio_cut: float,
    cut_diagnostics: Mapping[str, float],
    config: PreselectionConfig,
    sample_diagnostics: Mapping[str, Mapping[str, float | int]],
    reference_diagnostics: Mapping[str, Any],
    requested_events: Mapping[str, int],
    generation_seed: int,
) -> Path:
    """Checksum the complete generation/preselection contract."""

    root = Path(output_directory)
    target = root / MANIFEST_NAME
    if (
        set(raw_paths) != _EXPECTED_SAMPLE_NAMES
        or set(selected_paths) != _EXPECTED_SAMPLE_NAMES
    ):
        raise ValueError("The preselection manifest requires the exact sample set.")
    for name in SAMPLE_NAMES:
        raw_path = Path(raw_paths[name])
        selected_path = Path(selected_paths[name])
        if raw_path != root / f"{name}.parquet":
            raise ValueError(f"Raw sample {name!r} does not use its canonical path.")
        if selected_path != root / f"{name}_presel.parquet":
            raise ValueError(
                f"Selected sample {name!r} does not use its canonical path."
            )
        if raw_path.is_symlink() or selected_path.is_symlink():
            raise ValueError("Preselection manifests do not accept symbolic links.")
    payload = {
        "schema_version": "1",
        "config": asdict(config),
        "cut": {
            "ratio": float(ratio_cut),
            **{name: float(value) for name, value in cut_diagnostics.items()},
        },
        "files": {
            "raw": {
                name: _file_record(path, root=root)
                for name, path in sorted(raw_paths.items())
            },
            "selected": {
                name: _file_record(path, root=root)
                for name, path in sorted(selected_paths.items())
            },
        },
        "generation": {"seed": int(generation_seed)},
        "reference": dict(reference_diagnostics),
        "requested_events": {
            name: int(value) for name, value in requested_events.items()
        },
        "samples": {
            name: dict(values) for name, values in sorted(sample_diagnostics.items())
        },
        "selector": {
            "fingerprint": selector.fingerprint,
            "specification": selector.specification,
        },
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_preselection_manifest(
    output_directory: str | Path,
) -> dict[str, Any]:
    """Load and minimally validate the generated preselection manifest."""

    path = Path(output_directory) / MANIFEST_NAME
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise FileNotFoundError(path) from exc
    except ValueError as exc:
        raise ValueError(f"{path} is not valid JSON.") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        raise ValueError(f"{path} has an unsupported schema.")
    return payload


def verify_preselection_manifest(
    output_directory: str | Path,
    *,
    requested_events: Mapping[str, int] | None = None,
    generation_seed: int | None = None,
    config: PreselectionConfig | None = None,
    selector_fingerprint: str | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Verify requested counts plus every raw and selected file checksum."""

    root = Path(output_directory)
    try:
        payload = load_preselection_manifest(root)
    except (FileNotFoundError, ValueError) as exc:
        return False, (str(exc),)
    reasons: list[str] = []
    if requested_events is not None:
        expected = {name: int(value) for name, value in requested_events.items()}
        if payload.get("requested_events") != expected:
            reasons.append("requested generation counts changed")
    if generation_seed is not None and payload.get("generation") != {
        "seed": int(generation_seed)
    }:
        reasons.append("generation seed changed")
    if config is not None:
        expected_config = json.loads(json.dumps(asdict(config), allow_nan=False))
        if payload.get("config") != expected_config:
            reasons.append("preselection configuration changed")
    if selector_fingerprint is not None:
        selector = payload.get("selector")
        recorded_fingerprint = (
            selector.get("fingerprint") if isinstance(selector, Mapping) else None
        )
        if recorded_fingerprint != str(selector_fingerprint):
            reasons.append("preselection selector changed")
    files = payload.get("files")
    if not isinstance(files, Mapping):
        return False, ("preselection file inventory is missing",)
    raw_inventory = files.get("raw")
    selected_inventory = files.get("selected")
    if (
        not isinstance(raw_inventory, Mapping)
        or not isinstance(selected_inventory, Mapping)
        or set(raw_inventory) != _EXPECTED_SAMPLE_NAMES
        or set(selected_inventory) != _EXPECTED_SAMPLE_NAMES
    ):
        reasons.append("preselection file inventory is incomplete")
    resolved_root = root.resolve()
    for section in ("raw", "selected"):
        records = files.get(section)
        if not isinstance(records, Mapping) or not records:
            reasons.append(f"{section} preselection inventory is missing")
            continue
        for name, raw_record in sorted(records.items()):
            if not isinstance(raw_record, Mapping):
                reasons.append(f"{section} file {name!r} has invalid metadata")
                continue
            path_value = raw_record.get("path")
            if not isinstance(path_value, str):
                reasons.append(f"{section} file {name!r} has no path")
                continue
            expected_path = (
                f"{name}.parquet"
                if section == "raw"
                else f"{name}_presel.parquet"
            )
            if path_value != expected_path:
                reasons.append(
                    f"{section} file {name!r} does not use its canonical path"
                )
                continue
            relative = Path(path_value)
            if relative.is_absolute() or ".." in relative.parts:
                reasons.append(f"{section} file {name!r} has an unsafe path")
                continue
            path = root / relative
            if path.is_symlink():
                reasons.append(f"{section} file {name!r} is a symbolic link")
                continue
            if not path.is_file():
                reasons.append(f"{section} file {name!r} is missing")
                continue
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                reasons.append(f"{section} file {name!r} cannot be resolved")
                continue
            if not resolved.is_relative_to(resolved_root):
                reasons.append(f"{section} file {name!r} escapes the bundle root")
                continue
            if (
                raw_record.get("size_bytes") != path.stat().st_size
                or raw_record.get("sha256") != _sha256_file(path)
            ):
                reasons.append(f"{section} file {name!r} changed")
    return not reasons, tuple(reasons)
