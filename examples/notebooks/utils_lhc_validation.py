"""Analytical validation helpers for the synthetic LHC notebooks.

These functions deliberately live with the examples rather than in
``hnsbi``: arbitrary user samples do not have an analytical density.  The
helpers accept the exact mixture components and detector-response arrays from
``examples/lhc_analysis/generate_distributions.py`` so the validation cannot
silently drift away from the generator.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _provenance_record(path: str | Path) -> dict[str, int | str]:
    from hnsbi.artifacts import sha256_file

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _named_paths(
    paths: Mapping[str, str | Path],
    *,
    name: str,
) -> dict[str, str | Path]:
    normalized = {str(key): value for key, value in paths.items()}
    if len(normalized) != len(paths):
        raise ValueError(f"{name} contains duplicate names after string conversion.")
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def write_reuse_provenance(
    path: str | Path,
    *,
    configuration_path: str | Path,
    data_paths: Mapping[str, str | Path],
    artifact_paths: Mapping[str, str | Path],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Bind reusable notebook artifacts to their configuration and datasets."""

    normalized_data = _named_paths(data_paths, name="data_paths")
    normalized_artifacts = _named_paths(artifact_paths, name="artifact_paths")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "configuration": _provenance_record(configuration_path),
        "data": {
            name: _provenance_record(value)
            for name, value in sorted(normalized_data.items())
        },
        "artifacts": {
            name: _provenance_record(value)
            for name, value in sorted(normalized_artifacts.items())
        },
        "metadata": dict(metadata or {}),
    }
    target.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def verify_reuse_provenance(
    path: str | Path,
    *,
    configuration_path: str | Path,
    data_paths: Mapping[str, str | Path],
    artifact_paths: Mapping[str, str | Path],
) -> tuple[bool, tuple[str, ...]]:
    """Verify the exact identity recorded by :func:`write_reuse_provenance`."""

    normalized_data = _named_paths(data_paths, name="data_paths")
    normalized_artifacts = _named_paths(artifact_paths, name="artifact_paths")
    source = Path(path)
    if not source.is_file():
        return False, ("reuse provenance is missing",)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ("reuse provenance is not valid JSON",)
    if not isinstance(payload, dict) or payload.get("schema_version") != "1":
        return False, ("reuse provenance has an unsupported schema",)
    reasons = []
    try:
        current_configuration = _provenance_record(configuration_path)
    except FileNotFoundError:
        current_configuration = None
    if payload.get("configuration") != current_configuration:
        reasons.append("analysis configuration changed")
    for section, paths in (
        ("data", normalized_data),
        ("artifacts", normalized_artifacts),
    ):
        recorded = payload.get(section)
        if not isinstance(recorded, dict) or set(recorded) != set(paths):
            reasons.append(f"{section} inventory changed")
            continue
        for name, value in sorted(paths.items()):
            try:
                current = _provenance_record(value)
            except FileNotFoundError:
                reasons.append(f"{section} file {name!r} is missing")
                continue
            if recorded[name] != current:
                reasons.append(f"{section} file {name!r} changed")
    return not reasons, tuple(reasons)


def reconstructed_components(
    components: Sequence[tuple[float, np.ndarray, np.ndarray]],
    *,
    scale: np.ndarray,
    resolution: np.ndarray,
    response_scale: float = 1.0,
    resolution_scale: float = 1.0,
) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    """Map latent Gaussian-mixture components to reconstructed feature space."""

    scale_array = float(response_scale) * np.asarray(scale, dtype=np.float64)
    resolution_array = float(resolution_scale) * np.asarray(
        resolution, dtype=np.float64
    )
    if (
        scale_array.ndim != 1
        or resolution_array.shape != scale_array.shape
        or not np.isfinite(scale_array).all()
        or not np.isfinite(resolution_array).all()
        or np.any(resolution_array <= 0)
    ):
        raise ValueError("scale and resolution must be aligned finite vectors.")
    transform = np.diag(scale_array)
    response_covariance = np.diag(resolution_array**2)
    result = []
    for fraction, latent_mean, latent_covariance in components:
        mean = np.asarray(latent_mean, dtype=np.float64)
        covariance = np.asarray(latent_covariance, dtype=np.float64)
        if mean.shape != scale_array.shape or covariance.shape != (
            len(scale_array),
            len(scale_array),
        ):
            raise ValueError("A mixture component has incompatible dimensions.")
        result.append(
            (
                float(fraction),
                scale_array * mean,
                transform @ covariance @ transform.T + response_covariance,
            )
        )
    return tuple(result)


def mixture_log_density(
    values: np.ndarray,
    components: Sequence[tuple[float, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Evaluate a normalized Gaussian-mixture log density."""

    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite matrix.")
    fractions = np.asarray([component[0] for component in components], dtype=np.float64)
    if (
        not len(fractions)
        or not np.isfinite(fractions).all()
        or np.any(fractions <= 0)
        or not float(np.sum(fractions)) > 0
    ):
        raise ValueError("Mixture fractions must be finite and positive.")
    fractions /= np.sum(fractions)
    terms = []
    for fraction, (_, mean, covariance) in zip(fractions, components, strict=True):
        terms.append(
            np.log(fraction)
            + multivariate_normal(
                mean=np.asarray(mean, dtype=np.float64),
                cov=np.asarray(covariance, dtype=np.float64),
                allow_singular=False,
            ).logpdf(array)
        )
    return np.asarray(logsumexp(np.vstack(terms), axis=0), dtype=np.float64)


def reference_log_density(
    component_log_densities: Mapping[str, np.ndarray],
    *,
    fractions: Mapping[str, float],
) -> np.ndarray:
    """Combine normalized component log densities into a reference mixture."""

    from scipy.special import logsumexp

    if set(component_log_densities) != set(fractions):
        raise ValueError("Reference fractions must match the supplied components.")
    names = tuple(component_log_densities)
    arrays = {
        name: np.asarray(component_log_densities[name], dtype=np.float64).reshape(-1)
        for name in names
    }
    lengths = {len(values) for values in arrays.values()}
    weights = np.asarray([fractions[name] for name in names], dtype=np.float64)
    if (
        len(lengths) != 1
        or not next(iter(lengths))
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("Reference inputs must be aligned with positive fractions.")
    weights /= np.sum(weights)
    terms = np.vstack(
        [
            np.log(weight) + arrays[name]
            for name, weight in zip(names, weights, strict=True)
        ]
    )
    return np.asarray(logsumexp(terms, axis=0), dtype=np.float64)


def binned_log_density_calibration(
    truth_log_density: np.ndarray,
    model_log_density: np.ndarray,
    *,
    bins: int = 25,
) -> tuple[Any, np.ndarray]:
    """Return equal-population truth bins and model-minus-truth residuals."""

    import pandas as pd

    truth = np.asarray(truth_log_density, dtype=np.float64).reshape(-1)
    model = np.asarray(model_log_density, dtype=np.float64).reshape(-1)
    if (
        len(truth) != len(model)
        or not len(truth)
        or not np.isfinite(truth).all()
        or not np.isfinite(model).all()
    ):
        raise ValueError("Truth and model log densities must be aligned and finite.")
    if int(bins) < 2:
        raise ValueError("bins must be at least two.")
    edges = np.unique(np.quantile(truth, np.linspace(0.0, 1.0, int(bins) + 1)))
    if len(edges) < 2:
        raise ValueError(
            "Truth log density is constant; calibration bins are undefined."
        )
    epsilon = 1.0e-9 * max(1.0, float(edges[-1] - edges[0]))
    edges = edges.copy()
    edges[0] -= epsilon
    edges[-1] += epsilon
    index = np.clip(
        np.searchsorted(edges, truth, side="right") - 1,
        0,
        len(edges) - 2,
    )
    residual = model - truth
    rows = []
    for bin_index in range(len(edges) - 1):
        selected = index == bin_index
        count = int(np.sum(selected))
        if count:
            model_std = float(np.std(model[selected], ddof=1)) if count > 1 else 0.0
            residual_std = (
                float(np.std(residual[selected], ddof=1)) if count > 1 else 0.0
            )
            rows.append(
                {
                    "bin": bin_index,
                    "truth_lo": edges[bin_index],
                    "truth_hi": edges[bin_index + 1],
                    "count": count,
                    "truth_mean": float(np.mean(truth[selected])),
                    "flow_mean": float(np.mean(model[selected])),
                    "flow_sem": model_std / np.sqrt(count),
                    "delta_mean": float(np.mean(residual[selected])),
                    "delta_sem": residual_std / np.sqrt(count),
                }
            )
        else:
            rows.append(
                {
                    "bin": bin_index,
                    "truth_lo": edges[bin_index],
                    "truth_hi": edges[bin_index + 1],
                    "count": 0,
                    "truth_mean": np.nan,
                    "flow_mean": np.nan,
                    "flow_sem": np.nan,
                    "delta_mean": np.nan,
                    "delta_sem": np.nan,
                }
            )
    return pd.DataFrame(rows), edges


def extended_mu_scan(
    *,
    signal_log_density: np.ndarray,
    background_log_density: np.ndarray,
    event_weights: np.ndarray,
    signal_yield: float,
    background_yield: float,
    mu_values: np.ndarray,
) -> np.ndarray:
    """Evaluate a nominal one-parameter extended-likelihood scan.

    The returned curve is shifted to its grid minimum.  This is the controlled
    Exercise-5-style comparison; nuisance profiling belongs to the serialized
    workspace and is intentionally handled separately.
    """

    signal = np.asarray(signal_log_density, dtype=np.float64).reshape(-1)
    background = np.asarray(background_log_density, dtype=np.float64).reshape(-1)
    weights = np.asarray(event_weights, dtype=np.float64).reshape(-1)
    grid = np.asarray(mu_values, dtype=np.float64).reshape(-1)
    if (
        len(signal) != len(background)
        or len(signal) != len(weights)
        or not len(signal)
        or not np.isfinite(signal).all()
        or not np.isfinite(background).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("Log densities and non-negative weights must align.")
    if (
        not len(grid)
        or not np.isfinite(grid).all()
        or np.any(grid < 0)
        or float(signal_yield) <= 0
        or float(background_yield) <= 0
    ):
        raise ValueError("Yields must be positive and mu_values finite/non-negative.")
    log_signal_intensity = np.log(float(signal_yield)) + signal
    log_background_intensity = np.log(float(background_yield)) + background
    nll = []
    for mu in grid:
        signal_term = (
            np.full_like(log_signal_intensity, -np.inf)
            if mu == 0
            else np.log(mu) + log_signal_intensity
        )
        log_intensity = np.logaddexp(log_background_intensity, signal_term)
        nll.append(
            2.0 * (float(background_yield) + mu * float(signal_yield))
            - 2.0 * float(np.sum(weights * log_intensity))
        )
    values = np.asarray(nll, dtype=np.float64)
    return np.maximum(values - np.min(values), 0.0)


def asimov_mu_scan(
    result: Any,
    mu_values: np.ndarray,
    *,
    signal_yield: float,
    background_yield: float,
    signal_name: str = "signal",
    background_name: str = "background",
    truth_mu: float = 1.0,
) -> np.ndarray:
    r"""Evaluate a fixed-nuisance $t_A(\mu)$ curve from an ``AsimovResult``."""

    grid = np.asarray(mu_values, dtype=np.float64).reshape(-1)
    if not len(grid) or not np.isfinite(grid).all() or np.any(grid < 0):
        raise ValueError("mu_values must be finite, non-empty, and non-negative.")
    if float(signal_yield) <= 0 or float(background_yield) <= 0 or truth_mu < 0:
        raise ValueError("Yields must be positive and truth_mu non-negative.")
    try:
        signal_ratio = np.asarray(
            result.normalized_ratios[signal_name], dtype=np.float64
        ).reshape(-1)
        background_ratio = np.asarray(
            result.normalized_ratios[background_name], dtype=np.float64
        ).reshape(-1)
        weights = np.asarray(result.events.weights, dtype=np.float64).reshape(-1)
    except (AttributeError, KeyError) as exc:
        raise ValueError("result is missing aligned Asimov ratio arrays.") from exc
    if (
        len(signal_ratio) != len(background_ratio)
        or len(signal_ratio) != len(weights)
        or not len(weights)
        or not np.isfinite(signal_ratio).all()
        or not np.isfinite(background_ratio).all()
        or not np.isfinite(weights).all()
        or np.any(signal_ratio < 0)
        or np.any(background_ratio < 0)
        or np.any(weights < 0)
    ):
        raise ValueError("Asimov ratios and weights must be aligned and non-negative.")
    truth_intensity = (
        float(truth_mu) * float(signal_yield) * signal_ratio
        + float(background_yield) * background_ratio
    )
    if np.any(truth_intensity <= 0):
        raise ValueError("The truth intensity must be positive on every event.")
    curve = []
    for mu in grid:
        intensity = (
            mu * float(signal_yield) * signal_ratio
            + float(background_yield) * background_ratio
        )
        if np.any(intensity <= 0):
            raise ValueError("A scan intensity is non-positive on an Asimov event.")
        curve.append(
            2.0
            * (
                (mu - float(truth_mu)) * float(signal_yield)
                - float(np.sum(weights * np.log(intensity / truth_intensity)))
            )
        )
    values = np.asarray(curve, dtype=np.float64)
    return np.maximum(values, 0.0)


@dataclass(frozen=True)
class CompressedToyModel:
    """Binned likelihood-ratio model for fast nominal toy campaigns."""

    signal_probability: np.ndarray
    background_probability: np.ndarray
    signal_over_background: np.ndarray
    signal_yield: float
    background_yield: float

    @property
    def bins(self) -> int:
        return len(self.signal_over_background)

    def asimov_scan(
        self,
        mu_values: np.ndarray,
        *,
        truth_mu: float = 1.0,
    ) -> np.ndarray:
        """Return the exact binned Asimov statistic for this compression."""

        grid = np.asarray(mu_values, dtype=np.float64).reshape(-1)
        if (
            not len(grid)
            or not np.isfinite(grid).all()
            or np.any(grid < 0)
            or not np.isfinite(truth_mu)
            or truth_mu < 0
        ):
            raise ValueError("mu_values and truth_mu must be finite and non-negative.")
        expected_counts = (
            float(truth_mu) * self.signal_yield * self.signal_probability
            + self.background_yield * self.background_probability
        )
        truth_log_factor = np.log1p(float(truth_mu) * self.signal_over_background)
        result = []
        for mu in grid:
            statistic = 2.0 * (
                (mu - float(truth_mu)) * self.signal_yield
                - np.sum(
                    expected_counts
                    * (np.log1p(mu * self.signal_over_background) - truth_log_factor)
                )
            )
            result.append(max(0.0, float(statistic)))
        return np.asarray(result, dtype=np.float64)


def build_compressed_toy_model(
    signal_ratio: np.ndarray,
    background_ratio: np.ndarray,
    *,
    signal_yield: float,
    background_yield: float,
    reference_weights: np.ndarray | None = None,
    bins: int = 256,
    tail_quantile: float = 1.0e-6,
) -> CompressedToyModel:
    """Compress a nominal hybrid likelihood along its sufficient ratio.

    The supplied process/reference ratios are renormalized on their common
    support before compression. This makes the binned process probabilities
    close exactly and keeps the toy likelihood independent of an external
    Monte-Carlo normalizer.
    """

    signal = np.asarray(signal_ratio, dtype=np.float64).reshape(-1)
    background = np.asarray(background_ratio, dtype=np.float64).reshape(-1)
    if (
        len(signal) != len(background)
        or not len(signal)
        or not np.isfinite(signal).all()
        or not np.isfinite(background).all()
        or np.any(signal < 0)
        or np.any(background < 0)
    ):
        raise ValueError("Process ratios must be aligned, finite, and non-negative.")
    if (
        not np.isfinite(signal_yield)
        or not np.isfinite(background_yield)
        or signal_yield <= 0
        or background_yield <= 0
    ):
        raise ValueError("Process yields must be finite and positive.")
    if int(bins) < 2:
        raise ValueError("bins must be at least two.")
    if not 0 <= float(tail_quantile) < 0.5:
        raise ValueError("tail_quantile must lie in [0, 0.5).")
    if reference_weights is None:
        probability = np.full(len(signal), 1.0 / len(signal), dtype=np.float64)
    else:
        probability = np.asarray(reference_weights, dtype=np.float64).reshape(-1)
        if (
            len(probability) != len(signal)
            or not np.isfinite(probability).all()
            or np.any(probability < 0)
            or not float(np.sum(probability)) > 0
        ):
            raise ValueError(
                "reference_weights must align and define a positive finite measure."
            )
        probability = probability / np.sum(probability)
    signal_mean = float(np.sum(probability * signal))
    background_mean = float(np.sum(probability * background))
    if signal_mean <= 0 or background_mean <= 0:
        raise ValueError("Each process ratio must have positive reference mean.")
    signal = signal / signal_mean
    background = background / background_mean

    tiny = np.finfo(np.float64).tiny
    event_log_ratio = (
        np.log(float(signal_yield) / float(background_yield))
        + np.log(np.maximum(signal, tiny))
        - np.log(np.maximum(background, tiny))
    )
    lower, upper = np.quantile(
        event_log_ratio,
        [float(tail_quantile), 1.0 - float(tail_quantile)],
    )
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("The process likelihood ratio has a degenerate range.")
    interior = np.linspace(lower, upper, int(bins) - 1)
    edges = np.concatenate(([-np.inf], interior, [np.inf]))
    signal_probability = np.histogram(
        event_log_ratio,
        bins=edges,
        weights=probability * signal,
    )[0].astype(np.float64)
    background_probability = np.histogram(
        event_log_ratio,
        bins=edges,
        weights=probability * background,
    )[0].astype(np.float64)
    probability_floor = 1.0e-15
    signal_probability += probability_floor
    background_probability += probability_floor
    signal_probability /= np.sum(signal_probability)
    background_probability /= np.sum(background_probability)
    signal_over_background = (
        float(signal_yield)
        / float(background_yield)
        * signal_probability
        / background_probability
    )
    return CompressedToyModel(
        signal_probability=signal_probability,
        background_probability=background_probability,
        signal_over_background=signal_over_background,
        signal_yield=float(signal_yield),
        background_yield=float(background_yield),
    )


def run_compressed_toys(
    model: CompressedToyModel,
    *,
    hypotheses: Sequence[float] = (0.0, 1.0),
    n_toys: int = 5_000,
    seed: int = 314_159,
    batch_size: int = 1_000,
    newton_steps: int = 16,
    mu_max: float = 12.0,
) -> Any:
    """Generate and fit a fast bounded nominal toy campaign.

    Returns a pandas data frame compatible with the original Exercise-5 toy
    plotting helpers. Counts are drawn exactly in the compressed Poisson model;
    only the one-dimensional process likelihood ratio is binned.
    """

    import pandas as pd

    parsed_hypotheses = tuple(float(value) for value in hypotheses)
    if (
        not parsed_hypotheses
        or not np.isfinite(parsed_hypotheses).all()
        or any(value < 0 for value in parsed_hypotheses)
    ):
        raise ValueError("hypotheses must be finite, non-empty, and non-negative.")
    if int(n_toys) < 1 or int(batch_size) < 1 or int(newton_steps) < 1:
        raise ValueError("n_toys, batch_size, and newton_steps must be positive.")
    if not np.isfinite(mu_max) or mu_max <= max(parsed_hypotheses):
        raise ValueError("mu_max must be finite and exceed every hypothesis.")

    rng = np.random.default_rng(seed)
    ratio = model.signal_over_background
    chunks = []
    for mu_true in parsed_hypotheses:
        signal_means = mu_true * model.signal_yield * model.signal_probability
        background_means = model.background_yield * model.background_probability
        for start in range(0, int(n_toys), int(batch_size)):
            size = min(int(batch_size), int(n_toys) - start)
            signal_counts = rng.poisson(signal_means, size=(size, model.bins))
            background_counts = rng.poisson(
                background_means,
                size=(size, model.bins),
            )
            counts = signal_counts + background_counts
            mu_hat = np.clip(
                (np.sum(counts, axis=1) - model.background_yield) / model.signal_yield,
                0.0,
                float(mu_max),
            )
            score_at_zero = model.signal_yield - np.sum(
                counts * ratio,
                axis=1,
            )
            for _ in range(int(newton_steps)):
                response = ratio / (1.0 + mu_hat[:, None] * ratio)
                score = model.signal_yield - np.sum(counts * response, axis=1)
                information = np.sum(counts * response**2, axis=1)
                step = np.divide(
                    score,
                    information,
                    out=np.zeros_like(score),
                    where=information > 0,
                )
                mu_hat = np.clip(
                    mu_hat - np.clip(step, -2.0, 2.0),
                    0.0,
                    float(mu_max),
                )
            mu_hat = np.where(score_at_zero >= 0.0, 0.0, mu_hat)

            response_at_truth = ratio / (1.0 + mu_true * ratio)
            chunks.append(
                pd.DataFrame(
                    {
                        "mu_true": mu_true,
                        "toy": np.arange(start, start + size),
                        "n_events": np.sum(counts, axis=1),
                        "n_signal": np.sum(signal_counts, axis=1),
                        "n_background": np.sum(background_counts, axis=1),
                        "mu_hat": mu_hat,
                        "t_mu": _compressed_test_statistic(
                            counts,
                            mu_hat,
                            ratio,
                            signal_yield=model.signal_yield,
                            test_mu=mu_true,
                        ),
                        "q_zero": _compressed_test_statistic(
                            counts,
                            mu_hat,
                            ratio,
                            signal_yield=model.signal_yield,
                            test_mu=0.0,
                        ),
                        "information": np.sum(
                            counts * response_at_truth**2,
                            axis=1,
                        ),
                    }
                )
            )
    return pd.concat(chunks, ignore_index=True)


def _compressed_test_statistic(
    counts: np.ndarray,
    mu_hat: np.ndarray,
    signal_over_background: np.ndarray,
    *,
    signal_yield: float,
    test_mu: float,
) -> np.ndarray:
    values = 2.0 * (
        (float(test_mu) - mu_hat) * float(signal_yield)
        - np.sum(
            counts
            * (
                np.log1p(float(test_mu) * signal_over_background)
                - np.log1p(mu_hat[:, None] * signal_over_background)
            ),
            axis=1,
        )
    )
    return np.maximum(values, 0.0)
