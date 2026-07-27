"""Numerical and visual diagnostics for native density-ratio estimators.

The report deliberately keeps plotting optional.  Every check has a numerical
representation that can be serialized and tested in headless environments.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import write_artifact_manifest
from .diagnostics import json_safe


def _vector(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be non-empty and finite.")
    return result


def _weights(values: Any, length: int, *, name: str) -> np.ndarray:
    result = _vector(values, name=name)
    if len(result) != length or np.any(result < 0) or not np.sum(result) > 0:
        raise ValueError(f"{name} must align and define a finite non-negative measure.")
    return result / np.sum(result)


def weighted_ks(
    left: Any,
    right: Any,
    *,
    left_weights: Any,
    right_weights: Any,
) -> float:
    """Return the two-sample weighted Kolmogorov--Smirnov distance."""

    first = _vector(left, name="left")
    second = _vector(right, name="right")
    first_weights = _weights(left_weights, len(first), name="left_weights")
    second_weights = _weights(right_weights, len(second), name="right_weights")
    support = np.sort(np.unique(np.concatenate([first, second])))
    first_order = np.argsort(first)
    second_order = np.argsort(second)
    first_cdf = np.cumsum(first_weights[first_order])
    second_cdf = np.cumsum(second_weights[second_order])
    first_index = np.searchsorted(first[first_order], support, side="right") - 1
    second_index = np.searchsorted(second[second_order], support, side="right") - 1
    first_values = np.where(first_index >= 0, first_cdf[np.maximum(first_index, 0)], 0)
    second_values = np.where(
        second_index >= 0,
        second_cdf[np.maximum(second_index, 0)],
        0,
    )
    return float(np.max(np.abs(first_values - second_values)))


def weighted_auc(
    numerator_scores: Any,
    denominator_scores: Any,
    *,
    numerator_weights: Any,
    denominator_weights: Any,
) -> float:
    """Return the probability that a weighted numerator score is larger."""

    positive = _vector(numerator_scores, name="numerator_scores")
    negative = _vector(denominator_scores, name="denominator_scores")
    positive_weights = _weights(
        numerator_weights, len(positive), name="numerator_weights"
    )
    negative_weights = _weights(
        denominator_weights, len(negative), name="denominator_weights"
    )
    order = np.argsort(negative, kind="stable")
    sorted_negative = negative[order]
    cumulative = np.concatenate([[0.0], np.cumsum(negative_weights[order])])
    lower = np.searchsorted(sorted_negative, positive, side="left")
    upper = np.searchsorted(sorted_negative, positive, side="right")
    probability_lower = cumulative[lower]
    probability_equal = cumulative[upper] - cumulative[lower]
    return float(
        np.sum(positive_weights * (probability_lower + 0.5 * probability_equal))
    )


def _bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if len(edges) < 2:
        center = float(edges[0])
        width = max(1.0, abs(center)) * 1.0e-6
        return np.asarray([center - width, center + width])
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def _normalized_histogram(
    values: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    counts = np.histogram(values, bins=edges, weights=weights)[0].astype(np.float64)
    total = float(np.sum(counts))
    return counts / total if total > 0 else counts


def _score_calibration(
    numerator_score: np.ndarray,
    denominator_score: np.ndarray,
    numerator_weight: np.ndarray,
    denominator_weight: np.ndarray,
    *,
    bins: int,
) -> dict[str, Any]:
    values = np.concatenate([numerator_score, denominator_score])
    weights = np.concatenate([0.5 * numerator_weight, 0.5 * denominator_weight])
    labels = np.concatenate(
        [np.ones(len(numerator_score)), np.zeros(len(denominator_score))]
    )
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    indices = np.clip(
        np.searchsorted(edges, values, side="right") - 1,
        0,
        len(edges) - 2,
    )
    predicted: list[float] = []
    observed: list[float] = []
    masses: list[float] = []
    centers: list[float] = []
    for index in range(len(edges) - 1):
        selected = indices == index
        mass = float(np.sum(weights[selected]))
        if mass <= 0:
            continue
        normalized = weights[selected] / mass
        predicted.append(float(np.sum(normalized * values[selected])))
        observed.append(float(np.sum(normalized * labels[selected])))
        masses.append(mass)
        centers.append(0.5 * (edges[index] + edges[index + 1]))
    predicted_array = np.asarray(predicted)
    observed_array = np.asarray(observed)
    mass_array = np.asarray(masses)
    return {
        "bin_center": centers,
        "mean_prediction": predicted,
        "observed_fraction": observed,
        "mass": masses,
        "observed_over_prediction": np.divide(
            observed_array,
            predicted_array,
            out=np.full_like(observed_array, np.nan),
            where=predicted_array > 0,
        ),
        "expected_calibration_error": (
            float(np.sum(mass_array * np.abs(predicted_array - observed_array)))
            if len(mass_array)
            else math.nan
        ),
        "brier_score": float(
            np.sum(weights * (values - labels) ** 2) / np.sum(weights)
        ),
    }


def _log_ratio_calibration(
    numerator_ratio: np.ndarray,
    denominator_ratio: np.ndarray,
    numerator_weight: np.ndarray,
    denominator_weight: np.ndarray,
    *,
    bins: int,
) -> dict[str, Any]:
    floor = 1.0e-12
    numerator_log = np.log(np.maximum(numerator_ratio, floor))
    denominator_log = np.log(np.maximum(denominator_ratio, floor))
    combined = np.concatenate([numerator_log, denominator_log])
    finite = np.isfinite(combined)
    if not np.any(finite):
        raise ValueError("No finite predicted log ratios are available.")
    edges = _bin_edges(combined[finite], bins)
    numerator_hist = _normalized_histogram(
        numerator_log,
        numerator_weight,
        edges,
    )
    denominator_hist = _normalized_histogram(
        denominator_log,
        denominator_weight,
        edges,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    populated = (numerator_hist > 0) & (denominator_hist > 0)
    empirical = np.full(len(centers), np.nan)
    empirical[populated] = np.log(
        numerator_hist[populated] / denominator_hist[populated]
    )
    residual = empirical - centers
    mass = 0.5 * (numerator_hist + denominator_hist)
    usable_mass = float(np.sum(mass[populated]))
    rmse = (
        math.sqrt(
            float(np.sum(mass[populated] * residual[populated] ** 2) / usable_mass)
        )
        if usable_mass > 0
        else math.nan
    )
    return {
        "edges": edges,
        "predicted_log_ratio": centers,
        "empirical_log_ratio": empirical,
        "residual": residual,
        "occupied_mass": usable_mass,
        "weighted_rmse": rmse,
    }


@dataclass(frozen=True)
class RatioDiagnosticReport:
    """Serializable classifier, calibration, closure, and normalization checks."""

    metrics: Mapping[str, Any]
    feature_metrics: Mapping[str, Mapping[str, float]]
    curves: Mapping[str, Any] = field(default_factory=dict)
    figure_paths: tuple[Path, ...] = ()
    checks: tuple[str, ...] = ()

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        report_path = target / "ratio_diagnostics.json"
        report_path.write_text(
            json.dumps(
                {
                    "metrics": json_safe(self.metrics),
                    "feature_metrics": json_safe(self.feature_metrics),
                    "curves": json_safe(self.curves),
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        files = {"report": report_path}
        files.update(
            {
                f"figure-{index:02d}": path
                for index, path in enumerate(self.figure_paths)
            }
        )
        manifest = target / "ratio_diagnostics.manifest.json"
        write_artifact_manifest(
            manifest,
            artifact_type="density-ratio-diagnostics",
            files=files,
            metadata={"checks": ["loss", *self.checks]},
        )
        return report_path, manifest


def diagnose_ratio(
    *,
    numerator_train: np.ndarray,
    denominator_train: np.ndarray,
    numerator_holdout: np.ndarray,
    denominator_holdout: np.ndarray,
    numerator_train_weights: np.ndarray,
    denominator_train_weights: np.ndarray,
    numerator_holdout_weights: np.ndarray,
    denominator_holdout_weights: np.ndarray,
    train_ratios: tuple[np.ndarray, np.ndarray],
    holdout_ratios: tuple[np.ndarray, np.ndarray],
    features: Sequence[str],
    history: Sequence[Mapping[str, float | int]] = (),
    bins: int = 30,
    output_directory: str | Path | None = None,
    checks: Sequence[str] | None = None,
) -> RatioDiagnosticReport:
    """Run loss, overtraining, calibration, closure, and normalization checks."""

    selected_checks = (
        {"overfit", "calibration", "reweighting", "normalization"}
        if checks is None
        else {str(value) for value in checks}
    )
    unknown_checks = selected_checks.difference(
        {"overfit", "calibration", "reweighting", "normalization"}
    )
    if unknown_checks:
        raise ValueError(f"Unknown ratio diagnostics {sorted(unknown_checks)}.")
    feature_names = tuple(features)
    n_features = len(feature_names)
    arrays = (
        numerator_train,
        denominator_train,
        numerator_holdout,
        denominator_holdout,
    )
    if any(
        np.asarray(values).ndim != 2 or np.asarray(values).shape[1] != n_features
        for values in arrays
    ):
        raise ValueError("All diagnostic samples must share the declared features.")
    ntr_ratio = _vector(train_ratios[0], name="numerator_train_ratio")
    dtr_ratio = _vector(train_ratios[1], name="denominator_train_ratio")
    nho_ratio = _vector(holdout_ratios[0], name="numerator_holdout_ratio")
    dho_ratio = _vector(holdout_ratios[1], name="denominator_holdout_ratio")
    ntr_weight = _weights(
        numerator_train_weights, len(ntr_ratio), name="numerator_train_weights"
    )
    dtr_weight = _weights(
        denominator_train_weights, len(dtr_ratio), name="denominator_train_weights"
    )
    nho_weight = _weights(
        numerator_holdout_weights, len(nho_ratio), name="numerator_holdout_weights"
    )
    dho_weight = _weights(
        denominator_holdout_weights, len(dho_ratio), name="denominator_holdout_weights"
    )
    for name, ratio in (
        ("numerator_train_ratio", ntr_ratio),
        ("denominator_train_ratio", dtr_ratio),
        ("numerator_holdout_ratio", nho_ratio),
        ("denominator_holdout_ratio", dho_ratio),
    ):
        if np.any(ratio < 0):
            raise ValueError(f"{name} must be non-negative.")
    ntr_score = ntr_ratio / (1.0 + ntr_ratio)
    dtr_score = dtr_ratio / (1.0 + dtr_ratio)
    nho_score = nho_ratio / (1.0 + nho_ratio)
    dho_score = dho_ratio / (1.0 + dho_ratio)

    training_calibration = _score_calibration(
        ntr_score,
        dtr_score,
        ntr_weight,
        dtr_weight,
        bins=bins,
    )
    holdout_calibration = _score_calibration(
        nho_score,
        dho_score,
        nho_weight,
        dho_weight,
        bins=bins,
    )
    training_log_ratio_calibration = _log_ratio_calibration(
        ntr_ratio,
        dtr_ratio,
        ntr_weight,
        dtr_weight,
        bins=bins,
    )
    holdout_log_ratio_calibration = _log_ratio_calibration(
        nho_ratio,
        dho_ratio,
        nho_weight,
        dho_weight,
        bins=bins,
    )

    feature_metrics: dict[str, dict[str, float]] = {}
    feature_curves: dict[str, Any] = {}
    for index, name in enumerate(feature_names):
        target = np.asarray(numerator_holdout)[:, index]
        reference = np.asarray(denominator_holdout)[:, index]
        edges = _bin_edges(np.concatenate([target, reference]), bins)
        target_hist = _normalized_histogram(target, nho_weight, edges)
        reference_hist = _normalized_histogram(reference, dho_weight, edges)
        reweighted_hist = _normalized_histogram(
            reference,
            dho_weight * dho_ratio,
            edges,
        )
        total_variation = 0.5 * float(np.sum(np.abs(target_hist - reweighted_hist)))
        denominator = np.maximum(target_hist + reweighted_hist, 1.0e-12)
        triangular = 0.5 * float(
            np.sum((target_hist - reweighted_hist) ** 2 / denominator)
        )
        training_target = np.asarray(numerator_train)[:, index]
        training_reference = np.asarray(denominator_train)[:, index]
        training_edges = _bin_edges(
            np.concatenate([training_target, training_reference]),
            bins,
        )
        training_target_hist = _normalized_histogram(
            training_target,
            ntr_weight,
            training_edges,
        )
        training_reference_hist = _normalized_histogram(
            training_reference,
            dtr_weight,
            training_edges,
        )
        training_reweighted_hist = _normalized_histogram(
            training_reference,
            dtr_weight * dtr_ratio,
            training_edges,
        )
        feature_metrics[name] = {
            "unweighted_total_variation": 0.5
            * float(np.sum(np.abs(target_hist - reference_hist))),
            "reweighted_total_variation": total_variation,
            "reweighted_triangular_discrimination": triangular,
            "training_unweighted_total_variation": 0.5
            * float(np.sum(np.abs(training_target_hist - training_reference_hist))),
            "training_reweighted_total_variation": 0.5
            * float(np.sum(np.abs(training_target_hist - training_reweighted_hist))),
        }
        feature_curves[name] = {
            "edges": edges,
            "target": target_hist,
            "reference": reference_hist,
            "reweighted_reference": reweighted_hist,
            "training": {
                "edges": training_edges,
                "target": training_target_hist,
                "reference": training_reference_hist,
                "reweighted_reference": training_reweighted_hist,
            },
        }

    denominator_mean = float(np.sum(dho_weight * dho_ratio))
    denominator_variance = float(
        np.sum(dho_weight * (dho_ratio - denominator_mean) ** 2)
    )
    effective_n = float(1.0 / np.sum(dho_weight**2))
    saturation_threshold = 1.0e-6
    saturation_warning_fraction = 0.01
    training_scores = np.concatenate([ntr_score, dtr_score])
    holdout_scores = np.concatenate([nho_score, dho_score])
    training_score_weights = np.concatenate([0.5 * ntr_weight, 0.5 * dtr_weight])
    holdout_score_weights = np.concatenate([0.5 * nho_weight, 0.5 * dho_weight])
    training_saturation = float(
        np.sum(
            training_score_weights
            * (
                (training_scores <= saturation_threshold)
                | (training_scores >= 1.0 - saturation_threshold)
            )
        )
    )
    holdout_saturation = float(
        np.sum(
            holdout_score_weights
            * (
                (holdout_scores <= saturation_threshold)
                | (holdout_scores >= 1.0 - saturation_threshold)
            )
        )
    )
    metrics: dict[str, Any] = {
        "classification": {
            "holdout_auc": weighted_auc(
                nho_score,
                dho_score,
                numerator_weights=nho_weight,
                denominator_weights=dho_weight,
            )
        },
        "overtraining": {
            "numerator_weighted_ks": weighted_ks(
                ntr_score,
                nho_score,
                left_weights=ntr_weight,
                right_weights=nho_weight,
            ),
            "denominator_weighted_ks": weighted_ks(
                dtr_score,
                dho_score,
                left_weights=dtr_weight,
                right_weights=dho_weight,
            ),
        },
        "calibration": {
            "brier_score": holdout_calibration["brier_score"],
            "expected_calibration_error": holdout_calibration[
                "expected_calibration_error"
            ],
            "training_brier_score": training_calibration["brier_score"],
            "training_expected_calibration_error": training_calibration[
                "expected_calibration_error"
            ],
            "holdout_log_ratio_weighted_rmse": (
                holdout_log_ratio_calibration["weighted_rmse"]
            ),
            "training_log_ratio_weighted_rmse": (
                training_log_ratio_calibration["weighted_rmse"]
            ),
        },
        "saturation": {
            "threshold": saturation_threshold,
            "warning_fraction": saturation_warning_fraction,
            "training_fraction": training_saturation,
            "holdout_fraction": holdout_saturation,
            "warning": bool(
                max(training_saturation, holdout_saturation)
                > saturation_warning_fraction
            ),
        },
        "ratio_tails": {
            "training": {
                "minimum": float(min(np.min(ntr_ratio), np.min(dtr_ratio))),
                "maximum": float(max(np.max(ntr_ratio), np.max(dtr_ratio))),
                "p001": float(
                    np.quantile(np.concatenate([ntr_ratio, dtr_ratio]), 0.001)
                ),
                "p999": float(
                    np.quantile(np.concatenate([ntr_ratio, dtr_ratio]), 0.999)
                ),
            },
            "holdout": {
                "minimum": float(min(np.min(nho_ratio), np.min(dho_ratio))),
                "maximum": float(max(np.max(nho_ratio), np.max(dho_ratio))),
                "p001": float(
                    np.quantile(np.concatenate([nho_ratio, dho_ratio]), 0.001)
                ),
                "p999": float(
                    np.quantile(np.concatenate([nho_ratio, dho_ratio]), 0.999)
                ),
            },
        },
        "normalization": {
            "reference_mean_ratio": denominator_mean,
            "reference_mean_standard_error": math.sqrt(
                max(0.0, denominator_variance) / max(effective_n, 1.0)
            ),
            "reference_effective_events": effective_n,
        },
        "loss": {
            "epochs": len(history),
            "best_validation_loss": (
                float(min(float(row["validation_loss"]) for row in history))
                if history
                else None
            ),
            "final_training_loss": (
                float(history[-1]["training_loss"]) if history else None
            ),
        },
    }
    curves = {
        "calibration": holdout_calibration,
        "training_calibration": training_calibration,
        "log_ratio_calibration": holdout_log_ratio_calibration,
        "training_log_ratio_calibration": training_log_ratio_calibration,
        "features": feature_curves,
    }
    selected_metrics = {"loss": metrics["loss"]}
    if "overfit" in selected_checks:
        selected_metrics.update(
            {
                "classification": metrics["classification"],
                "overtraining": metrics["overtraining"],
            }
        )
    if "calibration" in selected_checks:
        selected_metrics.update(
            {
                "calibration": metrics["calibration"],
                "ratio_tails": metrics["ratio_tails"],
                "saturation": metrics["saturation"],
            }
        )
    if "normalization" in selected_checks:
        selected_metrics["normalization"] = metrics["normalization"]
    selected_curves: dict[str, Any] = {}
    if "calibration" in selected_checks:
        selected_curves.update(
            {
                "calibration": curves["calibration"],
                "training_calibration": curves["training_calibration"],
                "log_ratio_calibration": curves["log_ratio_calibration"],
                "training_log_ratio_calibration": curves[
                    "training_log_ratio_calibration"
                ],
            }
        )
    if "reweighting" in selected_checks:
        selected_curves["features"] = curves["features"]
    report = RatioDiagnosticReport(
        metrics=selected_metrics,
        feature_metrics=(feature_metrics if "reweighting" in selected_checks else {}),
        curves=selected_curves,
        checks=tuple(sorted(selected_checks)),
    )
    if output_directory is None:
        return report
    figures = _plot_ratio_diagnostics(
        report=report,
        history=history,
        numerator_train_score=ntr_score,
        denominator_train_score=dtr_score,
        numerator_holdout_score=nho_score,
        denominator_holdout_score=dho_score,
        numerator_train_weights=ntr_weight,
        denominator_train_weights=dtr_weight,
        numerator_holdout_weights=nho_weight,
        denominator_holdout_weights=dho_weight,
        output_directory=Path(output_directory),
        bins=bins,
        checks=selected_checks,
    )
    return RatioDiagnosticReport(
        metrics=report.metrics,
        feature_metrics=report.feature_metrics,
        curves=report.curves,
        figure_paths=figures,
        checks=report.checks,
    )


def _plot_ratio_diagnostics(
    *,
    report: RatioDiagnosticReport,
    history: Sequence[Mapping[str, float | int]],
    numerator_train_score: np.ndarray,
    denominator_train_score: np.ndarray,
    numerator_holdout_score: np.ndarray,
    denominator_holdout_score: np.ndarray,
    numerator_train_weights: np.ndarray,
    denominator_train_weights: np.ndarray,
    numerator_holdout_weights: np.ndarray,
    denominator_holdout_weights: np.ndarray,
    output_directory: Path,
    bins: int,
    checks: set[str],
) -> tuple[Path, ...]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return ()
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if history:
        figure, axis = plt.subplots()
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [row["training_loss"] for row in history], label="train")
        axis.plot(
            epochs,
            [row["validation_loss"] for row in history],
            label="validation",
        )
        axis.set(xlabel="epoch", ylabel="weighted BCE", title="Ratio training")
        axis.legend()
        path = output_directory / "loss.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    if "overfit" in checks:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        score_edges = np.linspace(0.0, 1.0, int(bins) + 1)
        for axis, label, train, holdout, train_weight, holdout_weight in (
            (
                axes[0],
                "numerator",
                numerator_train_score,
                numerator_holdout_score,
                numerator_train_weights,
                numerator_holdout_weights,
            ),
            (
                axes[1],
                "reference",
                denominator_train_score,
                denominator_holdout_score,
                denominator_train_weights,
                denominator_holdout_weights,
            ),
        ):
            axis.hist(
                train,
                bins=score_edges,
                weights=train_weight,
                histtype="step",
                label="train",
            )
            axis.hist(
                holdout,
                bins=score_edges,
                weights=holdout_weight,
                histtype="step",
                label="holdout",
            )
            axis.set(xlabel="classifier score", title=label)
            axis.legend()
        axes[0].set_ylabel("probability per bin")
        path = output_directory / "overtraining.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    if "calibration" in checks:
        calibrations = (
            ("training", report.curves["training_calibration"]),
            ("holdout", report.curves["calibration"]),
        )
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(10, 6),
            sharex="col",
            gridspec_kw={"height_ratios": [3, 1]},
        )
        for column, (label, calibration) in enumerate(calibrations):
            axes[0, column].plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                color="black",
                label="ideal",
            )
            axes[0, column].plot(
                calibration["mean_prediction"],
                calibration["observed_fraction"],
                marker="o",
                label=label,
            )
            axes[0, column].set(
                ylabel="observed numerator fraction",
                title=f"Calibration ({label})",
            )
            axes[0, column].legend()
            axes[1, column].axhline(1.0, linestyle="--", color="black")
            axes[1, column].plot(
                calibration["mean_prediction"],
                calibration["observed_over_prediction"],
                marker="o",
            )
            axes[1, column].set(
                xlabel="mean predicted probability",
                ylabel="obs./pred.",
            )
        path = output_directory / "calibration.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

        figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        for axis, (label, curve) in zip(
            axes,
            (
                ("training", report.curves["training_log_ratio_calibration"]),
                ("holdout", report.curves["log_ratio_calibration"]),
            ),
            strict=True,
        ):
            predicted = np.asarray(curve["predicted_log_ratio"])
            empirical = np.asarray(curve["empirical_log_ratio"])
            finite = np.isfinite(empirical)
            axis.plot(predicted[finite], empirical[finite], marker="o")
            if np.any(finite):
                low = float(min(np.min(predicted[finite]), np.min(empirical[finite])))
                high = float(max(np.max(predicted[finite]), np.max(empirical[finite])))
                axis.plot([low, high], [low, high], linestyle="--", color="black")
            axis.set(
                xlabel="predicted log density ratio",
                ylabel="MC log density ratio",
                title=f"Log-ratio calibration ({label})",
            )
        path = output_directory / "calibration_log_ratio.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)

    for name, payload in report.curves.get("features", {}).items():
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for axis, label, curve in (
            (axes[0], "training", payload["training"]),
            (axes[1], "holdout", payload),
        ):
            edges = np.asarray(curve["edges"])
            centers = 0.5 * (edges[:-1] + edges[1:])
            axis.step(centers, curve["target"], where="mid", label="target")
            axis.step(
                centers,
                curve["reference"],
                where="mid",
                label="reference",
            )
            axis.step(
                centers,
                curve["reweighted_reference"],
                where="mid",
                label="ratio × reference",
            )
            axis.set(xlabel=name, title=f"Reweighting closure ({label})")
            axis.legend()
        axes[0].set_ylabel("probability per bin")
        path = output_directory / f"reweighted_{name}.png"
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        paths.append(path)
    return tuple(paths)
