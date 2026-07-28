"""Generate the complete synthetic LHC analysis used by this example.

The nominal mixtures and detector response are the same construction used by
the ML4HEP-TIF ``generate_distributions.py`` tutorial. This version adds
response-scale, resolution, and signal-theory anchors while preserving the
latent/reconstruction columns needed for generator-level validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from preselection import (
        MANIFEST_NAME,
        GaussianMixtureRatioSelector,
        PreselectionConfig,
        derive_ratio_cut,
        reconstructed_components,
        select_process_frame,
        select_reference_frame,
        write_parquet_atomic,
        write_preselection_manifest,
    )
except ModuleNotFoundError:
    # Package-style import used by the test suite and repository scripts.
    from examples.lhc_analysis.preselection import (
        MANIFEST_NAME,
        GaussianMixtureRatioSelector,
        PreselectionConfig,
        derive_ratio_cut,
        reconstructed_components,
        select_process_frame,
        select_reference_frame,
        write_parquet_atomic,
        write_preselection_manifest,
    )

FEATURES = ("x1", "x2", "x3", "x4", "x5")
LATENT = ("z1", "z2", "z3", "z4", "z5")
SCALE = np.asarray([1.2, 1.1, 0.99, 0.96, 1.01])
RESOLUTION = np.asarray([1.0, 0.1, 0.9, 1.3, 0.2])
THEORY_SHIFT = np.asarray([0.35, -0.20, 0.30, 0.15, -0.25])
EXPECTED_YIELDS = {"signal": 1_100.0, "background": 1_000_000.0}


def _covariance(
    sigmas: list[float],
    correlations: dict[tuple[int, int], float] | None = None,
) -> np.ndarray:
    sigma = np.asarray(sigmas)
    correlation = np.eye(len(sigma))
    for (left, right), value in (correlations or {}).items():
        correlation[left, right] = correlation[right, left] = value
    return np.outer(sigma, sigma) * correlation


def background_components() -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    return (
        (
            0.45,
            np.asarray([2.0, 1.0, 2.5, 0.8, 2.0]),
            _covariance(
                [1.0, 0.9, 1.2, 0.9, 1.0],
                {(0, 1): 0.6, (3, 4): 0.5},
            ),
        ),
        (
            0.35,
            np.asarray([4.2, 3.1, 4.6, 2.6, 3.6]),
            _covariance(
                [0.9, 1.0, 1.0, 0.9, 1.1],
                {(0, 1): -0.5, (2, 3): 0.5},
            ),
        ),
        (
            0.20,
            np.asarray([2.5, 2.0, 3.0, 1.5, 2.5]),
            _covariance([2.5, 2.4, 2.5, 2.4, 2.5]),
        ),
    )


def signal_components(
    *,
    theory: float = 0.0,
) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    components = (
        (
            0.50,
            np.asarray([4.5, 3.2, 4.3, 2.3, 4.0]),
            _covariance(
                [0.9, 0.8, 1.0, 0.8, 0.9],
                {(0, 2): 0.5, (1, 4): 0.4},
            ),
        ),
        (
            0.30,
            np.asarray([4.3, 3.2, 5.0, 2.6, 4.0]),
            _covariance(
                [0.8, 0.9, 0.9, 0.8, 1.0],
                {(0, 1): 0.5, (2, 4): -0.4},
            ),
        ),
        (
            0.20,
            np.asarray([2.5, 2.0, 3.0, 1.5, 2.5]),
            _covariance([2.5, 2.4, 2.5, 2.4, 2.5]),
        ),
    )
    # The broad common-support component stays fixed. The theory nuisance
    # changes the centers of the two signal Gaussian components in latent z.
    return tuple(
        (
            fraction,
            mean + float(theory) * THEORY_SHIFT if index < 2 else mean,
            covariance,
        )
        for index, (fraction, mean, covariance) in enumerate(components)
    )


def sample_mixture(
    components: tuple[tuple[float, np.ndarray, np.ndarray], ...],
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    fractions = np.asarray([component[0] for component in components])
    counts = rng.multinomial(int(size), fractions / np.sum(fractions))
    values = np.concatenate(
        [
            rng.multivariate_normal(mean, covariance, size=count)
            for count, (_, mean, covariance) in zip(
                counts,
                components,
                strict=True,
            )
            if count
        ],
        axis=0,
    )
    rng.shuffle(values)
    return values


def reconstruct(
    latent: np.ndarray,
    residual: np.ndarray,
    *,
    response_scale: float = 1.0,
    resolution_scale: float = 1.0,
) -> np.ndarray:
    return (
        float(response_scale) * latent * SCALE
        + float(resolution_scale) * residual * RESOLUTION
    )


def event_frame(
    *,
    components: tuple[tuple[float, np.ndarray, np.ndarray], ...],
    size: int,
    expected_yield: float,
    rng: np.random.Generator,
    sample: str,
    response_scale: float = 1.0,
    resolution_scale: float = 1.0,
) -> pd.DataFrame:
    latent = sample_mixture(components, size, rng)
    residual = rng.normal(size=latent.shape)
    reconstructed = reconstruct(
        latent,
        residual,
        response_scale=response_scale,
        resolution_scale=resolution_scale,
    )
    frame = pd.DataFrame(latent, columns=LATENT)
    for index, feature in enumerate(FEATURES):
        frame[feature] = reconstructed[:, index]
    frame["event_id"] = [f"{sample}-{index:09d}" for index in range(size)]
    frame["weight"] = float(expected_yield) / size
    frame["split"] = np.where(
        np.arange(size) % 10 == 0,
        "holdout",
        np.where(np.arange(size) % 10 == 1, "validation", "train"),
    )
    return frame


def detector_variation(
    nominal: pd.DataFrame,
    *,
    response_scale: float = 1.0,
    resolution_scale: float = 1.0,
) -> pd.DataFrame:
    latent = nominal.loc[:, list(LATENT)].to_numpy()
    reconstructed = nominal.loc[:, list(FEATURES)].to_numpy()
    residual = (reconstructed - latent * SCALE) / RESOLUTION
    varied = nominal.copy()
    varied.loc[:, FEATURES] = reconstruct(
        latent,
        residual,
        response_scale=response_scale,
        resolution_scale=resolution_scale,
    )
    return varied


def nominal_preselection_selector() -> GaussianMixtureRatioSelector:
    """Return the deterministic legacy-equivalent reconstructed ratio."""

    return GaussianMixtureRatioSelector(
        features=FEATURES,
        signal_components=reconstructed_components(
            signal_components(),
            scale=SCALE,
            resolution=RESOLUTION,
        ),
        background_components=reconstructed_components(
            background_components(),
            scale=SCALE,
            resolution=RESOLUTION,
        ),
    )


def _reference_component_frame(
    *,
    components: tuple[tuple[float, np.ndarray, np.ndarray], ...],
    size: int,
    rng: np.random.Generator,
    component: str,
) -> pd.DataFrame:
    latent = sample_mixture(components, size, rng)
    reconstructed = reconstruct(latent, rng.normal(size=latent.shape))
    frame = pd.DataFrame(latent, columns=LATENT)
    for index, feature in enumerate(FEATURES):
        frame[feature] = reconstructed[:, index]
    frame["reference_component"] = component
    return frame


def _inclusive_reference(
    *,
    size: int,
    signal_rng: np.random.Generator,
    background_rng: np.random.Generator,
    shuffle_rng: np.random.Generator,
) -> pd.DataFrame:
    signal_size = size // 2
    background_size = size - signal_size
    frame = pd.concat(
        [
            _reference_component_frame(
                components=signal_components(),
                size=signal_size,
                rng=signal_rng,
                component="signal",
            ),
            _reference_component_frame(
                components=background_components(),
                size=background_size,
                rng=background_rng,
                component="background",
            ),
        ],
        ignore_index=True,
    )
    frame = frame.iloc[shuffle_rng.permutation(len(frame))].reset_index(drop=True)
    frame["event_id"] = [
        f"reference-{index:09d}" for index in range(len(frame))
    ]
    frame["weight"] = 1.0
    frame["split"] = "train"
    return frame


def _selected_reference(
    *,
    size: int,
    selector: GaussianMixtureRatioSelector,
    ratio_cut: float,
    config: PreselectionConfig,
    signal_rng: np.random.Generator,
    background_rng: np.random.Generator,
    shuffle_rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rejection-generate an exactly balanced post-selection reference."""

    if size % 2:
        raise ValueError(
            "reference_events must be even so the selected reference can contain "
            "equal signal and background counts."
        )
    per_component = size // 2
    accepted_components: list[pd.DataFrame] = []
    candidate_counts: dict[str, int] = {}
    for name, components, rng in (
        ("signal", signal_components(), signal_rng),
        ("background", background_components(), background_rng),
    ):
        accepted: list[pd.DataFrame] = []
        accepted_rows = 0
        candidate_rows = 0
        while accepted_rows < per_component:
            candidate_size = max(
                int(config.reference_batch_size),
                per_component - accepted_rows,
            )
            candidates = _reference_component_frame(
                components=components,
                size=candidate_size,
                rng=rng,
                component=name,
            )
            candidate_rows += len(candidates)
            passing = select_reference_frame(
                candidates,
                selector=selector,
                ratio_cut=ratio_cut,
            )
            if len(passing):
                needed = per_component - accepted_rows
                accepted.append(passing.iloc[:needed].copy())
                accepted_rows += min(len(passing), needed)
        component_frame = pd.concat(accepted, ignore_index=True)
        component_frame["weight"] = 0.5 / per_component
        accepted_components.append(component_frame)
        candidate_counts[name] = candidate_rows

    frame = pd.concat(accepted_components, ignore_index=True)
    frame = frame.iloc[shuffle_rng.permutation(len(frame))].reset_index(drop=True)
    frame["event_id"] = [
        f"reference-presel-{index:09d}" for index in range(len(frame))
    ]
    frame["split"] = "train"
    frame["preselection_split"] = "train"
    frame["preselection_partition"] = "independent_reference"
    component_weights = {
        name: float(
            frame.loc[frame["reference_component"] == name, "weight"].sum()
        )
        for name in ("signal", "background")
    }
    return frame, {
        "candidate_events": candidate_counts,
        "component_events": {
            name: int(np.sum(frame["reference_component"] == name))
            for name in ("signal", "background")
        },
        "component_weights": component_weights,
        "selected_events": int(len(frame)),
        "selected_weight": float(frame["weight"].sum()),
    }


def generate(
    output: str | Path,
    *,
    signal_events: int = 20_000,
    background_events: int = 50_000,
    reference_events: int = 60_000,
    seed: int = 20260727,
    preselection: PreselectionConfig | None = None,
) -> dict[str, Path]:
    """Write inclusive and selected copies of every synthetic sample.

    The one nominal cut is derived from the disjoint legacy flow-training
    partition, applied unchanged to all reconstructed variations, and recorded
    with checksums in ``preselection.manifest.json``.
    """

    requested_events = {
        "signal": int(signal_events),
        "background": int(background_events),
        "reference": int(reference_events),
    }
    if any(value < 2 for value in requested_events.values()):
        raise ValueError("Every requested event count must be at least two.")
    if requested_events["reference"] % 2:
        raise ValueError("reference_events must be even.")
    selection_config = preselection or PreselectionConfig()
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    (target / MANIFEST_NAME).unlink(missing_ok=True)
    sequence = np.random.SeedSequence(seed)
    children = iter(sequence.spawn(11))
    nominal = {
        "signal": event_frame(
            components=signal_components(),
            size=signal_events,
            expected_yield=EXPECTED_YIELDS["signal"],
            rng=np.random.default_rng(next(children)),
            sample="signal",
        ),
        "background": event_frame(
            components=background_components(),
            size=background_events,
            expected_yield=EXPECTED_YIELDS["background"],
            rng=np.random.default_rng(next(children)),
            sample="background",
        ),
    }
    raw_frames: dict[str, pd.DataFrame] = {}
    for sample, frame in nominal.items():
        raw_frames[sample] = frame
        for label, kwargs in {
            "response_up": {"response_scale": 1.10},
            "response_down": {"response_scale": 0.90},
            "resolution_up": {"resolution_scale": 1.25},
            "resolution_down": {"resolution_scale": 0.75},
        }.items():
            raw_frames[f"{sample}_{label}"] = detector_variation(frame, **kwargs)
    for label, value in (("up", 1.0), ("down", -1.0)):
        raw_frames[f"signal_theory_{label}"] = event_frame(
            components=signal_components(theory=value),
            size=signal_events,
            expected_yield=EXPECTED_YIELDS["signal"],
            rng=np.random.default_rng(next(children)),
            sample=f"signal-theory-{label}",
        )
    raw_frames["reference"] = _inclusive_reference(
        size=reference_events,
        signal_rng=np.random.default_rng(next(children)),
        background_rng=np.random.default_rng(next(children)),
        shuffle_rng=np.random.default_rng(next(children)),
    )

    raw_paths = {
        name: write_parquet_atomic(frame, target / f"{name}.parquet")
        for name, frame in raw_frames.items()
    }
    selector = nominal_preselection_selector()
    ratio_cut, cut_diagnostics = derive_ratio_cut(
        nominal["signal"],
        nominal["background"],
        selector=selector,
        config=selection_config,
    )
    selected_frames: dict[str, pd.DataFrame] = {}
    sample_diagnostics: dict[str, dict[str, float | int]] = {}
    for name, frame in raw_frames.items():
        if name == "reference":
            continue
        selected, diagnostics = select_process_frame(
            frame,
            selector=selector,
            ratio_cut=ratio_cut,
            config=selection_config,
        )
        selected_frames[name] = selected
        sample_diagnostics[name] = diagnostics
    selected_reference, reference_diagnostics = _selected_reference(
        size=reference_events,
        selector=selector,
        ratio_cut=ratio_cut,
        config=selection_config,
        signal_rng=np.random.default_rng(next(children)),
        background_rng=np.random.default_rng(next(children)),
        shuffle_rng=np.random.default_rng(next(children)),
    )
    selected_frames["reference"] = selected_reference
    selected_paths = {
        name: write_parquet_atomic(frame, target / f"{name}_presel.parquet")
        for name, frame in selected_frames.items()
    }
    manifest_path = write_preselection_manifest(
        target,
        raw_paths=raw_paths,
        selected_paths=selected_paths,
        selector=selector,
        ratio_cut=ratio_cut,
        cut_diagnostics=cut_diagnostics,
        config=selection_config,
        sample_diagnostics=sample_diagnostics,
        reference_diagnostics=reference_diagnostics,
        requested_events=requested_events,
        generation_seed=seed,
    )
    paths = dict(raw_paths)
    paths.update(
        {f"{name}_presel": path for name, path in selected_paths.items()}
    )
    paths["preselection_manifest"] = manifest_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--signal-events", type=int, default=20_000)
    parser.add_argument("--background-events", type=int, default=50_000)
    parser.add_argument("--reference-events", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260727)
    arguments = parser.parse_args()
    generated = generate(
        arguments.output,
        signal_events=arguments.signal_events,
        background_events=arguments.background_events,
        reference_events=arguments.reference_events,
        seed=arguments.seed,
    )
    parquet_count = sum(path.suffix == ".parquet" for path in generated.values())
    print(
        f"Wrote {parquet_count} Parquet samples and "
        f"{generated['preselection_manifest'].name} to {arguments.output}."
    )


if __name__ == "__main__":
    main()
