"""Generate the complete synthetic LHC analysis used by this example.

The nominal mixtures and detector response are the same construction used by
the ML4HEP-TIF ``generate_distributions.py`` tutorial. This version adds
response-scale, resolution, and signal-theory anchors while preserving the
latent/reconstruction columns needed for generator-level validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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


def generate(
    output: str | Path,
    *,
    signal_events: int = 20_000,
    background_events: int = 50_000,
    reference_events: int = 60_000,
    seed: int = 20260727,
) -> dict[str, Path]:
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    sequence = np.random.SeedSequence(seed)
    children = iter(sequence.spawn(8))
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
    paths: dict[str, Path] = {}
    for sample, frame in nominal.items():
        path = target / f"{sample}.parquet"
        frame.to_parquet(path, index=False)
        paths[sample] = path
        for label, kwargs in {
            "response_up": {"response_scale": 1.10},
            "response_down": {"response_scale": 0.90},
            "resolution_up": {"resolution_scale": 1.25},
            "resolution_down": {"resolution_scale": 0.75},
        }.items():
            varied = detector_variation(frame, **kwargs)
            varied_path = target / f"{sample}_{label}.parquet"
            varied.to_parquet(varied_path, index=False)
            paths[f"{sample}_{label}"] = varied_path
    for label, value in (("up", 1.0), ("down", -1.0)):
        frame = event_frame(
            components=signal_components(theory=value),
            size=signal_events,
            expected_yield=EXPECTED_YIELDS["signal"],
            rng=np.random.default_rng(next(children)),
            sample=f"signal-theory-{label}",
        )
        path = target / f"signal_theory_{label}.parquet"
        frame.to_parquet(path, index=False)
        paths[f"signal_theory_{label}"] = path
    reference_signal = sample_mixture(
        signal_components(),
        reference_events // 2,
        np.random.default_rng(next(children)),
    )
    reference_background = sample_mixture(
        background_components(),
        reference_events - reference_events // 2,
        np.random.default_rng(next(children)),
    )
    latent = np.vstack([reference_signal, reference_background])
    rng = np.random.default_rng(next(children))
    residual = rng.normal(size=latent.shape)
    reference = pd.DataFrame(
        reconstruct(latent, residual),
        columns=FEATURES,
    )
    for index, feature in enumerate(LATENT):
        reference[feature] = latent[:, index]
    reference["event_id"] = [
        f"reference-{index:09d}" for index in range(len(reference))
    ]
    reference["weight"] = 1.0
    reference["split"] = "train"
    reference_path = target / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    paths["reference"] = reference_path
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
    print(f"Wrote {len(generated)} Parquet samples to {arguments.output}.")


if __name__ == "__main__":
    main()
