"""Generate a reduced synthetic BBH problem inspired by DINGO examples.

This module is deliberately independent of DINGO, Bilby, and LALSuite.  It
implements a small frequency-domain inspiral surrogate with additive,
whitened Gaussian noise.  The resulting likelihood is known exactly and is
therefore useful for validating the dual hNPE--hNDE construction.

It is not a waveform model for gravitational-wave data analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

import numpy as np

THETA_FEATURES: Final[tuple[str, ...]] = (
    "chirp_mass_msun",
    "mass_ratio",
    "phase",
    "geocent_time_s",
    "luminosity_distance_mpc",
)
FREQUENCIES_HZ: Final[np.ndarray] = np.geomspace(30.0, 256.0, 12)
OBSERVATION_FEATURES: Final[tuple[str, ...]] = tuple(
    name
    for index in range(len(FREQUENCIES_HZ))
    for name in (f"strain_f{index:02d}_re", f"strain_f{index:02d}_im")
)
PRIOR_LOW: Final[np.ndarray] = np.asarray(
    [22.0, 0.5, -np.pi, -0.02, 500.0],
    dtype=np.float64,
)
PRIOR_HIGH: Final[np.ndarray] = np.asarray(
    [45.0, 1.0, np.pi, 0.02, 1500.0],
    dtype=np.float64,
)
SOLAR_MASS_SECONDS: Final[float] = 4.925490947e-6
DATASET_FILENAMES: Final[dict[str, str]] = {
    "rho": "rho_posterior_flow.parquet",
    "rho_residual": "rho_posterior_residual.parquet",
    "nu": "nu_likelihood_flow.parquet",
    "kappa": "kappa_likelihood_residual.parquet",
    "validation": "independent_validation.parquet",
}
DATASET_CODES: Final[dict[str, int]] = {
    "rho": 1,
    "rho_residual": 2,
    "nu": 3,
    "kappa": 4,
    "validation": 5,
}
PROFILES: Final[dict[str, tuple[int, int]]] = {
    "smoke": (64, 32),
    "quick": (12_000, 2_000),
    "publication": (50_000, 10_000),
}


def _as_rows(values: np.ndarray, *, width: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim == 1:
        result = result.reshape(1, -1)
    if result.ndim != 2 or result.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width}).")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values.")
    return result


def waveform_mean(theta: np.ndarray) -> np.ndarray:
    """Return the deterministic reduced whitened strain mean.

    The phase contains the Newtonian and 1PN non-spinning inspiral terms.  The
    amplitude is a pedagogical normalization chosen to yield moderate SNRs.
    """

    rows = _as_rows(theta, width=len(THETA_FEATURES), name="theta")
    chirp_mass, mass_ratio, phase, geocent_time, distance = rows.T
    if np.any((rows < PRIOR_LOW) | (rows > PRIOR_HIGH)):
        raise ValueError("theta lies outside the declared design box.")

    symmetric_mass_ratio = mass_ratio / (1.0 + mass_ratio) ** 2.0
    total_mass = chirp_mass / symmetric_mass_ratio ** (3.0 / 5.0)
    frequency = FREQUENCIES_HZ.reshape(1, -1)
    velocity = (np.pi * total_mass.reshape(-1, 1) * SOLAR_MASS_SECONDS * frequency) ** (
        1.0 / 3.0
    )
    one_pn = (
        3715.0 / 756.0 + 55.0 * symmetric_mass_ratio.reshape(-1, 1) / 9.0
    ) * velocity**2.0
    inspiral_phase = (
        2.0 * np.pi * frequency * geocent_time.reshape(-1, 1)
        - phase.reshape(-1, 1)
        - np.pi / 4.0
        + 3.0
        / (128.0 * symmetric_mass_ratio.reshape(-1, 1) * velocity**5.0)
        * (1.0 + one_pn)
    )
    amplitude = (
        1.8
        * (chirp_mass.reshape(-1, 1) / 30.0) ** (5.0 / 6.0)
        * (1000.0 / distance.reshape(-1, 1))
        * (frequency / 100.0) ** (-7.0 / 6.0)
    )
    strain = amplitude * np.exp(1j * inspiral_phase)
    return np.stack((strain.real, strain.imag), axis=-1).reshape(len(rows), -1)


def exact_log_likelihood(theta: np.ndarray, observation: np.ndarray) -> np.ndarray:
    """Evaluate the normalized reduced-data Gaussian log likelihood."""

    theta_rows = _as_rows(theta, width=len(THETA_FEATURES), name="theta")
    observation_rows = _as_rows(
        observation,
        width=len(OBSERVATION_FEATURES),
        name="observation",
    )
    if len(theta_rows) == 1 and len(observation_rows) > 1:
        theta_rows = np.repeat(theta_rows, len(observation_rows), axis=0)
    if len(observation_rows) == 1 and len(theta_rows) > 1:
        observation_rows = np.repeat(observation_rows, len(theta_rows), axis=0)
    if len(theta_rows) != len(observation_rows):
        raise ValueError("theta and observation rows cannot be aligned.")
    residual = observation_rows - waveform_mean(theta_rows)
    dimension = residual.shape[1]
    return -0.5 * (np.sum(residual**2.0, axis=1) + dimension * np.log(2.0 * np.pi))


def sample_theta(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw from the box design used by rho, nu, and kappa."""

    if int(n) < 1:
        raise ValueError("n must be positive.")
    return rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(int(n), len(PRIOR_LOW)))


def simulate(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw parameter/observation pairs from the reduced simulator."""

    theta = sample_theta(int(n), rng)
    noise = rng.standard_normal((int(n), len(OBSERVATION_FEATURES)))
    observation = waveform_mean(theta) + noise
    return theta.astype(np.float32), observation.astype(np.float32)


def scientific_splits(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return shuffled train/validation/holdout labels with non-empty parts."""

    n = int(n)
    if n < 3:
        raise ValueError("At least three rows are required for scientific splits.")
    n_validation = max(1, int(round(0.1 * n)))
    n_holdout = max(1, int(round(0.1 * n)))
    if n_validation + n_holdout >= n:
        n_validation = n_holdout = 1
    labels = np.full(n, "train", dtype=object)
    labels[:n_validation] = "validation"
    labels[n_validation : n_validation + n_holdout] = "holdout"
    return labels[rng.permutation(n)]


def _table_for_design(
    design: str,
    *,
    n: int,
    rng: np.random.Generator,
    include_split: bool,
):
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "Generating Parquet data requires pyarrow; install hnsbi-toolkit[bayes]."
        ) from exc

    theta, observation = simulate(n, rng)
    values: dict[str, object] = {
        name: theta[:, index] for index, name in enumerate(THETA_FEATURES)
    }
    values.update(
        {name: observation[:, index] for index, name in enumerate(OBSERVATION_FEATURES)}
    )
    code = DATASET_CODES[design]
    values["simulation_id"] = code * 1_000_000_000 + np.arange(int(n), dtype=np.int64)
    if include_split:
        values["split"] = scientific_splits(n, rng)
    return pa.table(values)


def generate_datasets(
    output_dir: str | Path,
    *,
    rows_per_design: int = PROFILES["quick"][0],
    validation_rows: int = PROFILES["quick"][1],
    seed: int = 20_260_727,
) -> dict[str, Path]:
    """Generate all independent dual-training sources and a manifest."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Generating Parquet data requires pyarrow; install hnsbi-toolkit[bayes]."
        ) from exc

    rows_per_design = int(rows_per_design)
    validation_rows = int(validation_rows)
    if rows_per_design < 3:
        raise ValueError("rows_per_design must be at least three.")
    if validation_rows < 1:
        raise ValueError("validation_rows must be positive.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    streams = np.random.SeedSequence(int(seed)).spawn(len(DATASET_FILENAMES))
    paths: dict[str, Path] = {}
    for (design, filename), stream in zip(
        DATASET_FILENAMES.items(), streams, strict=True
    ):
        n = validation_rows if design == "validation" else rows_per_design
        table = _table_for_design(
            design,
            n=n,
            rng=np.random.default_rng(stream),
            include_split=design != "validation",
        )
        path = destination / filename
        pq.write_table(table, path, compression="zstd")
        paths[design] = path

    manifest = {
        "example": "dingo-inspired-reduced-bbh",
        "seed": int(seed),
        "rows_per_design": rows_per_design,
        "validation_rows": validation_rows,
        "theta_features": list(THETA_FEATURES),
        "observation_features": list(OBSERVATION_FEATURES),
        "frequencies_hz": FREQUENCIES_HZ.tolist(),
        "prior_low": PRIOR_LOW.tolist(),
        "prior_high": PRIOR_HIGH.tolist(),
        "noise": "independent standard normal in every scalar coefficient",
        "likelihood": "normalized reduced Gaussian; see exact_log_likelihood",
        "not_dingo": True,
        "datasets": {name: path.name for name, path in paths.items()},
    }
    manifest_path = destination / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["manifest"] = manifest_path
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="quick",
        help="Dataset-size preset; explicit row options override it.",
    )
    parser.add_argument("--rows-per-design", type=int)
    parser.add_argument("--validation-rows", type=int)
    parser.add_argument("--seed", type=int, default=20_260_727)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    profile_rows, profile_validation = PROFILES[arguments.profile]
    paths = generate_datasets(
        arguments.output_dir,
        rows_per_design=(
            profile_rows
            if arguments.rows_per_design is None
            else arguments.rows_per_design
        ),
        validation_rows=(
            profile_validation
            if arguments.validation_rows is None
            else arguments.validation_rows
        ),
        seed=arguments.seed,
    )
    print(f"Wrote {len(paths) - 1} datasets and {paths['manifest']}")


if __name__ == "__main__":
    main()
