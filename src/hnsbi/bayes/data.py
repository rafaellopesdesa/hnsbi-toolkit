"""Proposal-sampled datasets and leakage-safe paired classifier data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._array import as_2d

SCIENTIFIC_SPLIT_LABELS = ("train", "validation", "holdout")
"""Canonical labels accepted from a configured Bayesian ``split_column``."""


def _validated_split_values(
    values: np.ndarray | None,
    *,
    rows: int,
) -> np.ndarray | None:
    if values is None:
        return None
    split_values = np.asarray(values).reshape(-1)
    if len(split_values) != rows:
        raise ValueError("split_values must contain one value per simulator row.")
    labels = np.asarray([str(value) for value in split_values], dtype=str)
    invalid = sorted(set(labels).difference(SCIENTIFIC_SPLIT_LABELS))
    if invalid:
        raise ValueError(
            "split_values accepts exactly 'train', 'validation', or 'holdout'; "
            f"received {invalid}."
        )
    return labels


@dataclass(frozen=True)
class ProposalDataset:
    """Dense simulator pairs sampled from one declared parameter design.

    ``simulation_ids`` identify the underlying simulator call. They are the
    grouping unit used when a positive row and a flow-generated negative row
    are split into ratio-training and validation subsets.
    """

    theta: np.ndarray
    observation: np.ndarray
    simulation_ids: np.ndarray
    design: str
    parameter_names: tuple[str, ...] = ()
    observation_names: tuple[str, ...] = ()
    split_values: np.ndarray | None = None
    log_density: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        theta = as_2d(self.theta, "theta")
        observation = as_2d(self.observation, "observation")
        simulation_ids = np.asarray(self.simulation_ids).reshape(-1)
        if len(theta) != len(observation):
            raise ValueError(
                "theta and observation must contain the same number of rows."
            )
        if len(simulation_ids) != len(theta):
            raise ValueError(
                "simulation_ids must contain one identifier per simulator row."
            )
        if len(np.unique(simulation_ids)) != len(simulation_ids):
            raise ValueError("simulation_ids must be unique within a proposal dataset.")
        design = str(self.design).strip()
        if not design:
            raise ValueError("design must be a non-empty name.")
        parameter_names = tuple(self.parameter_names)
        observation_names = tuple(self.observation_names)
        if parameter_names and len(parameter_names) != theta.shape[1]:
            raise ValueError(
                "parameter_names do not match the number of theta columns."
            )
        if observation_names and len(observation_names) != observation.shape[1]:
            raise ValueError("observation_names do not match the observation columns.")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("parameter_names must be unique.")
        if len(set(observation_names)) != len(observation_names):
            raise ValueError("observation_names must be unique.")
        split_values = _validated_split_values(self.split_values, rows=len(theta))
        log_density = (
            None
            if self.log_density is None
            else np.asarray(self.log_density, dtype=np.float64).reshape(-1)
        )
        if log_density is not None and (
            len(log_density) != len(theta) or not np.isfinite(log_density).all()
        ):
            raise ValueError("log_density must align and contain finite values.")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "simulation_ids", simulation_ids)
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "observation_names", observation_names)
        object.__setattr__(self, "split_values", split_values)
        object.__setattr__(self, "log_density", log_density)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def subset(self, indices: np.ndarray) -> ProposalDataset:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        return ProposalDataset(
            theta=self.theta[indices],
            observation=self.observation[indices],
            simulation_ids=self.simulation_ids[indices],
            design=self.design,
            parameter_names=self.parameter_names,
            observation_names=self.observation_names,
            split_values=(
                None if self.split_values is None else self.split_values[indices]
            ),
            log_density=(
                None if self.log_density is None else self.log_density[indices]
            ),
            metadata=self.metadata,
        )

    def split_subset(
        self,
        label: str,
        *,
        required: bool = False,
    ) -> ProposalDataset | None:
        """Return one explicit scientific split without crossing simulator rows.

        When no ``split_column`` was configured, ``train`` means the complete
        dataset and the two evaluation subsets are absent. Explicit labels are
        never inferred or silently remapped.
        """

        if label not in SCIENTIFIC_SPLIT_LABELS:
            raise ValueError(f"label must be one of {list(SCIENTIFIC_SPLIT_LABELS)}.")
        if self.split_values is None:
            result = self if label == "train" else None
        else:
            indices = np.flatnonzero(self.split_values == label)
            result = self.subset(indices) if len(indices) else None
        if required and result is None:
            raise ValueError(
                f"Dataset {self.design!r} has no rows in required split {label!r}."
            )
        return result


@dataclass(frozen=True)
class GroupSplit:
    """Indices for a disjoint group-wise training/validation split."""

    training_indices: np.ndarray
    validation_indices: np.ndarray
    training_groups: np.ndarray
    validation_groups: np.ndarray

    def __post_init__(self) -> None:
        training_indices = np.asarray(self.training_indices, dtype=np.int64).reshape(-1)
        validation_indices = np.asarray(
            self.validation_indices, dtype=np.int64
        ).reshape(-1)
        training_groups = np.asarray(self.training_groups).reshape(-1)
        validation_groups = np.asarray(self.validation_groups).reshape(-1)
        if np.intersect1d(training_indices, validation_indices).size:
            raise ValueError("Training and validation row indices overlap.")
        if np.intersect1d(training_groups, validation_groups).size:
            raise ValueError("Training and validation groups overlap.")
        object.__setattr__(self, "training_indices", training_indices)
        object.__setattr__(self, "validation_indices", validation_indices)
        object.__setattr__(self, "training_groups", training_groups)
        object.__setattr__(self, "validation_groups", validation_groups)


def group_train_validation_split(
    group_ids: np.ndarray,
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
) -> GroupSplit:
    """Split complete groups before any paired classifier rows are stacked."""

    groups = np.asarray(group_ids).reshape(-1)
    if len(groups) < 2:
        raise ValueError("At least two grouped rows are required.")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two distinct groups are required.")
    validation_fraction = float(validation_fraction)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    order = np.random.default_rng(int(seed)).permutation(unique)
    n_validation = min(
        len(order) - 1,
        max(1, int(round(validation_fraction * len(order)))),
    )
    validation_groups = order[:n_validation]
    training_groups = order[n_validation:]
    validation_mask = np.isin(groups, validation_groups)
    training_mask = np.isin(groups, training_groups)
    if not np.all(validation_mask | training_mask):
        raise RuntimeError("Group splitting lost one or more rows.")
    return GroupSplit(
        training_indices=np.flatnonzero(training_mask),
        validation_indices=np.flatnonzero(validation_mask),
        training_groups=training_groups,
        validation_groups=validation_groups,
    )


@dataclass(frozen=True)
class PairedClassifierDataset:
    """Positive and negative rows sharing one simulator-level group."""

    positive: np.ndarray
    negative: np.ndarray
    group_ids: np.ndarray
    shared_quantity: str
    split_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        positive = as_2d(self.positive, "positive")
        negative = as_2d(self.negative, "negative")
        groups = np.asarray(self.group_ids).reshape(-1)
        if positive.shape != negative.shape:
            raise ValueError(
                "positive and negative paired arrays must have equal shapes."
            )
        if len(groups) != len(positive):
            raise ValueError("group_ids must contain one identifier per pair.")
        if self.shared_quantity not in {"observation", "theta"}:
            raise ValueError("shared_quantity must be either 'observation' or 'theta'.")
        split_values = _validated_split_values(self.split_values, rows=len(positive))
        object.__setattr__(self, "positive", positive)
        object.__setattr__(self, "negative", negative)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "split_values", split_values)

    def subset(self, indices: np.ndarray) -> PairedClassifierDataset:
        """Select complete paired rows while retaining their scientific split."""

        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        return PairedClassifierDataset(
            positive=self.positive[indices],
            negative=self.negative[indices],
            group_ids=self.group_ids[indices],
            shared_quantity=self.shared_quantity,
            split_values=(
                None if self.split_values is None else self.split_values[indices]
            ),
        )

    def split_subset(
        self,
        label: str,
        *,
        required: bool = False,
    ) -> PairedClassifierDataset | None:
        """Return complete positive/negative pairs for one explicit split."""

        if label not in SCIENTIFIC_SPLIT_LABELS:
            raise ValueError(f"label must be one of {list(SCIENTIFIC_SPLIT_LABELS)}.")
        if self.split_values is None:
            result = self if label == "train" else None
        else:
            indices = np.flatnonzero(self.split_values == label)
            result = self.subset(indices) if len(indices) else None
        if required and result is None:
            raise ValueError(f"Paired data has no rows in required split {label!r}.")
        return result

    def stacked(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return values, balanced labels, and repeated group identifiers."""

        values = np.concatenate([self.positive, self.negative], axis=0)
        labels = np.concatenate(
            [
                np.ones(len(self.positive), dtype=np.float64),
                np.zeros(len(self.negative), dtype=np.float64),
            ]
        )
        groups = np.concatenate([self.group_ids, self.group_ids])
        return values, labels, groups

    def split(
        self,
        *,
        validation_fraction: float = 0.2,
        seed: int = 0,
    ) -> GroupSplit:
        """Return row indices for a leakage-safe split of the stacked rows."""

        _, _, stacked_groups = self.stacked()
        return group_train_validation_split(
            stacked_groups,
            validation_fraction=validation_fraction,
            seed=seed,
        )


def posterior_residual_pairs(
    simulator: ProposalDataset,
    denominator_theta: np.ndarray,
) -> PairedClassifierDataset:
    """Build matched-``x`` hNPE classifier rows.

    The positive rows are ``(theta_i, x_i)`` and negative rows are
    ``(theta_tilde_i, x_i)``.
    """

    denominator_theta = as_2d(denominator_theta, "denominator_theta")
    if denominator_theta.shape != simulator.theta.shape:
        raise ValueError(
            "denominator_theta must contain one theta row per simulator pair."
        )
    return PairedClassifierDataset(
        positive=np.column_stack([simulator.theta, simulator.observation]),
        negative=np.column_stack([denominator_theta, simulator.observation]),
        group_ids=simulator.simulation_ids,
        shared_quantity="observation",
        split_values=simulator.split_values,
    )


def likelihood_residual_pairs(
    simulator: ProposalDataset,
    reference_observation: np.ndarray,
) -> PairedClassifierDataset:
    """Build matched-``theta`` conditional-hNDE classifier rows."""

    reference_observation = as_2d(reference_observation, "reference_observation")
    if reference_observation.shape != simulator.observation.shape:
        raise ValueError(
            "reference_observation must contain one row per simulator pair."
        )
    return PairedClassifierDataset(
        positive=np.column_stack([simulator.theta, simulator.observation]),
        negative=np.column_stack([simulator.theta, reference_observation]),
        group_ids=simulator.simulation_ids,
        shared_quantity="theta",
        split_values=simulator.split_values,
    )


@dataclass(frozen=True)
class DualTrainingData:
    """The four proposal samples needed by the dual training construction."""

    rho_flow: ProposalDataset
    rho_ratio: ProposalDataset
    nu_flow: ProposalDataset
    kappa_ratio: ProposalDataset
    validation: ProposalDataset | None = None

    def __post_init__(self) -> None:
        datasets = [
            self.rho_flow,
            self.rho_ratio,
            self.nu_flow,
            self.kappa_ratio,
        ]
        if self.validation is not None:
            datasets.append(self.validation)
        theta_dimensions = {item.theta.shape[1] for item in datasets}
        observation_dimensions = {item.observation.shape[1] for item in datasets}
        if len(theta_dimensions) != 1 or len(observation_dimensions) != 1:
            raise ValueError(
                "All dual-training datasets must use common theta and "
                "observation dimensions."
            )
        parameter_names = {
            item.parameter_names for item in datasets if item.parameter_names
        }
        observation_names = {
            item.observation_names for item in datasets if item.observation_names
        }
        if len(parameter_names) > 1 or len(observation_names) > 1:
            raise ValueError("Named columns must have the same order in every dataset.")
