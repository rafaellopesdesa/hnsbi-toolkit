"""Backend-neutral orchestration of the five Bayesian training artifacts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ._array import as_2d, call_conditional_samples, call_samples
from .data import (
    DualTrainingData,
    PairedClassifierDataset,
    ProposalDataset,
    likelihood_residual_pairs,
    posterior_residual_pairs,
)
from .model import DualModel


@runtime_checkable
class BayesianTrainingBackend(Protocol):
    """Training hooks required by :class:`DualTrainer`.

    Implementations may use PyTorch, ONNX-exportable common flow classes, or
    any other backend. Ratio hooks must return log-ratio evaluators.
    """

    def train_conditional_density(
        self,
        target: np.ndarray,
        context: np.ndarray,
        *,
        artifact_name: str,
        group_ids: np.ndarray,
        validation: Any | None = None,
    ) -> Any:
        """Train and freeze one normalized conditional density."""

    def train_log_ratio(
        self,
        pairs: PairedClassifierDataset,
        *,
        artifact_name: str,
        validation: Any | None = None,
    ) -> Any:
        """Train and freeze one paired classifier log-ratio estimator."""

    def train_log_normalizer(
        self,
        q_eta: Any,
        r_c: Any,
        context: np.ndarray,
        *,
        artifact_name: str,
        validation: Any | None = None,
    ) -> Any:
        """Train ``theta -> log Z_C(theta)`` from frozen likelihood artifacts."""


@dataclass
class FunctionTrainingBackend:
    """Adapt three injected callables to the backend protocol."""

    conditional_density_trainer: Any
    log_ratio_trainer: Any
    log_normalizer_trainer: Any

    def train_conditional_density(self, *args: Any, **kwargs: Any) -> Any:
        return self.conditional_density_trainer(*args, **kwargs)

    def train_log_ratio(self, *args: Any, **kwargs: Any) -> Any:
        return self.log_ratio_trainer(*args, **kwargs)

    def train_log_normalizer(self, *args: Any, **kwargs: Any) -> Any:
        return self.log_normalizer_trainer(*args, **kwargs)


@dataclass
class LazyTrainingBackend:
    """Resolve backend callables only when training is requested.

    This keeps importing ``hnsbi.bayes`` independent of Torch, nflows, and
    ONNX. The named callables must obey :class:`BayesianTrainingBackend`.
    """

    module: str
    conditional_density_name: str
    log_ratio_name: str
    log_normalizer_name: str
    _resolved: FunctionTrainingBackend | None = field(
        default=None, init=False, repr=False
    )

    def _backend(self) -> FunctionTrainingBackend:
        if self._resolved is None:
            module = importlib.import_module(self.module)
            self._resolved = FunctionTrainingBackend(
                conditional_density_trainer=getattr(
                    module, self.conditional_density_name
                ),
                log_ratio_trainer=getattr(module, self.log_ratio_name),
                log_normalizer_trainer=getattr(module, self.log_normalizer_name),
            )
        return self._resolved

    def train_conditional_density(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend().train_conditional_density(*args, **kwargs)

    def train_log_ratio(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend().train_log_ratio(*args, **kwargs)

    def train_log_normalizer(self, *args: Any, **kwargs: Any) -> Any:
        return self._backend().train_log_normalizer(*args, **kwargs)


@dataclass
class DualTrainer:
    """Train the dual artifacts in the statistically required order."""

    backend: BayesianTrainingBackend
    seed: int = 0

    @staticmethod
    def _with_split(
        dataset: ProposalDataset,
        label: str,
    ) -> ProposalDataset:
        return ProposalDataset(
            theta=dataset.theta,
            observation=dataset.observation,
            simulation_ids=dataset.simulation_ids,
            design=dataset.design,
            parameter_names=dataset.parameter_names,
            observation_names=dataset.observation_names,
            split_values=np.full(len(dataset.theta), label),
            log_density=dataset.log_density,
            metadata=dataset.metadata,
        )

    @staticmethod
    def _combine_evaluation(
        parts: list[tuple[str, ProposalDataset]],
        *,
        artifact_name: str,
    ) -> ProposalDataset | None:
        if not parts:
            return None
        first = parts[0][1]
        for _, item in parts[1:]:
            if (
                item.parameter_names != first.parameter_names
                or item.observation_names != first.observation_names
            ):
                raise ValueError(
                    f"{artifact_name} evaluation datasets use inconsistent columns."
                )
        log_density = (
            np.concatenate([item.log_density for _, item in parts])
            if all(item.log_density is not None for _, item in parts)
            else None
        )
        group_ids = np.concatenate(
            [
                np.asarray(
                    [f"{origin}:{index}:{value}" for value in item.simulation_ids],
                    dtype=object,
                )
                for index, (origin, item) in enumerate(parts)
            ]
        )
        return ProposalDataset(
            theta=np.concatenate([item.theta for _, item in parts]),
            observation=np.concatenate([item.observation for _, item in parts]),
            simulation_ids=group_ids,
            design=f"{artifact_name}-evaluation",
            parameter_names=first.parameter_names,
            observation_names=first.observation_names,
            split_values=np.concatenate([item.split_values for _, item in parts]),
            log_density=log_density,
            metadata={
                "artifact_name": artifact_name,
                "sources": [
                    {
                        "design": item.design,
                        "origin": origin,
                        "rows": len(item.theta),
                    }
                    for origin, item in parts
                ],
            },
        )

    def _stage_data(
        self,
        source: ProposalDataset,
        independent: ProposalDataset | None,
        *,
        artifact_name: str,
    ) -> tuple[ProposalDataset, ProposalDataset | None]:
        """Resolve configured labels before any model or pair construction.

        Source ``train`` rows are the only rows returned for fitting. Source
        ``validation`` and ``holdout`` rows retain their declared roles.
        An unlabeled independent validation dataset supplies validation rows
        unless the source already declares them, in which case it supplies a
        genuinely untouched holdout. A labeled independent dataset may contain
        only validation/holdout rows.
        """

        training = source.split_subset("train", required=True)
        assert training is not None
        parts: list[tuple[str, ProposalDataset]] = []
        for label in ("validation", "holdout"):
            subset = source.split_subset(label)
            if subset is not None:
                parts.append((f"source-{label}", self._with_split(subset, label)))

        if independent is not None:
            if independent.split_values is None:
                independent_label = (
                    "holdout"
                    if any(
                        np.all(item.split_values == "validation") for _, item in parts
                    )
                    else "validation"
                )
                parts.append(
                    (
                        f"independent-{independent_label}",
                        self._with_split(independent, independent_label),
                    )
                )
            else:
                if independent.split_subset("train") is not None:
                    raise ValueError(
                        "bayesian.datasets.validation may contain only "
                        "'validation' and 'holdout' split labels."
                    )
                for label in ("validation", "holdout"):
                    subset = independent.split_subset(label)
                    if subset is not None:
                        parts.append(
                            (
                                f"independent-{label}",
                                self._with_split(subset, label),
                            )
                        )
        return training, self._combine_evaluation(parts, artifact_name=artifact_name)

    def _posterior_negative(
        self,
        q_phi: Any,
        rho: Any,
        observation: np.ndarray,
        *,
        epsilon: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        flow = call_conditional_samples(q_phi, 1, context=observation, rng=rng)[:, 0, :]
        if epsilon == 0.0:
            return flow
        design = call_samples(rho, len(observation), rng=rng)
        use_design = rng.random(len(observation)) < epsilon
        return np.where(use_design[:, None], design, flow)

    def fit(
        self,
        data: DualTrainingData,
        *,
        rho: Any,
        defensive_epsilon: float = 0.0,
        normalizer_context: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DualModel:
        """Train flows first, then their exact paired residual classifiers."""

        epsilon = float(defensive_epsilon)
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("defensive_epsilon must lie in [0, 1).")
        if not hasattr(rho, "sample") or not hasattr(rho, "log_prob"):
            raise TypeError("rho must provide normalized sample() and log_prob().")
        rng = np.random.default_rng(int(self.seed))
        rho_flow, q_phi_validation = self._stage_data(
            data.rho_flow, data.validation, artifact_name="q_phi"
        )

        q_phi = self.backend.train_conditional_density(
            rho_flow.theta,
            rho_flow.observation,
            artifact_name="q_phi",
            group_ids=rho_flow.simulation_ids,
            validation=q_phi_validation,
        )
        rho_ratio, r_p_validation_data = self._stage_data(
            data.rho_ratio, data.validation, artifact_name="r_p"
        )
        denominator_theta = self._posterior_negative(
            q_phi,
            rho,
            rho_ratio.observation,
            epsilon=epsilon,
            rng=rng,
        )
        posterior_pairs = posterior_residual_pairs(rho_ratio, denominator_theta)
        posterior_validation_pairs = None
        if r_p_validation_data is not None:
            validation_theta = self._posterior_negative(
                q_phi,
                rho,
                r_p_validation_data.observation,
                epsilon=epsilon,
                rng=rng,
            )
            posterior_validation_pairs = posterior_residual_pairs(
                r_p_validation_data, validation_theta
            )
        r_p = self.backend.train_log_ratio(
            posterior_pairs,
            artifact_name="r_p",
            validation=posterior_validation_pairs,
        )

        nu_flow, q_eta_validation = self._stage_data(
            data.nu_flow, data.validation, artifact_name="q_eta"
        )
        q_eta = self.backend.train_conditional_density(
            nu_flow.observation,
            nu_flow.theta,
            artifact_name="q_eta",
            group_ids=nu_flow.simulation_ids,
            validation=q_eta_validation,
        )
        kappa_ratio, r_c_validation_data = self._stage_data(
            data.kappa_ratio, data.validation, artifact_name="r_c"
        )
        reference_observation = call_conditional_samples(
            q_eta,
            1,
            context=kappa_ratio.theta,
            rng=rng,
        )[:, 0, :]
        likelihood_pairs = likelihood_residual_pairs(kappa_ratio, reference_observation)
        likelihood_validation_pairs = None
        if r_c_validation_data is not None:
            validation_reference = call_conditional_samples(
                q_eta,
                1,
                context=r_c_validation_data.theta,
                rng=rng,
            )[:, 0, :]
            likelihood_validation_pairs = likelihood_residual_pairs(
                r_c_validation_data, validation_reference
            )
        r_c = self.backend.train_log_ratio(
            likelihood_pairs,
            artifact_name="r_c",
            validation=likelihood_validation_pairs,
        )

        if normalizer_context is None:
            normalizer_context = kappa_ratio.theta
        normalizer_context = as_2d(normalizer_context, "normalizer_context")
        z_c = self.backend.train_log_normalizer(
            q_eta,
            r_c,
            normalizer_context,
            artifact_name="z_c",
            validation=r_c_validation_data,
        )
        return DualModel(
            q_phi=q_phi,
            r_p=r_p,
            q_eta=q_eta,
            r_c=r_c,
            z_c=z_c,
            rho=rho,
            posterior_ratio_reference=("defensive" if epsilon > 0.0 else "flow"),
            defensive_epsilon=epsilon,
            metadata={} if metadata is None else metadata,
        )
