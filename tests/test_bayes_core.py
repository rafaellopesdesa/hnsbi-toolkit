from __future__ import annotations

import math

import numpy as np
import pytest

from hnsbi.bayes import (
    DualModel,
    DualTrainer,
    DualTrainingData,
    FunctionTrainingBackend,
    ProposalDataset,
    WeightedSamples,
    bridge_diagnostic,
    conditional_normalization_diagnostic,
    estimate_evidence,
    geometric_consensus,
    group_train_validation_split,
    hnde_log_weights,
    hnpe_log_weights,
    posterior_normalization_diagnostic,
    posterior_predictive,
    posterior_residual_pairs,
    prior_auxiliary_log_update,
    route_diagnostic,
    sample_posterior,
    selection_integral,
    update_posterior_weights,
)

LOG_2PI = math.log(2.0 * math.pi)


def normal_logpdf(values, mean, sigma):
    values = np.asarray(values, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    return -0.5 * (((values - mean) / sigma) ** 2 + LOG_2PI + 2.0 * math.log(sigma))


class NormalDistribution:
    def __init__(self, mean=0.0, sigma=1.0):
        self.mean = float(mean)
        self.sigma = float(sigma)

    def sample(self, n, *, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        return rng.normal(self.mean, self.sigma, size=(int(n), 1))

    def log_prob(self, values):
        values = np.asarray(values, dtype=np.float64)
        return normal_logpdf(values[:, 0], self.mean, self.sigma)


class GaussianConditional:
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = float(sigma)
        self.sample_calls = 0

    def _mean(self, context):
        context = np.asarray(context, dtype=np.float64)
        result = np.asarray(self.mean(context), dtype=np.float64)
        if result.ndim == 1:
            result = result[:, None]
        return result

    def sample(self, n, *, context, rng=None):
        self.sample_calls += 1
        rng = np.random.default_rng() if rng is None else rng
        mean = self._mean(context)
        return mean[:, None, :] + rng.normal(
            0.0, self.sigma, size=(len(mean), int(n), mean.shape[1])
        )

    def log_prob(self, values, *, context):
        values = np.asarray(values, dtype=np.float64)
        mean = self._mean(context)
        return np.sum(normal_logpdf(values, mean, self.sigma), axis=1)


def zero_log_ratio(target, context):
    return np.zeros(len(np.asarray(target)), dtype=np.float64)


def zero_log_normalizer(theta):
    return np.zeros(len(np.asarray(theta)), dtype=np.float64)


def exact_gaussian_model():
    rho = NormalDistribution()
    q_phi = GaussianConditional(lambda x: 0.5 * x, math.sqrt(0.5))
    q_eta = GaussianConditional(lambda theta: theta, 1.0)
    return DualModel(
        q_phi=q_phi,
        r_p=zero_log_ratio,
        q_eta=q_eta,
        r_c=zero_log_ratio,
        z_c=zero_log_normalizer,
        rho=rho,
    )


def proposal_dataset(design, offset=0, n=20):
    theta = np.linspace(-1.0, 1.0, n)[:, None]
    observation = (theta**2 + 0.1 * offset).astype(np.float64)
    return ProposalDataset(
        theta=theta,
        observation=observation,
        simulation_ids=np.arange(offset, offset + n, dtype=np.int64),
        design=design,
        parameter_names=("theta",),
        observation_names=("x",),
    )


def test_proposal_schema_and_group_split_reject_leakage():
    data = proposal_dataset("rho")
    negative = data.theta + 0.25
    pairs = posterior_residual_pairs(data, negative)
    values, labels, groups = pairs.stacked()
    assert values.shape == (40, 2)
    assert labels.sum() == 20
    assert np.array_equal(values[:20, 1], values[20:, 1])

    split = pairs.split(validation_fraction=0.25, seed=4)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        in_training = np.isin(rows, split.training_indices)
        in_validation = np.isin(rows, split.validation_indices)
        assert in_training.all() or in_validation.all()
        assert not (in_training.any() and in_validation.any())

    direct = group_train_validation_split(
        np.repeat(np.arange(10), 3), validation_fraction=0.3, seed=9
    )
    assert not np.intersect1d(direct.training_groups, direct.validation_groups).size


def test_proposal_schema_validates_ids_names_and_finiteness():
    with pytest.raises(ValueError, match="unique"):
        ProposalDataset(
            theta=np.zeros((2, 1)),
            observation=np.zeros((2, 1)),
            simulation_ids=np.array([1, 1]),
            design="rho",
        )
    with pytest.raises(ValueError, match="parameter_names"):
        ProposalDataset(
            theta=np.zeros((2, 2)),
            observation=np.zeros((2, 1)),
            simulation_ids=np.array([1, 2]),
            design="rho",
            parameter_names=("only_one",),
        )
    with pytest.raises(ValueError, match="non-finite"):
        ProposalDataset(
            theta=np.array([[0.0], [np.nan]]),
            observation=np.zeros((2, 1)),
            simulation_ids=np.array([1, 2]),
            design="rho",
        )
    with pytest.raises(ValueError, match="accepts exactly"):
        ProposalDataset(
            theta=np.zeros((2, 1)),
            observation=np.zeros((2, 1)),
            simulation_ids=np.array([1, 2]),
            design="rho",
            split_values=np.asarray(["train", "test"]),
        )


def test_exact_routes_bridge_and_ess_agree():
    model = exact_gaussian_model()
    x = np.array([[0.7]])
    theta = model.sample_posterior_denominator(
        4_000, observation=x, rng=np.random.default_rng(3)
    )
    log_p = hnpe_log_weights(model, theta, x)
    log_l = hnde_log_weights(model, theta, x)
    assert np.ptp(log_p) == pytest.approx(0.0, abs=1.0e-14)
    assert np.ptp(log_l) == pytest.approx(0.0, abs=2.0e-14)

    diagnostics = route_diagnostic(model, theta, x)
    assert diagnostics.log_weight_rms < 2.0e-14
    assert diagnostics.hnpe["ESS"] == pytest.approx(len(theta))
    assert diagnostics.hnde["ESS"] == pytest.approx(len(theta))

    log_evidence = float(normal_logpdf(x[0, 0], 0.0, math.sqrt(2.0)))
    bridge = bridge_diagnostic(model, theta, x, log_design_evidence=log_evidence)
    assert bridge.rms_residual < 2.0e-14
    assert bridge.median_absolute_residual < 2.0e-14


def test_geometric_consensus_is_route_normalized_and_constant_invariant():
    log_p = np.log(np.array([1.0, 2.0, 7.0]))
    log_l = np.log(np.array([4.0, 5.0, 1.0]))
    result = geometric_consensus(log_p, log_l)
    expected = np.sqrt(
        np.exp(log_p) / np.exp(log_p).sum() * np.exp(log_l) / np.exp(log_l).sum()
    )
    expected /= expected.sum()
    assert np.allclose(result, expected)
    assert np.allclose(geometric_consensus(log_p + 100.0, log_l - 47.0), expected)
    assert np.allclose(
        geometric_consensus(log_p, log_p),
        np.exp(log_p) / np.exp(log_p).sum(),
    )


def test_mandatory_parameter_dependent_zc_removes_artificial_preference():
    rho = NormalDistribution()
    q_phi = GaussianConditional(lambda x: np.zeros_like(x), 1.0)
    q_eta = GaussianConditional(lambda theta: theta, 1.0)

    def raw_log_ratio(observation, theta):
        return np.asarray(theta)[:, 0]

    model = DualModel(
        q_phi=q_phi,
        r_p=zero_log_ratio,
        q_eta=q_eta,
        r_c=raw_log_ratio,
        z_c=lambda theta: np.asarray(theta)[:, 0],
        rho=rho,
    )
    wrong = DualModel(
        q_phi=q_phi,
        r_p=zero_log_ratio,
        q_eta=q_eta,
        r_c=raw_log_ratio,
        z_c=zero_log_normalizer,
        rho=rho,
    )
    theta = np.array([[-1.0], [0.0], [1.0]])
    x = theta.copy()
    assert np.allclose(model.log_likelihood(x, theta), q_eta.log_prob(x, context=theta))
    assert np.allclose(
        wrong.log_likelihood(x, theta) - model.log_likelihood(x, theta),
        theta[:, 0],
    )

    normalization = conditional_normalization_diagnostic(
        model, theta, n_reference=2_000, rng=np.random.default_rng(7)
    )
    assert np.allclose(normalization.raw_z, np.exp(theta[:, 0]))
    assert np.allclose(normalization.modeled_z, np.exp(theta[:, 0]))
    assert np.allclose(normalization.corrected_z, 1.0)
    assert np.allclose(normalization.corrected_ess, 2_000)


def test_defensive_mixture_density_sampling_and_ratio_accounting():
    epsilon = 0.3
    rho = NormalDistribution(mean=5.0, sigma=0.5)
    q_phi = GaussianConditional(lambda x: np.full((len(x), 1), -5.0), 0.5)
    q_eta = GaussianConditional(lambda theta: theta, 1.0)
    model = DualModel(
        q_phi=q_phi,
        r_p=zero_log_ratio,
        q_eta=q_eta,
        r_c=zero_log_ratio,
        z_c=zero_log_normalizer,
        rho=rho,
        posterior_ratio_reference="defensive",
        defensive_epsilon=epsilon,
    )
    x = np.array([[0.0]])
    draws = model.sample_posterior_denominator(
        30_000, observation=x, rng=np.random.default_rng(2)
    )
    assert np.mean(draws[:, 0] > 0.0) == pytest.approx(epsilon, abs=0.015)

    theta = np.array([[-5.0], [5.0], [0.0]])
    log_flow = q_phi.log_prob(theta, context=np.repeat(x, 3, axis=0))
    log_design = rho.log_prob(theta)
    expected = np.logaddexp(
        np.log1p(-epsilon) + log_flow,
        np.log(epsilon) + log_design,
    )
    assert np.allclose(model.posterior_denominator_log_prob(theta, x), expected)
    assert np.allclose(hnpe_log_weights(model, theta, x), 0.0)

    norm = posterior_normalization_diagnostic(
        model, x, n_reference=2_000, rng=np.random.default_rng(5)
    )
    assert norm.value == pytest.approx(1.0)
    assert norm.ess == pytest.approx(2_000)


def test_prior_auxiliary_update_uses_f_over_f0():
    theta = np.array([[-1.0], [0.0], [1.0]])
    rho = NormalDistribution()
    shifted = NormalDistribution(mean=0.5, sigma=0.8)

    def auxiliary(values):
        return normal_logpdf(0.2, values[:, 0], 0.4)

    update = prior_auxiliary_log_update(
        theta,
        analysis_log_prior=shifted,
        design_log_prior=rho,
        auxiliary_log_likelihood=auxiliary,
    )
    expected = shifted.log_prob(theta) - rho.log_prob(theta) + auxiliary(theta)
    assert np.allclose(update, expected)

    cancelled = update_posterior_weights(
        np.zeros(len(theta)),
        theta,
        analysis_log_prior=rho,
        design_log_prior=rho,
        auxiliary_log_likelihood=auxiliary,
        baseline_auxiliary_log_likelihood=auxiliary,
    )
    assert np.allclose(cancelled, 1.0 / len(theta))


def test_absolute_evidence_matches_analytic_gaussian_result():
    model = exact_gaussian_model()
    rng = np.random.default_rng(11)
    theta = model.rho.sample(120_000, rng=rng)
    x = np.array([[0.7]])
    estimate = estimate_evidence(
        model,
        x,
        theta,
        integration_log_prob=model.rho,
        analysis_log_prior=model.rho,
    )
    truth = math.exp(float(normal_logpdf(0.7, 0.0, math.sqrt(2.0))))
    assert estimate.evidence == pytest.approx(truth, rel=0.012)
    assert estimate.relative_mc_error < 0.01
    assert estimate.ess > 50_000


def test_simulator_free_posterior_predictive_has_expected_moments():
    model = exact_gaussian_model()
    x = np.array([[0.8]])
    posterior = sample_posterior(
        model,
        x,
        n=45_000,
        route="dual",
        rng=np.random.default_rng(13),
    )
    calls_before = model.q_eta.sample_calls
    predictive = posterior_predictive(
        model, posterior, n=45_000, rng=np.random.default_rng(14)
    )
    assert model.q_eta.sample_calls == calls_before + 1
    mean = float(np.sum(predictive.weights * predictive.observation[:, 0]))
    variance = float(
        np.sum(predictive.weights * (predictive.observation[:, 0] - mean) ** 2)
    )
    assert mean == pytest.approx(0.4, abs=0.025)
    assert variance == pytest.approx(1.5, abs=0.04)
    assert predictive.diagnostics["ESS_fraction"] == pytest.approx(1.0)


def test_selection_integral_matches_gaussian_tail():
    model = exact_gaussian_model()
    theta = np.array([[-1.0], [0.0], [1.0]])
    result = selection_integral(
        model,
        theta,
        lambda observation: observation[:, 0] > 0.0,
        n_reference=50_000,
        rng=np.random.default_rng(17),
    )
    truth = np.array(
        [0.5 * (1.0 + math.erf(value / math.sqrt(2.0))) for value in theta[:, 0]]
    )
    assert np.allclose(result.estimate, truth, atol=0.008)
    assert np.allclose(result.self_normalized_estimate, truth, atol=0.008)
    assert np.allclose(result.reference_normalization, 1.0)
    assert np.allclose(result.ess, 50_000)


class CapturingBackend:
    def __init__(self):
        self.flows = {}
        self.ratios = {}
        self.normalizer_context = None
        self.normalizer_validation = None

    def train_conditional_density(
        self,
        target,
        context,
        *,
        artifact_name,
        group_ids,
        validation=None,
    ):
        self.flows[artifact_name] = (
            np.asarray(target),
            np.asarray(context),
            np.asarray(group_ids),
            validation,
        )
        target_dim = np.asarray(target).shape[1]
        return GaussianConditional(lambda rows: np.zeros((len(rows), target_dim)), 1.0)

    def train_log_ratio(self, pairs, *, artifact_name, validation=None):
        self.ratios[artifact_name] = (pairs, validation)
        return zero_log_ratio

    def train_log_normalizer(
        self,
        q_eta,
        r_c,
        context,
        *,
        artifact_name,
        validation=None,
    ):
        assert artifact_name == "z_c"
        self.normalizer_context = np.asarray(context)
        self.normalizer_validation = validation
        return zero_log_normalizer


def test_dual_trainer_builds_matched_pairs_and_five_artifacts():
    rho_flow = proposal_dataset("rho", offset=0)
    rho_ratio = proposal_dataset("rho", offset=100)
    nu_flow = proposal_dataset("nu", offset=200)
    kappa_ratio = proposal_dataset("kappa", offset=300)
    validation = proposal_dataset("validation", offset=400)
    data = DualTrainingData(
        rho_flow=rho_flow,
        rho_ratio=rho_ratio,
        nu_flow=nu_flow,
        kappa_ratio=kappa_ratio,
        validation=validation,
    )
    capture = CapturingBackend()
    trainer = DualTrainer(
        FunctionTrainingBackend(
            capture.train_conditional_density,
            capture.train_log_ratio,
            capture.train_log_normalizer,
        ),
        seed=21,
    )
    model = trainer.fit(
        data,
        rho=NormalDistribution(),
        defensive_epsilon=0.2,
        normalizer_context=np.array([[-1.0], [0.0], [1.0]]),
    )
    assert set(model.artifacts) == {"q_phi", "r_p", "q_eta", "r_c", "z_c"}
    assert model.posterior_ratio_reference == "defensive"
    assert model.defensive_epsilon == pytest.approx(0.2)

    posterior_pairs = capture.ratios["r_p"][0]
    likelihood_pairs = capture.ratios["r_c"][0]
    assert posterior_pairs.shared_quantity == "observation"
    assert likelihood_pairs.shared_quantity == "theta"
    assert np.array_equal(
        posterior_pairs.positive[:, 1:], posterior_pairs.negative[:, 1:]
    )
    assert np.array_equal(
        likelihood_pairs.positive[:, :1], likelihood_pairs.negative[:, :1]
    )
    assert np.array_equal(posterior_pairs.group_ids, rho_ratio.simulation_ids)
    assert np.array_equal(likelihood_pairs.group_ids, kappa_ratio.simulation_ids)
    assert np.array_equal(capture.normalizer_context, np.array([[-1.0], [0.0], [1.0]]))
    for artifact_name in ("r_p", "r_c"):
        validation_pairs = capture.ratios[artifact_name][1]
        assert validation_pairs is not None
        assert set(validation_pairs.split_values) == {"validation"}
    assert capture.normalizer_validation is not None
    assert set(capture.normalizer_validation.split_values) == {"validation"}


def test_dual_trainer_honors_scientific_splits_without_group_leakage():
    labels = np.asarray(["train"] * 12 + ["validation"] * 4 + ["holdout"] * 4)

    def split_dataset(design, offset):
        source = proposal_dataset(design, offset=offset)
        return ProposalDataset(
            theta=source.theta,
            observation=source.observation,
            simulation_ids=source.simulation_ids,
            design=source.design,
            parameter_names=source.parameter_names,
            observation_names=source.observation_names,
            split_values=labels,
        )

    rho_flow = split_dataset("rho-flow", 0)
    rho_ratio = split_dataset("rho-ratio", 100)
    nu_flow = split_dataset("nu-flow", 200)
    kappa_ratio = split_dataset("kappa-ratio", 300)
    capture = CapturingBackend()
    DualTrainer(
        FunctionTrainingBackend(
            capture.train_conditional_density,
            capture.train_log_ratio,
            capture.train_log_normalizer,
        ),
        seed=7,
    ).fit(
        DualTrainingData(
            rho_flow=rho_flow,
            rho_ratio=rho_ratio,
            nu_flow=nu_flow,
            kappa_ratio=kappa_ratio,
        ),
        rho=NormalDistribution(),
    )

    np.testing.assert_array_equal(capture.flows["q_phi"][2], np.arange(0, 12))
    np.testing.assert_array_equal(capture.flows["q_eta"][2], np.arange(200, 212))
    np.testing.assert_array_equal(
        capture.ratios["r_p"][0].group_ids, np.arange(100, 112)
    )
    np.testing.assert_array_equal(
        capture.ratios["r_c"][0].group_ids, np.arange(300, 312)
    )
    np.testing.assert_array_equal(capture.normalizer_context, kappa_ratio.theta[:12])

    for artifact_name in ("q_phi", "q_eta"):
        evaluation = capture.flows[artifact_name][3]
        assert evaluation is not None
        assert set(evaluation.split_values) == {"validation", "holdout"}
        assert not set(capture.flows[artifact_name][2]).intersection(
            set(evaluation.simulation_ids)
        )
    for artifact_name in ("r_p", "r_c"):
        training_pairs, evaluation_pairs = capture.ratios[artifact_name]
        assert evaluation_pairs is not None
        assert set(evaluation_pairs.split_values) == {"validation", "holdout"}
        assert not set(training_pairs.group_ids).intersection(
            set(evaluation_pairs.group_ids)
        )
        for group in evaluation_pairs.group_ids:
            rows = np.flatnonzero(evaluation_pairs.stacked()[2] == group)
            assert len(rows) == 2


def test_independent_validation_fills_missing_holdout_and_rejects_train_rows():
    source = proposal_dataset("rho", offset=0)
    source = ProposalDataset(
        theta=source.theta,
        observation=source.observation,
        simulation_ids=source.simulation_ids,
        design=source.design,
        parameter_names=source.parameter_names,
        observation_names=source.observation_names,
        split_values=np.asarray(["train"] * 12 + ["validation"] * 8),
    )
    independent = proposal_dataset("independent", offset=100)
    trainer = DualTrainer(CapturingBackend())
    training, evaluation = trainer._stage_data(
        source, independent, artifact_name="q_phi"
    )
    assert len(training.theta) == 12
    assert evaluation is not None
    assert np.count_nonzero(evaluation.split_values == "validation") == 8
    assert np.count_nonzero(evaluation.split_values == "holdout") == 20

    invalid = ProposalDataset(
        theta=independent.theta,
        observation=independent.observation,
        simulation_ids=independent.simulation_ids,
        design=independent.design,
        parameter_names=independent.parameter_names,
        observation_names=independent.observation_names,
        split_values=np.asarray(["train"] * 20),
    )
    with pytest.raises(ValueError, match="may contain only"):
        trainer._stage_data(source, invalid, artifact_name="q_phi")


def test_weighted_samples_require_normalization():
    with pytest.raises(ValueError, match="normalized"):
        WeightedSamples(
            values=np.zeros((2, 1)),
            weights=np.array([1.0, 1.0]),
            log_weights=np.zeros(2),
        )
