from __future__ import annotations

import json

import numpy as np
import pytest

from hnsbi.asimov import AsimovBuilder
from hnsbi.data import DataSource
from hnsbi.expressions import Expression, ExpressionError
from hnsbi.intensity import Component, IntensityModel, Parameter, RatioNormalizer
from hnsbi.nis import DefensiveMixture, NISAsimovBuilder
from hnsbi.systematics import SystematicAnchor, SystematicsTrainer
from hnsbi.toys import ToyGenerator
from hnsbi.workspace import load_workspace_model, write_nsbi_workspace


class NormalSampler:
    def __init__(self, mean: float = 0.0) -> None:
        self.mean = mean

    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return rng.normal(self.mean, 1.0, size=(n, 1))


class StandardNormalDensity:
    def log_prob(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values)[:, 0]
        return -0.5 * x**2 - 0.5 * np.log(2.0 * np.pi)


def _model() -> IntensityModel:
    return IntensityModel(
        [
            Component("background", 20.0, "1"),
            Component("signal", 10.0, "mu"),
        ],
        [Parameter("mu", 1.0, (0.0, 5.0))],
    )


def _ratios():
    return {
        "background": lambda values: np.ones(len(values)),
        "signal": lambda values: np.exp(0.4 * values[:, 0] - 0.08),
    }


def test_safe_expressions_and_intensity_formulas() -> None:
    expression = Expression.parse("mu * exp(0.5 * alpha)")
    assert expression.evaluate({"mu": 2.0, "alpha": 0.0}) == pytest.approx(2.0)
    assert Expression.parse("mu * beta").simple_normfactors() == ("mu", "beta")
    assert Expression.parse("mu * mu").simple_normfactors() is None
    assert Expression.parse("mu + beta").simple_normfactors() is None
    with pytest.raises(ExpressionError):
        Expression.parse("__import__('os').system('id')")

    model = IntensityModel(
        [Component("signal", 4.0, "mu * exp(alpha)")],
        [
            Parameter("mu", 1.0, (0.0, 4.0)),
            Parameter("alpha", 0.0, (-2.0, 2.0)),
        ],
    )
    assert model.expected_yield({"mu": 2.0, "alpha": 0.0}) == pytest.approx(8.0)


def test_asimov_same_support_normalization_closes_yield_and_score() -> None:
    result = AsimovBuilder(
        reference=NormalSampler(),
        ratios=_ratios(),
        intensity=_model(),
        features=["x"],
    ).build({"mu": 1.7}, n_events=8192, seed=18)

    assert result.raw_count == 8192
    assert result.events.expected_count == pytest.approx(37.0, abs=1.0e-12)
    for ratio in result.normalized_ratios.values():
        assert np.average(ratio, weights=result.reference_weights) == pytest.approx(
            1.0, abs=1.0e-12
        )
    # d/dmu integral lambda(x;mu) dx closes to the nominal signal yield.
    signal_score_integral = np.sum(
        result.reference_weights * 10.0 * result.normalized_ratios["signal"]
    )
    assert signal_score_integral == pytest.approx(10.0, abs=1.0e-12)
    assert 0.0 < result.ess <= result.raw_count


def test_defensive_mixture_bound_and_nis_closure() -> None:
    # Identical q and g make q / [(1-eps)g+eps q] exactly one.
    mixture = DefensiveMixture(
        reference=NormalSampler(),
        reference_density=StandardNormalDensity(),
        proposal=NormalSampler(),
        proposal_density=StandardNormalDensity(),
        epsilon=0.1,
    )
    values, log_weights = mixture.sample_with_reference_log_weight(
        4096, rng=np.random.default_rng(4)
    )
    assert values.shape == (4096, 1)
    np.testing.assert_allclose(log_weights, 0.0, atol=1.0e-12)
    assert np.max(np.exp(log_weights)) <= 10.0

    result = NISAsimovBuilder(
        proposal=mixture,
        ratios=_ratios(),
        intensity=_model(),
        features=["x"],
    ).build({"mu": 0.8}, n_events=4096, seed=5)
    assert result.events.expected_count == pytest.approx(28.0, abs=1.0e-12)
    assert result.events.metadata["epsilon"] == pytest.approx(0.1)


def test_toys_use_componentwise_poisson_counts() -> None:
    generator = ToyGenerator(
        intensity=_model(),
        features=["x"],
        component_samplers={
            "background": NormalSampler(-1.0),
            "signal": NormalSampler(1.0),
        },
    )
    toys = generator.generate_many({"mu": 1.5}, n_toys=2000, seed=27)
    totals = np.asarray([toy.observed_count for toy in toys])
    assert np.mean(totals) == pytest.approx(35.0, abs=0.5)
    assert np.var(totals) == pytest.approx(35.0, abs=2.5)
    assert all(np.all(toy.events.weights == 1.0) for toy in toys)


def test_systematic_shape_is_normalized_on_active_support() -> None:
    anchor = SystematicAnchor(
        parameter="jes",
        component="signal",
        ratio_up=np.asarray([0.8, 1.2, 1.4]),
        ratio_down=np.asarray([1.3, 0.9, 0.7]),
        yield_up=1.1,
        yield_down=0.9,
    )
    nominal = np.asarray([0.5, 1.0, 1.5])
    weights = np.asarray([0.2, 0.3, 0.5])
    shape = anchor.normalized_shape(
        0.7, nominal_process_ratio=nominal, integration_weights=weights
    )
    assert np.sum(weights * nominal * shape) == pytest.approx(1.0)
    assert anchor.yield_factor(0.5) == pytest.approx(1.05)
    assert anchor.yield_factor(-0.5) == pytest.approx(0.95)


def test_systematics_trainer_uses_ratio_trainer_contract(tmp_path) -> None:
    class RecordingTrainer:
        def __init__(self) -> None:
            self.calls = []

        def fit(self, numerator, denominator, **kwargs):
            self.calls.append((numerator, denominator, kwargs))
            return kwargs["numerator_name"]

    backend = RecordingTrainer()
    result = SystematicsTrainer(backend).fit_variation(
        nominal=np.zeros((4, 1)),
        up=np.ones((4, 1)),
        down=-np.ones((4, 1)),
        parameter="jes",
        component="signal",
        output_dir=tmp_path,
        features=["x"],
    )
    assert result == {
        "up": "signal_jes_up",
        "down": "signal_jes_down",
    }
    assert backend.calls[0][2]["output_directory"] == tmp_path / "up"
    assert backend.calls[1][2]["denominator_name"] == "signal_nominal"


def test_numpy_data_source_batches_and_limits() -> None:
    values = np.arange(30, dtype=np.float64).reshape(10, 3)
    source = DataSource(values, features=["a", "b", "c"], weight=None)
    batches = list(source.iter_batches(batch_size=4))
    assert [len(batch.values) for batch in batches] == [4, 4, 2]
    materialized = source.materialize(batch_size=3, max_events=7)
    assert materialized.values.shape == (7, 3)
    np.testing.assert_array_equal(materialized.row_ids, np.arange(7))


def test_workspace_export_marks_formula_extension(tmp_path) -> None:
    model = IntensityModel(
        [
            Component("background", 20.0, "1"),
            Component("signal", 10.0, "mu * exp(alpha)"),
        ],
        [
            Parameter("mu", 1.0, (0.0, 5.0)),
            Parameter("alpha", 0.0, (-2.0, 2.0)),
        ],
    )
    ratios = {
        "background": lambda values: np.ones(len(values)),
        "signal": lambda values: np.exp(0.4 * values[:, 0] - 0.08),
    }
    result = AsimovBuilder(
        reference=NormalSampler(),
        ratios=ratios,
        intensity=model,
        features=["x"],
    ).build({"mu": 1.0, "alpha": 0.0}, n_events=512, seed=2)
    export = write_nsbi_workspace(
        result=result,
        intensity=model,
        output_dir=tmp_path,
        measurement="measurement",
        poi="mu",
    )
    assert export.schema_version == "2.0"
    workspace = json.loads(export.path.read_text())
    assert workspace["hnsbi"]["sample_multipliers"]["signal"] == "mu * exp(alpha)"
    parameters = workspace["measurements"][0]["config"]["parameters"]
    assert {item["name"]: item["initial"] for item in parameters} == {
        "mu": 1.0,
        "alpha": 0.0,
    }
    assert (tmp_path / "arrays" / "asimov_weights.npy").exists()
    recovered = load_workspace_model(export.path)
    assert recovered.features == ("x",)
    assert recovered.intensity.expected_yield(
        {"mu": 1.0, "alpha": 0.0}
    ) == pytest.approx(30.0)

    second = write_nsbi_workspace(
        result=result,
        intensity=model,
        output_dir=tmp_path / "second",
        measurement="measurement",
        poi="mu",
    )
    assert second.workspace["hnsbi"]["backend"] == "native"


def test_fixed_ratio_normalizer_validates_positive_means() -> None:
    normalizer = RatioNormalizer.fit(
        {"a": np.asarray([0.5, 1.5])}, np.asarray([0.25, 0.75])
    )
    assert normalizer.means["a"] == pytest.approx(1.25)
    with pytest.raises(ValueError, match="zero reference mean"):
        RatioNormalizer.fit({"a": np.zeros(3)})


def test_ratio_normalizer_round_trip_and_integrity(tmp_path) -> None:
    normalizer = RatioNormalizer.fit(
        {
            "background": np.asarray([0.8, 1.2]),
            "signal": np.asarray([0.5, 1.5]),
        },
        metadata={"source": "independent-reference"},
    )
    path, manifest = normalizer.write(tmp_path / "normalization.json")
    recovered = RatioNormalizer.load(path)
    assert recovered.means == normalizer.means
    assert recovered.standard_errors == normalizer.standard_errors
    assert recovered.metadata == normalizer.metadata
    assert manifest.is_file()

    path.write_text('{"means": {"signal": 99.0}}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed verification"):
        RatioNormalizer.load(path)
