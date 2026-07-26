from __future__ import annotations

import numpy as np
import pytest

from hnsbi.asimov import AsimovBuilder
from hnsbi.intensity import Component, IntensityModel, Parameter, RatioNormalizer
from hnsbi.likelihood import ExtendedUnbinnedLikelihood
from hnsbi.toys import ToyGenerator
from hnsbi.workspace import write_nsbi_workspace


class ConstantSampler:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def sample(self, n: int, *, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.full((n, 1), self.value)


def test_toy_generator_binds_persisted_workspace_formulas(tmp_path) -> None:
    model = IntensityModel(
        [
            Component("background", 4.0, "1"),
            Component("signal", 3.0, "mu"),
        ],
        [Parameter("mu", 1.0, (0.0, 4.0))],
    )
    result = AsimovBuilder(
        reference=ConstantSampler(),
        ratios={
            "background": lambda values: np.ones(len(values)),
            "signal": lambda values: np.ones(len(values)),
        },
        intensity=model,
        features=["x"],
    ).build({"mu": 1.0}, n_events=100, seed=1)
    export = write_nsbi_workspace(
        result=result,
        intensity=model,
        output_dir=tmp_path,
        measurement="measurement",
        poi="mu",
    )
    generator = ToyGenerator.from_workspace(
        export.path,
        component_samplers={
            "background": ConstantSampler(-1.0),
            "signal": ConstantSampler(1.0),
        },
    )
    toy = generator.generate({"mu": 2.0}, seed=9)
    assert toy.component_expectations == {"background": 4.0, "signal": 6.0}
    assert set(toy.events.columns["component"]).issubset({"background", "signal"})
    assert toy.events.metadata["point"] == {"mu": 2.0}
    assert toy.events.metadata["component_sampling_diagnostics"] == {
        "background": {
            "method": "direct",
            "proposal_pool_size": None,
            "proposal_pool_ess": None,
            "proposal_pool_rounds": 0,
        },
        "signal": {
            "method": "direct",
            "proposal_pool_size": None,
            "proposal_pool_ess": None,
            "proposal_pool_rounds": 0,
        },
    }


def test_importance_toy_records_proposal_diagnostics() -> None:
    model = IntensityModel(
        [Component("signal", 25.0, "mu")],
        [
            Parameter("mu", 1.0, (0.0, 4.0)),
            Parameter(
                "alpha",
                0.0,
                (-3.0, 3.0),
                constrained=True,
                constraint_mean=0.0,
                constraint_sigma=0.5,
            ),
        ],
    )
    generator = ToyGenerator(
        intensity=model,
        features=["x"],
        reference=ConstantSampler(),
        ratios={"signal": lambda values: np.ones(len(values))},
        normalizer=RatioNormalizer({"signal": 1.0}),
    )
    toy = generator.generate(
        {"mu": 1.3, "alpha": 0.7},
        seed=7,
        oversample=2,
        min_pool=50,
    )
    diagnostic = toy.events.metadata["component_sampling_diagnostics"]["signal"]
    assert toy.events.metadata["point"] == {"mu": 1.3, "alpha": 0.7}
    assert toy.events.metadata["constraint_observations"] == (
        toy.constraint_observations
    )
    assert set(toy.constraint_observations) == {"alpha"}
    assert np.isfinite(toy.constraint_observations["alpha"])
    assert diagnostic["method"] == "reference_importance_resampling"
    assert diagnostic["proposal_pool_size"] == max(
        50, 2 * toy.component_counts["signal"]
    )
    assert diagnostic["proposal_pool_ess"] == pytest.approx(
        diagnostic["proposal_pool_size"]
    )
    assert diagnostic["proposal_pool_rounds"] == 1


def test_upstream_workspace_initializes_at_asimov_generating_point(tmp_path) -> None:
    model = IntensityModel(
        [Component("signal", 3.0, "mu")],
        [Parameter("mu", 1.0, (0.0, 4.0))],
    )
    result = AsimovBuilder(
        reference=ConstantSampler(),
        ratios={"signal": lambda values: np.ones(len(values))},
        intensity=model,
        features=["x"],
    ).build({"mu": 2.5}, n_events=100, seed=1)
    export = write_nsbi_workspace(
        result=result,
        intensity=model,
        output_dir=tmp_path,
        measurement="measurement",
        poi="mu",
    )
    entry = export.workspace["measurements"][0]["config"]["parameters"][0]
    assert entry["inits"] == [2.5]
    assert export.workspace["hnsbi"]["parameter_nominals"] == {"mu": 1.0}


def test_nonstandard_constraint_routes_away_from_upstream(tmp_path) -> None:
    model = IntensityModel(
        [Component("sample", 4.0, "1")],
        [
            Parameter(
                "alpha",
                0.3,
                (-2.0, 2.0),
                constrained=True,
                constraint_mean=0.3,
                constraint_sigma=0.7,
            )
        ],
    )
    result = AsimovBuilder(
        reference=ConstantSampler(),
        ratios={"sample": lambda values: np.ones(len(values))},
        intensity=model,
        features=["x"],
    ).build({"alpha": 0.3}, n_events=40)
    export = write_nsbi_workspace(
        result=result,
        intensity=model,
        output_dir=tmp_path,
        measurement="measurement",
        poi="alpha",
    )
    assert not export.upstream_compatible
    likelihood = ExtendedUnbinnedLikelihood.from_workspace(export.path)
    assert likelihood.constraints["alpha"].mean == 0.3
    assert likelihood.constraints["alpha"].sigma == 0.7


def test_workspace_rejects_changed_intensity_after_asimov_build(tmp_path) -> None:
    original = IntensityModel(
        [Component("sample", 4.0, "1")],
        [Parameter("mu", 1.0, (0.0, 3.0))],
    )
    result = AsimovBuilder(
        reference=ConstantSampler(),
        ratios={"sample": lambda values: np.ones(len(values))},
        intensity=original,
        features=["x"],
    ).build({"mu": 1.0}, n_events=20)
    changed = IntensityModel(
        [Component("sample", 5.0, "1")],
        [Parameter("mu", 1.0, (0.0, 3.0))],
    )
    with pytest.raises(ValueError, match="exact intensity"):
        write_nsbi_workspace(
            result=result,
            intensity=changed,
            output_dir=tmp_path,
            measurement="measurement",
            poi="mu",
        )
