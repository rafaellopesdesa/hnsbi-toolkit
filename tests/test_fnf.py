from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hnsbi.artifacts import ArtifactIntegrityError
from hnsbi.fnf import (
    FactorizableDensity,
    FactorizableResidualStack,
    FNFAnchor,
    FNFResidualConfig,
    FNFStandardizer,
    FNFTrainer,
    FNFTrainingConfig,
    LogQuadraticYieldMorph,
    diagnose_fnf,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _anchor(values, point, *, groups=None, name=""):
    return FNFAnchor(
        values=np.asarray(values, dtype=np.float32),
        point=point,
        groups=groups,
        name=name,
    )


def test_fnf_config_canonicalizes_interactions_and_validates_scales():
    config = FNFResidualConfig(
        n_features=2,
        nuisance_names=("scale", "resolution"),
        interactions=(("resolution", "scale"),),
    )

    assert config.interactions == (("scale", "resolution"),)
    assert config.nuisance_centers == (0.0, 0.0)
    assert config.nuisance_scales == (1.0, 1.0)
    assert FNFResidualConfig.from_dict(config.to_dict()) == config

    with pytest.raises(ValueError, match="Duplicate FNF interaction"):
        FNFResidualConfig(
            n_features=2,
            nuisance_names=("a", "b"),
            interactions=(("a", "b"), ("b", "a")),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        FNFResidualConfig(
            n_features=1,
            nuisance_names=("a",),
            nuisance_scales=(0.0,),
        )


def test_log_quadratic_yield_morph_is_positive_and_closes_anchors():
    morph = LogQuadraticYieldMorph.from_anchors(
        100.0,
        {
            "scale": (80.0, 125.0),
            "theory": (90.0, 110.0),
        },
    )

    assert morph.expected_yield() == pytest.approx(100.0)
    assert morph.expected_yield({"scale": -1.0}) == pytest.approx(80.0)
    assert morph.expected_yield({"scale": 1.0}) == pytest.approx(125.0)
    assert morph.expected_yield({"theory": -1.0}) == pytest.approx(90.0)
    assert morph.expected_yield({"theory": 1.0}) == pytest.approx(110.0)
    assert morph.expected_yield({"scale": 4.0, "theory": -3.0}) > 0

    with pytest.raises(ValueError, match="positive"):
        LogQuadraticYieldMorph.from_anchors(100.0, {"scale": (0.0, 110.0)})


def test_interactions_require_joint_training_anchors():
    trainer = FNFTrainer(
        FNFResidualConfig(
            n_features=1,
            nuisance_names=("a", "b"),
            interactions=(("a", "b"),),
        )
    )
    axis_anchors = [
        _anchor([[0.0], [1.0]], {"a": -1.0}, name="a-down"),
        _anchor([[0.0], [1.0]], {"a": 1.0}, name="a-up"),
        _anchor([[0.0], [1.0]], {"b": -1.0}, name="b-down"),
        _anchor([[0.0], [1.0]], {"b": 1.0}, name="b-up"),
    ]

    with pytest.raises(ValueError, match="requires a joint anchor"):
        trainer.validate_anchors(axis_anchors)

    trainer.validate_anchors(
        [
            *axis_anchors,
            _anchor(
                [[0.0], [1.0]],
                {"a": 1.0, "b": -1.0},
                name="joint",
            ),
        ]
    )


def test_deterministic_group_split_is_safe_across_correlated_anchors():
    groups = np.asarray([f"event-{index}" for index in range(20)])
    values = np.arange(20, dtype=np.float32).reshape(-1, 1)
    anchors = [
        _anchor(values - 1.0, {"shift": -1.0}, groups=groups, name="down"),
        _anchor(values + 1.0, {"shift": 1.0}, groups=groups, name="up"),
    ]
    trainer = FNFTrainer(
        FNFResidualConfig(n_features=1, nuisance_names=("shift",)),
        FNFTrainingConfig(
            validation_fraction=0.2,
            holdout_fraction=0.2,
            epochs=1,
        ),
    )

    split = trainer.split(anchors, seed=19)
    repeated = trainer.split(anchors, seed=19)
    for first, second in zip(
        (
            *split.training,
            *split.validation,
            *split.holdout,
        ),
        (
            *repeated.training,
            *repeated.validation,
            *repeated.holdout,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(first, second)

    for anchor_index in range(2):
        training_groups = set(groups[split.training[anchor_index]])
        validation_groups = set(groups[split.validation[anchor_index]])
        holdout_groups = set(groups[split.holdout[anchor_index]])
        assert not training_groups.intersection(validation_groups)
        assert not training_groups.intersection(holdout_groups)
        assert not validation_groups.intersection(holdout_groups)

    assert set(groups[split.training[0]]) == set(groups[split.training[1]])
    assert set(groups[split.validation[0]]) == set(groups[split.validation[1]])
    assert set(groups[split.holdout[0]]) == set(groups[split.holdout[1]])


def test_equal_anchor_loss_does_not_depend_on_anchor_row_count():
    first = np.asarray([-1.0, -3.0])
    first_weights = np.ones(2)
    second = np.asarray([-5.0])
    second_weights = np.ones(1)

    original = FNFTrainer.equal_anchor_loss(
        (first, second), (first_weights, second_weights)
    )
    duplicated = FNFTrainer.equal_anchor_loss(
        (first, np.repeat(second, 20)),
        (first_weights, np.ones(20)),
    )

    assert original == pytest.approx(3.5)
    assert duplicated == pytest.approx(original)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is an optional flow dependency")
def test_residual_is_exact_identity_and_invertible_with_nonzero_parameters():
    import torch

    config = FNFResidualConfig(
        n_features=3,
        nuisance_names=("scale", "resolution"),
        num_layers=3,
        hidden_features=(8,),
        interactions=(("scale", "resolution"),),
        log_scale_clip=1.0,
        shift_clip=2.0,
    )
    residual = FactorizableResidualStack(
        config,
        standardizer=FNFStandardizer(
            mean=np.asarray([1.0, -2.0, 0.5]),
            scale=np.asarray([2.0, 0.5, 1.5]),
        ),
    )
    torch.manual_seed(7)
    with torch.no_grad():
        for parameter in residual.module.parameters():
            parameter.normal_(mean=0.0, std=0.04)
    values = np.random.default_rng(3).normal(size=(12, 3)).astype(np.float32)

    identity, identity_logdet = residual.to_nominal(values, {})
    np.testing.assert_allclose(identity, values, atol=2e-6)
    np.testing.assert_allclose(identity_logdet, 0.0, atol=2e-6)

    point = {"scale": 0.7, "resolution": -0.35}
    nominal, forward_logdet = residual.to_nominal(values, point)
    reconstructed, inverse_logdet = residual.from_nominal(nominal, point)
    forwarded = residual.forward(values, point)
    inverted = residual.inverse(nominal, point)
    np.testing.assert_allclose(forwarded[0], nominal)
    np.testing.assert_allclose(inverted[0], reconstructed)
    np.testing.assert_allclose(reconstructed, values, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(forward_logdet + inverse_logdet, 0.0, atol=2e-5)
    assert residual.jacobian_logdet_error(values[:2], point) < 2e-4


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is an optional flow dependency")
def test_pairwise_networks_vanish_exactly_on_nuisance_axes():
    import torch

    residual = FactorizableResidualStack(
        FNFResidualConfig(
            n_features=2,
            nuisance_names=("a", "b"),
            num_layers=1,
            hidden_features=(4,),
            interactions=(("a", "b"),),
        )
    )
    values = np.asarray([[0.2, -0.3], [1.0, 0.5]], dtype=np.float32)
    axis_before = residual.to_nominal(values, {"a": 0.8})[0]
    joint_before = residual.to_nominal(values, {"a": 0.8, "b": -0.6})[0]

    with torch.no_grad():
        for name, parameter in residual.module.named_parameters():
            if "cross_networks" in name:
                parameter.add_(0.25)

    axis_after = residual.to_nominal(values, {"a": 0.8})[0]
    joint_after = residual.to_nominal(values, {"a": 0.8, "b": -0.6})[0]
    np.testing.assert_array_equal(axis_after, axis_before)
    assert not np.allclose(joint_after, joint_before)


class _GaussianDensity:
    def log_prob(self, values):
        values = np.asarray(values)
        return -0.5 * np.square(values).sum(axis=1)

    def sample(self, n, *, rng=None):
        generator = np.random.default_rng() if rng is None else rng
        return generator.normal(size=(n, 2)).astype(np.float32)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is an optional flow dependency")
def test_density_apis_diagnostics_and_portable_round_trip(tmp_path):
    import torch

    residual = FactorizableResidualStack(
        FNFResidualConfig(
            n_features=2,
            nuisance_names=("shift",),
            num_layers=2,
            hidden_features=(6,),
        )
    )
    torch.manual_seed(9)
    with torch.no_grad():
        for parameter in residual.module.parameters():
            parameter.normal_(mean=0.0, std=0.03)
    gaussian = _GaussianDensity()
    density = FactorizableDensity(
        residual,
        gaussian,
        base_torch_log_prob=lambda values: -0.5 * torch.square(values).sum(dim=1),
    )
    values = gaussian.sample(300, rng=np.random.default_rng(21))

    assert density.log_prob(values[:7], {"shift": 0.4}).shape == (7,)
    assert density.log_ratio(values[:7], {"shift": 0.4}).shape == (7,)
    np.testing.assert_allclose(
        density.forward(values[:7], {"shift": 0.4}),
        density.to_nominal(values[:7], {"shift": 0.4}),
    )
    assert density.sample(6, {"shift": -0.2}, rng=np.random.default_rng(5)).shape == (
        6,
        2,
    )
    report = diagnose_fnf(
        density,
        values,
        points=({"shift": -0.5}, {"shift": 0.5}),
        check_jacobian=True,
        jacobian_rows=1,
    )
    assert report.identity_max_abs < 2e-6
    assert report.identity_logdet_max_abs < 2e-6
    assert all(item.round_trip_max_abs < 2e-5 for item in report.points)
    assert all(np.isfinite(item.normalization) for item in report.points)
    assert all(item.importance_ess > 0 for item in report.points)

    artifact = residual.save(tmp_path, metadata={"domain": "SR/signal"})
    loaded = FactorizableResidualStack.load(artifact.manifest_path)
    expected = residual.to_nominal(values[:10], {"shift": 0.35})
    observed = loaded.to_nominal(values[:10], {"shift": 0.35})
    np.testing.assert_allclose(observed[0], expected[0], atol=1e-7)
    np.testing.assert_allclose(observed[1], expected[1], atol=1e-7)

    artifact.state_path.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match="mismatch"):
        FactorizableResidualStack.load(artifact.manifest_path)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is an optional flow dependency")
def test_training_smoke_uses_group_safe_split_and_returns_finite_history():
    import torch

    rng = np.random.default_rng(15)
    nominal = rng.normal(size=(40, 1)).astype(np.float32)
    groups = np.arange(len(nominal))
    anchors = [
        _anchor(
            nominal - 0.7,
            {"shift": -1.0},
            groups=groups,
            name="down",
        ),
        _anchor(
            nominal + 0.7,
            {"shift": 1.0},
            groups=groups,
            name="up",
        ),
    ]
    trainer = FNFTrainer(
        FNFResidualConfig(
            n_features=1,
            nuisance_names=("shift",),
            num_layers=1,
            hidden_features=(4,),
        ),
        FNFTrainingConfig(
            epochs=2,
            batch_size=16,
            validation_fraction=0.2,
            patience=2,
        ),
    )

    result = trainer.fit(
        anchors,
        base_torch_log_prob=lambda values: -0.5 * torch.square(values).sum(dim=1),
        features=("x",),
        seed=4,
    )

    assert len(result.history) == 2
    assert result.best_epoch in {1, 2}
    assert all(np.isfinite(epoch.training_loss) for epoch in result.history)
    assert all(np.isfinite(epoch.validation_loss) for epoch in result.history)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is an optional flow dependency")
def test_training_learns_a_translation_on_unseen_rows():
    import torch

    rng = np.random.default_rng(31)
    nominal = rng.normal(size=(240, 1)).astype(np.float32)
    groups = np.arange(len(nominal))
    anchors = [
        _anchor(
            nominal - 0.8,
            {"shift": -1.0},
            groups=groups,
            name="down",
        ),
        _anchor(
            nominal + 0.8,
            {"shift": 1.0},
            groups=groups,
            name="up",
        ),
    ]
    trainer = FNFTrainer(
        FNFResidualConfig(
            n_features=1,
            nuisance_names=("shift",),
            num_layers=1,
            hidden_features=(12, 12),
            log_scale_clip=0.5,
        ),
        FNFTrainingConfig(
            epochs=30,
            steps_per_epoch=8,
            batch_size=32,
            learning_rate=0.005,
            validation_fraction=0.2,
            holdout_fraction=0.2,
            patience=30,
        ),
    )
    result = trainer.fit(
        anchors,
        base_torch_log_prob=lambda values: -0.5 * torch.square(values).sum(dim=1),
        features=("x",),
        seed=12,
    )
    holdout = result.split.holdout[1]
    varied = anchors[1].values[holdout]
    transformed = result.residual.to_nominal(
        varied,
        {"shift": 1.0},
    )[0]
    raw_bias = abs(float(np.mean(varied)))
    transformed_bias = abs(float(np.mean(transformed)))
    assert transformed_bias < 0.55 * raw_bias
