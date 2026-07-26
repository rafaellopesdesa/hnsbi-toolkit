from __future__ import annotations

import numpy as np

from hnsbi.bayes import BoxUniform, IndependentNormal


def test_independent_normal_sampling_and_log_prob() -> None:
    distribution = IndependentNormal(
        mean=np.asarray([0.0, 1.0]),
        scale=np.asarray([1.0, 2.0]),
    )
    first = distribution.sample(5, rng=np.random.default_rng(2))
    second = distribution.sample(5, rng=np.random.default_rng(2))
    np.testing.assert_allclose(first, second)
    expected = -np.log(4.0 * np.pi)
    assert distribution.log_prob(np.asarray([[0.0, 1.0]]))[0] == expected


def test_box_uniform_has_normalized_constant_density_and_support() -> None:
    distribution = BoxUniform(
        low=np.asarray([-1.0, 0.0]),
        high=np.asarray([1.0, 4.0]),
    )
    values = np.asarray([[0.0, 1.0], [2.0, 1.0]])
    log_prob = distribution.log_prob(values)
    assert log_prob[0] == -np.log(8.0)
    assert log_prob[1] == -np.inf
