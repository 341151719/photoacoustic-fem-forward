from __future__ import annotations

import numpy as np

from pa_fem.dual_compressed import bucket_delayed, make_patterns


def test_patterns_are_reproducible_nonnegative_and_nonempty():
    a = make_patterns(np.random.default_rng(7), 12, 5)
    b = make_patterns(np.random.default_rng(7), 12, 5)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) <= {0.0, 1.0}
    assert np.all(np.sum(a, axis=1) >= 1.0)


def test_channel_delay_occurs_before_bucket_sum():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    fields = np.zeros((4, 2, 1))
    fields[0, 0, 0] = 1.0
    fields[1, 1, 0] = 1.0
    result = bucket_delayed(fields, time, time, np.array([0.0, 1.0]),
                            np.array([0.5, 0.5]))
    assert np.allclose(result[:, 0], [0.5, 0.0, 0.5, 0.0])
