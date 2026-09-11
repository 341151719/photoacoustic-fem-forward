from __future__ import annotations

import numpy as np

from pa_fem.strict_validation import _analytic_sensor, _db


def test_plane_gaussian_analytic_peak_and_db_metric():
    y0 = 1.0e-3
    sigma = 0.1e-3
    c = 1480.0
    times = np.array([0.0, y0 / c])
    signal = _analytic_sensor(times, y0, sigma, c)
    assert signal[1] == 0.5
    assert signal[0] < 1e-20
    assert _db(1e-2) == -40.0
