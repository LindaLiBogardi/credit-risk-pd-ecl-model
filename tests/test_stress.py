"""Unit tests for the stress test functions."""

import numpy as np
import pandas as pd

from src.stress import calibrate_delta, shift_pd


def test_zero_shift_leaves_pd_unchanged():
    pd_ = pd.Series([0.02, 0.10, 0.30])
    assert np.allclose(shift_pd(pd_, 0.0), pd_)


def test_positive_shift_raises_pd_and_keeps_ranking():
    pd_ = pd.Series([0.02, 0.10, 0.30])
    shifted = shift_pd(pd_, 0.5)
    assert (shifted > pd_).all()
    assert shifted.is_monotonic_increasing


def test_calibrate_delta_hits_target():
    pd_ = pd.Series([0.02, 0.10, 0.30])
    delta = calibrate_delta(pd_, target_mean_pd=0.20)
    assert np.isclose(shift_pd(pd_, delta).mean(), 0.20)