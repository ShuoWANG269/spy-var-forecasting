import numpy as np
import pandas as pd
import pytest
from models.parametric import predict_var


@pytest.fixture
def small_df():
    np.random.seed(42)
    n = 260
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "log_ret": np.random.normal(0, 0.01, n),
            "rv5": np.random.uniform(1e-5, 1e-3, n),
            "bv": np.random.uniform(1e-5, 1e-4, n),
        },
        index=dates,
    )


CLS = [0.01, 0.05, 0.10]
WINDOW = 250


def test_predict_var_normal_shape(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS, dist="normal")
    assert result.shape == (10, 3)


def test_predict_var_t_shape(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS, dist="t")
    assert result.shape == (10, 3)


def test_predict_var_normal_ordering(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS, dist="normal")
    assert (result[0.01] <= result[0.05]).all()
    assert (result[0.05] <= result[0.10]).all()


def test_predict_var_t_ordering(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS, dist="t")
    assert (result[0.01] <= result[0.05]).all()
    assert (result[0.05] <= result[0.10]).all()


def test_predict_var_t_more_extreme_than_normal(small_df):
    """t-distribution VaR at 1% should be more extreme than normal (fat tails)"""
    r_n = predict_var(small_df, window=WINDOW, confidence_levels=CLS, dist="normal")
    r_t = predict_var(small_df, window=WINDOW, confidence_levels=CLS, dist="t")
    assert (r_t[0.01] <= r_n[0.01]).mean() > 0.5


def test_predict_var_invalid_dist_raises(small_df):
    with pytest.raises(ValueError, match="Unsupported dist"):
        predict_var(small_df, window=WINDOW, confidence_levels=CLS, dist="lognormal")
