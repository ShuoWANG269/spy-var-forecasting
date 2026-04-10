import numpy as np
import pandas as pd
import pytest

from models.historical import predict_var, predict_var_weighted


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


def test_predict_var_shape(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS)
    assert result.shape == (10, 3)


def test_predict_var_columns(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS)
    assert list(result.columns) == CLS


def test_predict_var_all_negative(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS)
    assert (result.values < 0).all()


def test_predict_var_ordering(small_df):
    """VaR(1%) <= VaR(5%) <= VaR(10%), i.e. 1% is most extreme"""
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS)
    assert (result[0.01] <= result[0.05]).all()
    assert (result[0.05] <= result[0.10]).all()


def test_predict_var_weighted_shape(small_df):
    result = predict_var_weighted(small_df, window=WINDOW, confidence_levels=CLS)
    assert result.shape == (10, 3)


def test_predict_var_weighted_ordering(small_df):
    result = predict_var_weighted(small_df, window=WINDOW, confidence_levels=CLS)
    assert (result[0.01] <= result[0.05]).all()
    assert (result[0.05] <= result[0.10]).all()


def test_predict_var_weighted_emphasizes_recent_losses():
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    recent_loss_df = pd.DataFrame(
        {
            "log_ret": [0.2, 0.1, 0.0, -0.1, -0.2, 0.0],
            "rv5": np.full(6, 1e-4),
            "bv": np.full(6, 5e-5),
        },
        index=dates,
    )
    oldest_loss_df = pd.DataFrame(
        {
            "log_ret": [-0.2, -0.1, 0.0, 0.1, 0.2, 0.0],
            "rv5": np.full(6, 1e-4),
            "bv": np.full(6, 5e-5),
        },
        index=dates,
    )

    recent_loss_var = predict_var_weighted(
        recent_loss_df,
        window=5,
        confidence_levels=[0.20],
        lam=0.5,
    )
    oldest_loss_var = predict_var_weighted(
        oldest_loss_df,
        window=5,
        confidence_levels=[0.20],
        lam=0.5,
    )

    assert recent_loss_var.iloc[0, 0] == pytest.approx(-0.2)
    assert oldest_loss_var.iloc[0, 0] == pytest.approx(0.0)
    assert recent_loss_var.iloc[0, 0] < oldest_loss_var.iloc[0, 0]


def test_predict_var_weighted_changes_with_lambda():
    dates = pd.date_range("2020-02-01", periods=6, freq="B")
    df = pd.DataFrame(
        {
            "log_ret": [0.02, 0.01, 0.0, -0.01, -0.05, 0.0],
            "rv5": np.full(6, 1e-4),
            "bv": np.full(6, 5e-5),
        },
        index=dates,
    )

    fast_decay = predict_var_weighted(
        df,
        window=5,
        confidence_levels=[0.30],
        lam=0.5,
    )
    slow_decay = predict_var_weighted(
        df,
        window=5,
        confidence_levels=[0.30],
        lam=0.99,
    )

    assert fast_decay.iloc[0, 0] == pytest.approx(-0.05)
    assert slow_decay.iloc[0, 0] == pytest.approx(-0.01)
    assert fast_decay.iloc[0, 0] < slow_decay.iloc[0, 0]
