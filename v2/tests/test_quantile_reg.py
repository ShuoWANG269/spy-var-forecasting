import numpy as np
import pandas as pd
import pytest
from models.quantile_reg import predict_var


@pytest.fixture
def small_df():
    np.random.seed(42)
    n = 280
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


def test_qr_shape(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS)
    assert result.shape[1] == 3
    assert result.shape[0] > 0


def test_qr_columns(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS)
    assert list(result.columns) == CLS


def test_qr_ordering(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS)
    assert (result[0.01] <= result[0.05]).all()
    assert (result[0.05] <= result[0.10]).all()


def test_qr_prediction_uses_lagged_features(monkeypatch):
    class FakeResult:
        def __init__(self, n_params: int):
            self.params = np.ones(n_params)

    class FakeQuantReg:
        def __init__(self, y, X):
            self.n_params = X.shape[1]

        def fit(self, q, max_iter, p_tol):
            assert max_iter == 2000
            assert p_tol == 1e-6
            return FakeResult(self.n_params)

    monkeypatch.setattr("models.quantile_reg.QuantReg", FakeQuantReg)

    n = 27
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    base_df = pd.DataFrame(
        {
            "log_ret": np.linspace(-0.01, 0.01, n),
            "rv5": np.linspace(1e-4, 5e-4, n),
            "bv": np.linspace(5e-5, 2e-4, n),
        },
        index=dates,
    )
    shifted_df = base_df.copy()
    shifted_df.iloc[-1, shifted_df.columns.get_loc("rv5")] = 5.0
    shifted_df.iloc[-1, shifted_df.columns.get_loc("bv")] = 2.5

    base_result = predict_var(base_df, window=5, confidence_levels=[0.05])
    shifted_result = predict_var(shifted_df, window=5, confidence_levels=[0.05])

    assert base_result.iloc[0, 0] == pytest.approx(shifted_result.iloc[0, 0])
