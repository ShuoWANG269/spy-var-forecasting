import numpy as np
import pandas as pd
import pytest
from scipy import stats
from models.garch import predict_var


@pytest.fixture
def small_df():
    np.random.seed(42)
    n = 260
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    # Add volatility clustering to make GARCH meaningful
    vol = 0.01 + 0.005 * np.abs(np.sin(np.linspace(0, 10, n)))
    return pd.DataFrame(
        {
            "log_ret": np.random.normal(0, vol, n),
            "rv5": np.random.uniform(1e-5, 1e-3, n),
            "bv": np.random.uniform(1e-5, 1e-4, n),
        },
        index=dates,
    )


CLS = [0.01, 0.05, 0.10]
WINDOW = 250


# ---------- basic predict_var tests (return_sigma=False default) ----------

def test_garch_normal_shape(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS,
                         model_type="GARCH", dist="normal")
    assert result.shape == (10, 3)


def test_gjr_t_shape(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS,
                         model_type="GJR", dist="t")
    assert result.shape == (10, 3)


def test_garch_ordering(small_df):
    result = predict_var(small_df, window=WINDOW, confidence_levels=CLS,
                         model_type="GARCH", dist="normal")
    assert (result[0.01] <= result[0.05]).all()
    assert (result[0.05] <= result[0.10]).all()


def test_garch_t_more_extreme_at_1pct(small_df):
    """t-residual GARCH should produce more extreme VaR at 1% than normal GARCH"""
    r_n = predict_var(small_df, window=WINDOW, confidence_levels=CLS,
                      model_type="GARCH", dist="normal")
    r_t = predict_var(small_df, window=WINDOW, confidence_levels=CLS,
                      model_type="GARCH", dist="t")
    assert (r_t[0.01] <= r_n[0.01]).mean() > 0.4


def test_garch_t_uses_standardized_student_t_quantile(monkeypatch):
    class FakeResult:
        params = {"nu": 5.0}

        def forecast(self, horizon: int):
            assert horizon == 1
            return type(
                "Forecast",
                (),
                {
                    "mean": pd.DataFrame([[0.0]]),
                    "variance": pd.DataFrame([[10000.0]]),
                },
            )()

    class FakeModel:
        def fit(self, disp: str, show_warning: bool):
            assert disp == "off"
            assert show_warning is False
            return FakeResult()

    def fake_arch_model(*args, **kwargs):
        return FakeModel()

    monkeypatch.setattr("models.garch.arch_model", fake_arch_model)

    df = pd.DataFrame(
        {
            "log_ret": np.linspace(-0.02, 0.02, 6),
            "rv5": np.full(6, 1e-4),
            "bv": np.full(6, 5e-5),
        },
        index=pd.date_range("2020-01-01", periods=6, freq="B"),
    )

    result = predict_var(
        df,
        window=5,
        confidence_levels=[0.01],
        model_type="GARCH",
        dist="t",
    )
    expected = stats.t.ppf(0.01, 5.0) * np.sqrt((5.0 - 2.0) / 5.0)

    assert result.iloc[0, 0] == pytest.approx(expected)


def test_garch_invalid_dist_raises(small_df):
    with pytest.raises(ValueError, match="Unsupported dist"):
        predict_var(
            small_df,
            window=WINDOW,
            confidence_levels=CLS,
            model_type="GARCH",
            dist="lognormal",
        )


def test_garch_invalid_model_type_raises(small_df):
    with pytest.raises(ValueError, match="Unsupported model_type"):
        predict_var(
            small_df,
            window=WINDOW,
            confidence_levels=CLS,
            model_type="EGARCH",
            dist="normal",
        )


# ---------- return_sigma=True tests (merged from test_garch_sigma.py) ----------

def test_return_sigma_returns_tuple(small_df):
    """predict_var with return_sigma=True should return (var_df, sigma_series)."""
    var_df, sigma_s = predict_var(
        small_df, window=WINDOW, confidence_levels=CLS,
        model_type="GARCH", dist="normal", return_sigma=True,
    )
    assert isinstance(var_df, pd.DataFrame)
    assert isinstance(sigma_s, pd.Series)
    assert len(var_df) == len(sigma_s)
    assert list(var_df.columns) == [0.01, 0.05, 0.10]
    assert len(var_df) == 10  # 260 - 250


def test_sigma_values_positive(small_df):
    """All sigma values should be positive."""
    _, sigma_s = predict_var(
        small_df, window=WINDOW, confidence_levels=[0.01],
        model_type="GARCH", dist="normal", return_sigma=True,
    )
    assert (sigma_s > 0).all()


def test_sigma_index_matches_var(small_df):
    """sigma index must match var_df index."""
    var_df, sigma_s = predict_var(
        small_df, window=WINDOW, confidence_levels=[0.01],
        model_type="GARCH", dist="normal", return_sigma=True,
    )
    pd.testing.assert_index_equal(var_df.index, sigma_s.index)


def test_var_with_sigma_matches_without(small_df):
    """VaR output should match whether return_sigma is True or False."""
    var_orig = predict_var(
        small_df, window=WINDOW, confidence_levels=[0.01],
        model_type="GARCH", dist="normal",
    )
    var_new, _ = predict_var(
        small_df, window=WINDOW, confidence_levels=[0.01],
        model_type="GARCH", dist="normal", return_sigma=True,
    )
    pd.testing.assert_frame_equal(var_orig, var_new)
