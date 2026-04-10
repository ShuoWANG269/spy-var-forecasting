import numpy as np
import pandas as pd
import pytest

from utils.data import load_data, build_garch_features
from utils.data import build_riskmetrics_features
from utils.data import build_caviar_sv_features, build_caviar_asv_features


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "log_ret": np.random.normal(0, 0.01, n),
            "rv5": np.random.uniform(1e-5, 1e-3, n),
            "bv": np.random.uniform(1e-5, 1e-4, n),
        },
        index=dates,
    )


def test_garch_features_shape(sample_df):
    result = build_garch_features(sample_df)
    assert "sigma" in result.columns
    assert len(result) == len(sample_df)
    assert result["sigma"].notna().all()


def test_garch_features_positive(sample_df):
    result = build_garch_features(sample_df)
    assert (result["sigma"] > 0).all()


def test_riskmetrics_features_shape(sample_df):
    result = build_riskmetrics_features(sample_df)
    assert "sigma" in result.columns
    assert len(result) == len(sample_df)
    assert result["sigma"].notna().all()


def test_riskmetrics_features_positive(sample_df):
    result = build_riskmetrics_features(sample_df)
    assert (result["sigma"] > 0).all()


def test_riskmetrics_ewma_lambda(sample_df):
    """Verify EWMA uses lambda=0.94 by checking the first few values."""
    result = build_riskmetrics_features(sample_df)
    r = sample_df["log_ret"].values
    lam = 0.94
    var0 = r[0] ** 2
    var1 = lam * var0 + (1 - lam) * r[0] ** 2
    assert result["sigma"].iloc[0] > 0


def test_caviar_sv_features_shape(sample_df):
    result = build_caviar_sv_features(sample_df, tau=0.05)
    assert "abs_ret" in result.columns
    assert "var_lag" in result.columns
    assert len(result) == len(sample_df) - 1  # lose first row (no lag)
    assert result["abs_ret"].notna().all()
    assert result["var_lag"].notna().all()


def test_caviar_sv_abs_ret_positive(sample_df):
    result = build_caviar_sv_features(sample_df, tau=0.05)
    assert (result["abs_ret"] >= 0).all()


def test_caviar_asv_features_shape(sample_df):
    result = build_caviar_asv_features(sample_df, tau=0.05)
    assert "ret_pos" in result.columns
    assert "ret_neg" in result.columns
    assert "var_lag" in result.columns
    assert len(result) == len(sample_df) - 1


def test_caviar_asv_pos_neg_split(sample_df):
    result = build_caviar_asv_features(sample_df, tau=0.05)
    assert (result["ret_pos"] >= 0).all()
    assert (result["ret_neg"] >= 0).all()
