import json

import numpy as np
import pandas as pd
import pytest

from models.nn import predict_var_rolling


@pytest.fixture
def synthetic_data():
    """Create synthetic data with sigma series for testing."""
    np.random.seed(42)
    n = 600  # need > 500 for window=500
    dates = pd.bdate_range("2000-01-01", periods=n)
    df = pd.DataFrame({
        "log_ret": np.random.randn(n) * 0.01,
        "rv5": np.abs(np.random.randn(n)) * 0.001,
        "bv": np.abs(np.random.randn(n)) * 0.001,
    }, index=dates)
    sigma = pd.Series(
        np.abs(np.random.randn(n - 500)) * 0.01 + 0.005,
        index=dates[500:],
        name="sigma",
    )
    return df, sigma


FAST_PARAMS = {
    0.05: {"n_layers": 1, "n_units": 1, "lr": 0.01,
           "dropout": 0.0, "weight_decay": 0.0001},
}

MULTI_PARAMS = {
    0.01: {"n_layers": 1, "n_units": 1, "lr": 0.0001,
           "dropout": 0.0, "weight_decay": 0.0001},
    0.05: {"n_layers": 1, "n_units": 1, "lr": 0.01,
           "dropout": 0.0, "weight_decay": 0.0001},
}


def test_output_shape(synthetic_data):
    """Output should have one row per rolling step and one column per tau."""
    df, sigma = synthetic_data
    result = predict_var_rolling(
        df, sigma_series=sigma, window=500, retrain_freq=20,
        confidence_levels=[0.05], fixed_params=FAST_PARAMS,
        epochs=50, patience=5,
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [0.05]
    assert len(result) == len(df) - 500  # 100 rows


def test_index_matches_sigma(synthetic_data):
    """Result index should match sigma_series index."""
    df, sigma = synthetic_data
    result = predict_var_rolling(
        df, sigma_series=sigma, window=500, retrain_freq=20,
        confidence_levels=[0.05], fixed_params=FAST_PARAMS,
        epochs=50, patience=5,
    )
    pd.testing.assert_index_equal(result.index, sigma.index)


def test_multiple_taus(synthetic_data):
    """Should handle multiple quantiles independently."""
    df, sigma = synthetic_data
    result = predict_var_rolling(
        df, sigma_series=sigma, window=500, retrain_freq=20,
        confidence_levels=[0.01, 0.05], fixed_params=MULTI_PARAMS,
        epochs=50, patience=5,
    )
    assert list(result.columns) == [0.01, 0.05]
    assert len(result) == len(df) - 500


def test_retrain_freq_1(synthetic_data):
    """retrain_freq=1 should still work (retrain every step)."""
    df, sigma = synthetic_data
    df_short = df.iloc[:520]
    sigma_short = sigma.iloc[:20]
    result = predict_var_rolling(
        df_short, sigma_series=sigma_short, window=500, retrain_freq=1,
        confidence_levels=[0.05], fixed_params=FAST_PARAMS,
        epochs=20, patience=5,
    )
    assert len(result) == 20


def test_values_finite(synthetic_data):
    """All predictions should be finite numbers."""
    df, sigma = synthetic_data
    result = predict_var_rolling(
        df, sigma_series=sigma, window=500, retrain_freq=20,
        confidence_levels=[0.05], fixed_params=FAST_PARAMS,
        epochs=50, patience=5,
    )
    assert result.notna().all().all()
    assert np.isfinite(result.values).all()


def test_saves_meta(synthetic_data, tmp_path):
    """When save_name is provided, should save meta JSON."""
    df, sigma = synthetic_data
    predict_var_rolling(
        df, sigma_series=sigma, window=500, retrain_freq=20,
        confidence_levels=[0.05], fixed_params=FAST_PARAMS,
        epochs=50, patience=5,
        save_name="test_rolling", results_dir=str(tmp_path),
    )
    meta_path = tmp_path / "test_rolling_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["window"] == 500
    assert meta["retrain_freq"] == 20
    assert "0.05" in meta["fixed_params"]


def test_missing_fixed_params_raises(synthetic_data):
    """Should raise ValueError if fixed_params is None."""
    df, sigma = synthetic_data
    with pytest.raises(ValueError, match="fixed_params"):
        predict_var_rolling(
            df, sigma_series=sigma, window=500,
            confidence_levels=[0.05], fixed_params=None,
        )
