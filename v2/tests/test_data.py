import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from utils.data import load_data

_DATA_PATH = str(Path(__file__).resolve().parent.parent.parent / "spy_data.csv")


def test_load_data_returns_dataframe():
    df = load_data(_DATA_PATH)
    assert isinstance(df, pd.DataFrame)


def test_load_data_columns():
    df = load_data(_DATA_PATH)
    assert list(df.columns) == ["log_ret", "rv5", "bv"]


def test_load_data_index_is_datetime():
    df = load_data(_DATA_PATH)
    assert pd.api.types.is_datetime64_any_dtype(df.index)


def test_load_data_no_nan():
    df = load_data(_DATA_PATH)
    assert df.isna().sum().sum() == 0


def test_load_data_row_count():
    df = load_data(_DATA_PATH)
    assert len(df) == 4640  # spy_data.csv has 4640 rows


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2000-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "log_ret": np.random.normal(0, 0.01, n),
            "rv5": np.random.uniform(1e-5, 1e-3, n),
            "bv": np.random.uniform(1e-5, 1e-4, n),
        },
        index=dates,
    )


def test_rolling_windows_count(sample_df):
    from utils.data import rolling_windows
    windows = list(rolling_windows(sample_df, window=250))
    assert len(windows) == 50  # 300 - 250


def test_rolling_windows_train_size(sample_df):
    from utils.data import rolling_windows
    train, test_date, test_ret = next(rolling_windows(sample_df, window=250))
    assert len(train) == 250


def test_rolling_windows_test_date_type(sample_df):
    from utils.data import rolling_windows
    _, test_date, _ = next(rolling_windows(sample_df, window=250))
    assert isinstance(test_date, pd.Timestamp)


def test_rolling_windows_test_return_is_float(sample_df):
    from utils.data import rolling_windows
    _, _, test_ret = next(rolling_windows(sample_df, window=250))
    assert isinstance(test_ret, float)


def test_rolling_windows_no_overlap_leakage(sample_df):
    """Training window must not include the test_date itself"""
    from utils.data import rolling_windows
    train, test_date, test_ret = next(rolling_windows(sample_df, window=250))
    assert test_date not in train.index


def test_build_har_features_columns(sample_df):
    from utils.data import build_har_features
    feat = build_har_features(sample_df)
    for col in ["C_d", "C_w", "C_m", "J_d"]:
        assert col in feat.columns


def test_build_har_features_no_nan(sample_df):
    from utils.data import build_har_features
    feat = build_har_features(sample_df)
    assert feat[["C_d", "C_w", "C_m", "J_d"]].isna().sum().sum() == 0


def test_build_har_features_jump_nonneg(sample_df):
    from utils.data import build_har_features
    feat = build_har_features(sample_df)
    assert (feat["J_d"] >= 0).all()


def test_build_har_features_c_d_equals_bv(sample_df):
    from utils.data import build_har_features
    feat = build_har_features(sample_df)
    assert (feat["C_d"] == feat["bv"]).all()


def test_build_har_features_row_reduction(sample_df):
    """Row count decreases after dropna: 300 - 22 + 1 = 279 rows"""
    from utils.data import build_har_features
    feat = build_har_features(sample_df)
    assert len(feat) == 300 - 22 + 1
