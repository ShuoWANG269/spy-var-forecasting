import numpy as np
import pandas as pd
import pytest
import torch

from models.lstm import _VaRLSTM


# -- Task 1: _VaRLSTM tests ------------------------------------------------


def test_v2_output_shape():
    torch.manual_seed(0)
    model = _VaRLSTM(input_size=5, hidden_size=64, fc_units=32, dropout=0.0)
    x = torch.randn(4, 20, 5)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1)


def test_v2_single_feature():
    torch.manual_seed(0)
    model = _VaRLSTM(input_size=1, hidden_size=64, fc_units=32, dropout=0.0)
    x = torch.randn(2, 20, 1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)


def test_v2_dropout():
    model = _VaRLSTM(input_size=3, hidden_size=32, fc_units=16, dropout=0.2)
    x = torch.randn(2, 10, 3)
    model.train()
    out = model(x)
    assert out.shape == (2, 1)


# -- Task 3: _build_features + _build_sequences tests ----------------------

from models.lstm import _build_features, _build_sequences, _StandardScaler


@pytest.fixture
def small_df():
    np.random.seed(42)
    n = 600
    dates = pd.bdate_range("2000-01-01", periods=n)
    return pd.DataFrame(
        {
            "log_ret": np.random.normal(0, 0.01, n),
            "rv5": np.random.uniform(1e-5, 1e-3, n),
            "bv": np.random.uniform(1e-5, 1e-4, n),
        },
        index=dates,
    )


def test_build_features_returns(small_df):
    feat_df, cols = _build_features(small_df.iloc[:500], feature_set="returns")
    assert cols == ["log_ret"]
    assert "log_ret" in feat_df.columns
    assert len(feat_df) == 500


def test_build_features_har(small_df):
    feat_df, cols = _build_features(small_df.iloc[:500], feature_set="har")
    assert cols == ["log_ret", "C_d", "C_w", "C_m", "J_d"]
    assert len(feat_df) < 500


def test_build_features_garch_returns(small_df):
    feat_df, cols = _build_features(small_df.iloc[:500], feature_set="garch_returns")
    assert cols == ["log_ret", "sigma"]
    assert len(feat_df) == 500


def test_build_features_invalid(small_df):
    with pytest.raises(ValueError, match="Unknown feature_set"):
        _build_features(small_df.iloc[:100], feature_set="invalid")


def test_build_sequences_shape():
    np.random.seed(0)
    n = 100
    dates = pd.bdate_range("2020-01-01", periods=n)
    feat_df = pd.DataFrame(
        {"log_ret": np.random.randn(n) * 0.01, "sigma": np.abs(np.random.randn(n)) * 0.01},
        index=dates,
    )
    cols = ["log_ret", "sigma"]
    X, y, scaler = _build_sequences(feat_df, cols, seq_len=10)
    assert X.shape == (90, 10, 2)
    assert y.shape == (90,)
    assert isinstance(scaler, _StandardScaler)


def test_build_sequences_normalized():
    np.random.seed(0)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    feat_df = pd.DataFrame(
        {"log_ret": np.random.randn(n) * 0.01 + 0.001},
        index=dates,
    )
    X, y, scaler = _build_sequences(feat_df, ["log_ret"], seq_len=20)
    raw_mean = feat_df["log_ret"].mean()
    assert abs(raw_mean) > 1e-4
    assert abs(X.mean()) < 1.0


# -- Task 4: predict_var_rolling tests --------------------------------------

import json
from models.lstm import predict_var_rolling

FAST_HPARAMS = {
    "hidden_size": 16, "fc_units": 8, "dropout": 0.0,
    "lr": 1e-3, "weight_decay": 0.0,
}


def test_rolling_output_shape(small_df):
    result = predict_var_rolling(
        small_df, window=500, retrain_freq=20,
        confidence_levels=[0.05], feature_set="returns",
        hparams=FAST_HPARAMS, epochs=20, patience=5,
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == [0.05]
    expected_rows = len(small_df) - 500
    assert len(result) == expected_rows


def test_rolling_index_dates(small_df):
    result = predict_var_rolling(
        small_df, window=500, retrain_freq=20,
        confidence_levels=[0.05], feature_set="returns",
        hparams=FAST_HPARAMS, epochs=20, patience=5,
    )
    assert result.index[0] == small_df.index[500]
    assert result.index[-1] == small_df.index[-1]


def test_rolling_multiple_taus(small_df):
    result = predict_var_rolling(
        small_df, window=500, retrain_freq=20,
        confidence_levels=[0.01, 0.05], feature_set="returns",
        hparams=FAST_HPARAMS, epochs=20, patience=5,
    )
    assert list(result.columns) == [0.01, 0.05]
    assert len(result) == len(small_df) - 500


def test_rolling_values_finite(small_df):
    result = predict_var_rolling(
        small_df, window=500, retrain_freq=20,
        confidence_levels=[0.05], feature_set="returns",
        hparams=FAST_HPARAMS, epochs=20, patience=5,
    )
    assert result.notna().all().all()
    assert np.isfinite(result.values).all()


def test_rolling_har_feature_set(small_df):
    result = predict_var_rolling(
        small_df, window=500, retrain_freq=20,
        confidence_levels=[0.05], feature_set="har",
        hparams=FAST_HPARAMS, epochs=20, patience=5,
    )
    assert len(result) == len(small_df) - 500
    assert result.notna().all().all()


def test_rolling_saves_meta(small_df, tmp_path):
    predict_var_rolling(
        small_df, window=500, retrain_freq=20,
        confidence_levels=[0.05], feature_set="returns",
        hparams=FAST_HPARAMS, epochs=20, patience=5,
        save_name="test_lstm", results_dir=str(tmp_path),
    )
    meta_path = tmp_path / "test_lstm_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["window"] == 500
    assert meta["feature_set"] == "returns"
    assert meta["hparams"]["hidden_size"] == 16


def test_rolling_saves_checkpoint(small_df, tmp_path):
    predict_var_rolling(
        small_df, window=500, retrain_freq=20,
        confidence_levels=[0.05], feature_set="returns",
        hparams=FAST_HPARAMS, epochs=20, patience=5,
        save_name="test_lstm", results_dir=str(tmp_path),
    )
    ckpt_dir = tmp_path / "checkpoints"
    assert ckpt_dir.exists()
    assert (ckpt_dir / "test_lstm_tau0.05.pt").exists()


def test_rolling_garch_returns_feature_set(small_df):
    """garch_returns feature set should work end-to-end."""
    result = predict_var_rolling(
        small_df, window=500, retrain_freq=20,
        confidence_levels=[0.05], feature_set="garch_returns",
        hparams=FAST_HPARAMS, epochs=20, patience=5,
    )
    assert len(result) == len(small_df) - 500
    assert result.notna().all().all()
    assert np.isfinite(result.values).all()


# -- Task 5: _grid_search tests --------------------------------------------

from models.lstm import _grid_search


def _make_seq_data(n_samples=100, seq_len=10, n_features=1):
    np.random.seed(42)
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32) * 0.01
    return X, y


def test_grid_search_returns_best_params():
    X, y = _make_seq_data(n_samples=80, seq_len=20, n_features=1)
    best_params, best_loss, best_ep, log = _grid_search(
        X, y, tau=0.05,
        hidden_grid=[16], fc_grid=[8], seq_len_grid=[10, 20],
        dropout_grid=[0.0], lr_grid=[1e-3], wd_grid=[0.0],
        epochs=20, patience=5, batch_size=32,
    )
    assert "hidden_size" in best_params
    assert "seq_len" in best_params
    assert best_params["hidden_size"] == 16
    assert best_params["seq_len"] in [10, 20]
    assert best_loss < float("inf")
    assert best_ep > 0


def test_grid_search_log_length():
    X, y = _make_seq_data(n_samples=60, seq_len=20, n_features=1)
    _, _, _, log = _grid_search(
        X, y, tau=0.05,
        hidden_grid=[16], fc_grid=[8],
        seq_len_grid=[20], dropout_grid=[0.0],
        lr_grid=[1e-3, 1e-2], wd_grid=[0.0],
        epochs=10, patience=5, batch_size=32,
    )
    # 1 * 1 * 1 * 1 * 2 * 1 = 2 combinations
    assert len(log) == 2
    assert all("val_loss" in entry for entry in log)


def test_grid_search_seq_len_truncation():
    """Smaller seq_len should truncate X properly."""
    X, y = _make_seq_data(n_samples=80, seq_len=20, n_features=1)
    best_params, _, _, log = _grid_search(
        X, y, tau=0.05,
        hidden_grid=[16], fc_grid=[8],
        seq_len_grid=[10], dropout_grid=[0.0],
        lr_grid=[1e-3], wd_grid=[0.0],
        epochs=10, patience=5, batch_size=32,
    )
    assert best_params["seq_len"] == 10
    assert len(log) == 1
