import numpy as np
import pandas as pd
import pytest
import torch

from models.nn import (
    _build_mlp,
    _grid_search,
    _predict_caviar_recursive,
    predict_var,
)
from utils.training import pinball_loss, train_single_quantile


def test_build_mlp_output_shape():
    model = _build_mlp(input_dim=5, n_layers=3, n_units=10, dropout=0.2)
    x = torch.randn(8, 5)
    model.eval()
    out = model(x)
    assert out.shape == (8, 1)


def test_build_mlp_single_layer():
    model = _build_mlp(input_dim=5, n_layers=1, n_units=5, dropout=0.0)
    x = torch.randn(4, 5)
    model.eval()
    out = model(x)
    assert out.shape == (4, 1)


def test_build_mlp_minimal():
    model = _build_mlp(input_dim=1, n_layers=1, n_units=1, dropout=0.0)
    x = torch.randn(2, 1)
    model.eval()
    out = model(x)
    assert out.shape == (2, 1)


def test_pinball_loss_perfect_prediction():
    pred = torch.tensor([[0.5], [1.0]])
    target = torch.tensor([0.5, 1.0])
    loss = pinball_loss(pred, target, tau=0.05)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_pinball_loss_asymmetry():
    pred = torch.tensor([[0.0]])
    target = torch.tensor([1.0])
    loss_low = pinball_loss(pred, target, tau=0.1)
    loss_high = pinball_loss(pred, target, tau=0.9)
    assert loss_high.item() > loss_low.item()


def test_grid_search_returns_best_params():
    np.random.seed(42)
    X = np.random.randn(200, 5).astype(np.float32)
    y = np.random.randn(200).astype(np.float32)
    best_params, best_loss, best_epochs, search_log = _grid_search(
        X, y, tau=0.05,
        layers_grid=[1], units_grid=[5],
        lr_grid=[0.001], dropout_grid=[0.0],
        wd_grid=[0.001],
        epochs=20, patience=5, batch_size=32,
    )
    assert "n_layers" in best_params
    assert "n_units" in best_params
    assert "lr" in best_params
    assert "dropout" in best_params
    assert "weight_decay" in best_params
    assert best_loss < float("inf")
    assert best_epochs > 0
    assert len(search_log) == 1
    assert "val_loss" in search_log[0]
    assert "time_sec" in search_log[0]


def test_grid_search_explores_multiple_configs():
    np.random.seed(42)
    X = np.random.randn(200, 5).astype(np.float32)
    y = np.random.randn(200).astype(np.float32)
    best_params, _, _, search_log = _grid_search(
        X, y, tau=0.05,
        layers_grid=[1, 5], units_grid=[5, 10],
        lr_grid=[0.001], dropout_grid=[0.0],
        wd_grid=[0.001],
        epochs=10, patience=3, batch_size=32,
    )
    assert len(search_log) == 4  # 2 layers x 2 units
    assert best_params["n_layers"] in [1, 5]
    assert best_params["n_units"] in [5, 10]


def test_caviar_recursive_shape():
    np.random.seed(42)
    X = np.random.randn(100, 2).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    factory = lambda: _build_mlp(2, 1, 5, 0.0)
    model, _, _ = train_single_quantile(
        X, y, tau=0.05, model_factory=factory,
        lr=0.001, weight_decay=0.001,
        epochs=20, patience=5, batch_size=32,
    )
    returns_test = np.random.randn(20).astype(np.float32)
    var_init = -0.02
    preds = _predict_caviar_recursive(model, returns_test, var_init, spec="sv")
    assert len(preds) == 20


def test_caviar_recursive_asv_shape():
    np.random.seed(42)
    X = np.random.randn(100, 3).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    factory = lambda: _build_mlp(3, 1, 5, 0.0)
    model, _, _ = train_single_quantile(
        X, y, tau=0.05, model_factory=factory,
        lr=0.001, weight_decay=0.001,
        epochs=20, patience=5, batch_size=32,
    )
    returns_test = np.random.randn(20).astype(np.float32)
    var_init = -0.02
    preds = _predict_caviar_recursive(model, returns_test, var_init, spec="asv")
    assert len(preds) == 20


@pytest.fixture
def small_df():
    np.random.seed(42)
    n = 300
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


def test_predict_var_har_shape(small_df):
    result = predict_var(
        small_df, confidence_levels=CLS, feature_set="har",
        layers_grid=[1], units_grid=[5], lr_grid=[0.001],
        dropout_grid=[0.0], wd_grid=[0.001], epochs=10, patience=3,
    )
    assert result.shape[1] == 3
    assert result.shape[0] > 0
    assert list(result.columns) == CLS


def test_predict_var_garch_shape(small_df):
    result = predict_var(
        small_df, confidence_levels=CLS, feature_set="garch",
        layers_grid=[1], units_grid=[5], lr_grid=[0.001],
        dropout_grid=[0.0], wd_grid=[0.001], epochs=10, patience=3,
    )
    assert result.shape[1] == 3
    assert result.shape[0] > 0


def test_predict_var_riskmetrics_shape(small_df):
    result = predict_var(
        small_df, confidence_levels=CLS, feature_set="riskmetrics",
        layers_grid=[1], units_grid=[5], lr_grid=[0.001],
        dropout_grid=[0.0], wd_grid=[0.001], epochs=10, patience=3,
    )
    assert result.shape[1] == 3
    assert result.shape[0] > 0


def test_predict_var_caviar_sv_shape(small_df):
    result = predict_var(
        small_df, confidence_levels=CLS, feature_set="caviar_sv",
        layers_grid=[1], units_grid=[5], lr_grid=[0.001],
        dropout_grid=[0.0], wd_grid=[0.001], epochs=10, patience=3,
    )
    assert result.shape[1] == 3
    assert result.shape[0] > 0


def test_predict_var_caviar_asv_shape(small_df):
    result = predict_var(
        small_df, confidence_levels=CLS, feature_set="caviar_asv",
        layers_grid=[1], units_grid=[5], lr_grid=[0.001],
        dropout_grid=[0.0], wd_grid=[0.001], epochs=10, patience=3,
    )
    assert result.shape[1] == 3
    assert result.shape[0] > 0


def test_predict_var_invalid_feature_set(small_df):
    with pytest.raises(ValueError, match="Unknown feature_set"):
        predict_var(small_df, feature_set="unknown")
