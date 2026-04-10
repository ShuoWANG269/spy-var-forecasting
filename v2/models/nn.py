"""MLP Deep Quantile Regression for VaR forecasting.

Implements the deep quantile estimator from Chronopoulos et al. (2024),
JFE 22(3), 636-669. Each quantile tau is trained as an independent network.
"""

import itertools
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.data import (
    build_caviar_asv_features,
    build_caviar_sv_features,
    build_garch_features,
    build_har_features,
    build_riskmetrics_features,
)
from utils.training import SEED, pinball_loss, train_single_quantile


def _build_mlp(
    input_dim: int, n_layers: int, n_units: int, dropout: float
) -> nn.Sequential:
    """Build feed-forward MLP with ReLU hidden layers and linear output."""
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, n_units))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = n_units
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def _grid_search(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    layers_grid: list[int] = [1, 5, 10],
    units_grid: list[int] = [1, 5, 10],
    lr_grid: list[float] = [0.01, 0.001, 0.0001],
    dropout_grid: list[float] = [0.0, 0.2],
    wd_grid: list[float] = [0.01, 0.001, 0.0001],
    epochs: int = 5000,
    patience: int = 10,
    batch_size: int = 32,
) -> tuple[dict, float, int, list[dict]]:
    """Grid search over hyperparameters for a single quantile.

    Returns (best_params_dict, best_val_loss, best_epochs, search_log).
    best_epochs is the actual epoch count of the best config (for retraining).
    search_log is a list of dicts, one per config evaluated.
    """
    best_loss = float("inf")
    best_params: dict = {}
    best_epochs: int = 0
    search_log: list[dict] = []

    for L, J, lr, do, wd in itertools.product(
        layers_grid, units_grid, lr_grid, dropout_grid, wd_grid
    ):
        t0 = time.time()
        factory = lambda _L=L, _J=J, _do=do: _build_mlp(X.shape[1], _L, _J, _do)
        model, epoch_count, val_loss = train_single_quantile(
            X, y, tau, model_factory=factory,
            lr=lr, weight_decay=wd,
            epochs=epochs, patience=patience, batch_size=batch_size,
        )
        elapsed = time.time() - t0

        search_log.append({
            "tau": tau,
            "n_layers": L,
            "n_units": J,
            "lr": lr,
            "dropout": do,
            "weight_decay": wd,
            "val_loss": val_loss,
            "train_epochs": epoch_count,
            "time_sec": round(elapsed, 3),
        })

        if not np.isnan(val_loss) and val_loss < best_loss:
            best_loss = val_loss
            best_epochs = epoch_count
            best_params = {
                "n_layers": L,
                "n_units": J,
                "lr": lr,
                "dropout": do,
                "weight_decay": wd,
            }

    if not best_params:
        best_params = {
            "n_layers": layers_grid[0],
            "n_units": units_grid[0],
            "lr": lr_grid[0],
            "dropout": dropout_grid[0],
            "weight_decay": wd_grid[0],
        }
        best_epochs = epochs

    return best_params, best_loss, best_epochs, search_log


def _predict_caviar_recursive(
    model: nn.Sequential,
    returns: np.ndarray,
    var_init: float,
    spec: str,
    r_prev_init: float | None = None,
) -> np.ndarray:
    """Recursive CAViaR prediction.

    At each step t, the model input includes VaR_{t-1} from previous prediction.
    spec='sv': input = (VaR_{t-1}, |r_{t-1}|)
    spec='asv': input = (VaR_{t-1}, r^+_{t-1}, r^-_{t-1})
    r_prev_init: return at t=-1 (last training return). If None, uses returns[0].
    """
    model.eval()
    n = len(returns)
    preds = np.empty(n, dtype=np.float32)
    var_prev = var_init

    with torch.no_grad():
        for t in range(n):
            if t > 0:
                r_prev = returns[t - 1]
            elif r_prev_init is not None:
                r_prev = r_prev_init
            else:
                r_prev = returns[0]
            if spec == "sv":
                x = torch.tensor([[var_prev, abs(r_prev)]], dtype=torch.float32)
            else:  # asv
                x = torch.tensor(
                    [[var_prev, max(r_prev, 0), -min(r_prev, 0)]],
                    dtype=torch.float32,
                )
            var_t = model(x).item()
            preds[t] = var_t
            var_prev = var_t

    return preds


def _save_experiment(
    save_name: str,
    feature_set: str,
    confidence_levels: list[float],
    search_space: dict,
    quantile_results: dict,
    all_search_logs: list[dict],
    train_info: dict,
    test_info: dict,
    total_time: float,
    results_dir: str = "results",
) -> None:
    """Persist experiment metadata, search logs, and checkpoints."""
    # 1. Meta JSON
    meta = {
        "feature_set": feature_set,
        "seed": SEED,
        "n_train": train_info["n_train"],
        "n_test": test_info["n_test"],
        "train_date_range": train_info["date_range"],
        "test_date_range": test_info["date_range"],
        "search_space": search_space,
        "quantiles": quantile_results,
        "total_time_sec": round(total_time, 2),
    }
    meta_path = os.path.join(results_dir, f"{save_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 2. Search log CSV
    if all_search_logs:
        log_df = pd.DataFrame(all_search_logs)
        log_path = os.path.join(results_dir, f"{save_name}_search_log.csv")
        log_df.to_csv(log_path, index=False)

    # 3. Checkpoints already saved in predict_var loop


_HAR_COLS = ["log_ret", "C_d", "C_w", "C_m", "J_d"]


def predict_var(
    df: pd.DataFrame,
    confidence_levels: list[float] = [0.01, 0.05, 0.10],
    feature_set: str = "har",
    layers_grid: list[int] = [1, 5, 10],
    units_grid: list[int] = [1, 5, 10],
    lr_grid: list[float] = [0.01, 0.001, 0.0001],
    dropout_grid: list[float] = [0.0, 0.2],
    wd_grid: list[float] = [0.01, 0.001, 0.0001],
    epochs: int = 5000,
    patience: int = 10,
    batch_size: int = 32,
    save_name: str | None = None,
    results_dir: str = "results",
) -> pd.DataFrame:
    """Deep quantile MLP VaR prediction.

    Supports 5 feature sets: 'har', 'garch', 'riskmetrics', 'caviar_sv', 'caviar_asv'.
    Each tau is trained independently with grid search over hyperparameters.
    Uses fixed 80/20 train/test split.

    If save_name is provided, saves experiment metadata (JSON), search log (CSV),
    and model checkpoints (.pt) to results_dir.
    """
    valid_sets = {"har", "garch", "riskmetrics", "caviar_sv", "caviar_asv"}
    if feature_set not in valid_sets:
        raise ValueError(f"Unknown feature_set: {feature_set}. Must be one of {valid_sets}")

    total_t0 = time.time()
    is_caviar = feature_set in ("caviar_sv", "caviar_asv")

    # Checkpoint dir
    ckpt_dir = None
    if save_name:
        ckpt_dir = os.path.join(results_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

    # --- Prepare features (non-CAViaR) ---
    # Compute 80/20 split index first, then build features with fit_end
    # to avoid look-ahead bias in GARCH parameter estimation.
    if feature_set == "har":
        feat_df = build_har_features(df)
        feature_cols = _HAR_COLS
    elif feature_set == "garch":
        n_total = len(df)
        fit_end = int(n_total * 0.8)
        feat_df = build_garch_features(df, fit_end=fit_end)
        feature_cols = ["sigma"]
    elif feature_set == "riskmetrics":
        feat_df = build_riskmetrics_features(df)
        feature_cols = ["sigma"]
    else:
        feat_df = None
        feature_cols = None

    # --- Fixed window 80/20 split ---
    if not is_caviar:
        n = len(feat_df)
        n_train = int(n * 0.8)
        train_df = feat_df.iloc[:n_train]
        test_df = feat_df.iloc[n_train:]

        X_full = train_df[feature_cols].values.astype(np.float32)
        y_full = train_df["log_ret"].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        test_dates = test_df.index

    results = {cl: [] for cl in confidence_levels}

    # Collect experiment metadata
    quantile_results = {}
    all_search_logs: list[dict] = []
    train_info: dict = {}
    test_info: dict = {}

    for tau in confidence_levels:
        tau_t0 = time.time()

        if is_caviar:
            caviar_spec = "sv" if feature_set == "caviar_sv" else "asv"
            # Compute fit_end on raw df to avoid look-ahead bias
            caviar_fit_end = int(len(df) * 0.8)
            if feature_set == "caviar_sv":
                feat_df_tau = build_caviar_sv_features(df, tau=tau, fit_end=caviar_fit_end)
                feature_cols_tau = ["var_lag", "abs_ret"]
            else:
                feat_df_tau = build_caviar_asv_features(df, tau=tau, fit_end=caviar_fit_end)
                feature_cols_tau = ["var_lag", "ret_pos", "ret_neg"]

            n = len(feat_df_tau)
            n_train = int(n * 0.8)
            train_df_tau = feat_df_tau.iloc[:n_train]
            test_df_tau = feat_df_tau.iloc[n_train:]

            X_train_tau = train_df_tau[feature_cols_tau].values.astype(np.float32)
            y_train_tau = train_df_tau["log_ret"].values.astype(np.float32)
            test_dates = test_df_tau.index

            # Grid search
            search_t0 = time.time()
            best_params, best_val_loss, best_ep, search_log = _grid_search(
                X_train_tau, y_train_tau, tau,
                layers_grid=layers_grid, units_grid=units_grid,
                lr_grid=lr_grid, dropout_grid=dropout_grid, wd_grid=wd_grid,
                epochs=epochs, patience=patience, batch_size=batch_size,
            )
            search_time = time.time() - search_t0
            all_search_logs.extend(search_log)

            # Retrain on full training set with fixed epochs
            retrain_t0 = time.time()
            factory = lambda: _build_mlp(X_train_tau.shape[1], best_params["n_layers"], best_params["n_units"], best_params["dropout"])
            final_model, final_epochs, _ = train_single_quantile(
                X_train_tau, y_train_tau, tau, model_factory=factory,
                lr=best_params["lr"], weight_decay=best_params["weight_decay"],
                batch_size=batch_size, fixed_epochs=best_ep,
            )
            retrain_time = time.time() - retrain_t0

            # Predict
            var_init = float(train_df_tau["var_lag"].iloc[-1])
            r_prev_init = float(train_df_tau["log_ret"].iloc[-1])
            returns_test = test_df_tau["log_ret"].values.astype(np.float32)
            preds = _predict_caviar_recursive(
                final_model, returns_test, var_init, spec=caviar_spec,
                r_prev_init=r_prev_init,
            )
            results[tau] = preds.tolist()

            # Record info (use last tau's split for meta)
            train_info = {
                "n_train": n_train,
                "date_range": [str(train_df_tau.index[0].date()), str(train_df_tau.index[-1].date())],
            }
            test_info = {
                "n_test": len(test_df_tau),
                "date_range": [str(test_df_tau.index[0].date()), str(test_df_tau.index[-1].date())],
            }

        else:
            # Grid search
            search_t0 = time.time()
            best_params, best_val_loss, best_ep, search_log = _grid_search(
                X_full, y_full, tau,
                layers_grid=layers_grid, units_grid=units_grid,
                lr_grid=lr_grid, dropout_grid=dropout_grid, wd_grid=wd_grid,
                epochs=epochs, patience=patience, batch_size=batch_size,
            )
            search_time = time.time() - search_t0
            all_search_logs.extend(search_log)

            # Retrain on full training set with fixed epochs
            retrain_t0 = time.time()
            factory = lambda: _build_mlp(X_full.shape[1], best_params["n_layers"], best_params["n_units"], best_params["dropout"])
            final_model, final_epochs, _ = train_single_quantile(
                X_full, y_full, tau, model_factory=factory,
                lr=best_params["lr"], weight_decay=best_params["weight_decay"],
                batch_size=batch_size, fixed_epochs=best_ep,
            )
            retrain_time = time.time() - retrain_t0

            # Predict
            final_model.eval()
            with torch.no_grad():
                preds = final_model(
                    torch.from_numpy(X_test)
                ).squeeze(-1).numpy()
            results[tau] = preds.tolist()

            train_info = {
                "n_train": len(train_df),
                "date_range": [str(train_df.index[0].date()), str(train_df.index[-1].date())],
            }
            test_info = {
                "n_test": len(test_df),
                "date_range": [str(test_df.index[0].date()), str(test_df.index[-1].date())],
            }

        # Save checkpoint
        tau_key = str(tau)
        ckpt_path = None
        if ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, f"{save_name}_tau{tau}.pt")
            torch.save(final_model.state_dict(), ckpt_path)

        quantile_results[tau_key] = {
            "best_params": best_params,
            "val_loss": round(best_val_loss, 8),
            "search_time_sec": round(search_time, 2),
            "retrain_time_sec": round(retrain_time, 2),
            "retrain_epochs": final_epochs,
            "checkpoint": ckpt_path,
        }

    total_time = time.time() - total_t0

    # Save experiment metadata
    if save_name:
        search_space = {
            "layers_grid": layers_grid,
            "units_grid": units_grid,
            "lr_grid": lr_grid,
            "dropout_grid": dropout_grid,
            "wd_grid": wd_grid,
            "batch_size": batch_size,
            "max_epochs": epochs,
            "patience": patience,
        }
        _save_experiment(
            save_name=save_name,
            feature_set=feature_set,
            confidence_levels=confidence_levels,
            search_space=search_space,
            quantile_results=quantile_results,
            all_search_logs=all_search_logs,
            train_info=train_info,
            test_info=test_info,
            total_time=total_time,
            results_dir=results_dir,
        )

    return pd.DataFrame(results, index=test_dates)


def predict_var_rolling(
    df: pd.DataFrame,
    sigma_series: pd.Series,
    window: int = 500,
    retrain_freq: int = 20,
    confidence_levels: list[float] = [0.01, 0.05, 0.10],
    fixed_params: dict[float, dict] | None = None,
    epochs: int = 5000,
    patience: int = 10,
    batch_size: int = 32,
    save_name: str | None = None,
    results_dir: str = "results",
) -> pd.DataFrame:
    """Rolling-window MLP VaR prediction with periodic retraining.

    Uses pre-computed GARCH conditional volatility (sigma_series) as the
    sole input feature for prediction. For training within each window,
    fits GARCH on the window to get within-window conditional_volatility.

    Each tau uses fixed hyperparameters from prior grid search.
    """
    from arch import arch_model as _am

    if fixed_params is None:
        raise ValueError("fixed_params is required for rolling prediction")
    for tau in confidence_levels:
        if tau not in fixed_params:
            raise ValueError(f"fixed_params missing for tau={tau}")

    total_t0 = time.time()
    n_total = len(df)
    n_steps = n_total - window

    # Validate sigma_series alignment
    assert len(sigma_series) == n_steps, (
        f"sigma_series length {len(sigma_series)} != expected {n_steps}"
    )
    assert sigma_series.index[0] == df.index[window], (
        f"sigma_series start date {sigma_series.index[0]} != df date {df.index[window]}"
    )

    sigma_arr = sigma_series.values.astype(np.float32)
    pred_dates = sigma_series.index

    results = {tau: np.empty(n_steps, dtype=np.float32) for tau in confidence_levels}

    # Cache GARCH training data per retrain step (shared across taus)
    garch_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for tau in confidence_levels:
        params = fixed_params[tau]
        model = None

        for step in range(n_steps):
            if step % retrain_freq == 0:
                # Reuse GARCH fit across taus for same step
                if step not in garch_cache:
                    train_slice = df.iloc[step : step + window]
                    r_window = train_slice["log_ret"].values
                    r_scaled = r_window * 100

                    # Match predict_var_with_sigma: default mean="Constant"
                    _garch = _am(r_scaled, vol="GARCH", p=1, q=1, dist="Normal")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _res = _garch.fit(disp="off", show_warning=False)
                    window_sigma = _res.conditional_volatility / 100

                    X_train = window_sigma[:-1].reshape(-1, 1).astype(np.float32)
                    y_train = r_window[1:].astype(np.float32)
                    garch_cache[step] = (X_train, y_train)

                X_train, y_train = garch_cache[step]

                factory = lambda: _build_mlp(1, params["n_layers"], params["n_units"], params["dropout"])

                # Early stopping to find best epoch count
                _, best_ep, _ = train_single_quantile(
                    X_train, y_train, tau, model_factory=factory,
                    lr=params["lr"], weight_decay=params["weight_decay"],
                    epochs=epochs, patience=patience, batch_size=batch_size,
                )
                # Retrain on full window with fixed epochs
                model, _, _ = train_single_quantile(
                    X_train, y_train, tau, model_factory=factory,
                    lr=params["lr"], weight_decay=params["weight_decay"],
                    batch_size=batch_size, fixed_epochs=max(best_ep, 1),
                )

            # Predict using pre-computed GARCH conditional_volatility
            model.eval()
            with torch.no_grad():
                x = torch.tensor([[sigma_arr[step]]], dtype=torch.float32)
                results[tau][step] = model(x).item()

    total_time = time.time() - total_t0

    if save_name:
        os.makedirs(results_dir, exist_ok=True)
        meta = {
            "window": window,
            "retrain_freq": retrain_freq,
            "seed": SEED,
            "n_steps": n_steps,
            "pred_date_range": [
                str(pred_dates[0].date()),
                str(pred_dates[-1].date()),
            ],
            "fixed_params": {
                str(tau): fixed_params[tau] for tau in confidence_levels
            },
            "total_time_sec": round(total_time, 2),
        }
        meta_path = os.path.join(results_dir, f"{save_name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return pd.DataFrame(results, index=pred_dates)
