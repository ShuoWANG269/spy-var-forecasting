import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.data import build_har_features
from utils.training import SEED, pinball_loss, train_single_quantile


class _StandardScaler:
    def fit(self, X: np.ndarray) -> "_StandardScaler":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _build_features(
    train_slice: pd.DataFrame, feature_set: str
) -> tuple[pd.DataFrame, list[str]]:
    """Build feature DataFrame for a training window.

    Args:
        train_slice: Raw df slice with columns [log_ret, rv5, bv].
        feature_set: One of 'returns', 'har', 'garch_returns'.
            GARCH sigma is fit only on train_slice to avoid look-ahead bias.

    Returns:
        (feat_df, feature_cols) — feature DataFrame and list of column names.
    """
    if feature_set == "returns":
        return train_slice.copy(), ["log_ret"]

    if feature_set == "har":
        _HAR_COLS = ["log_ret", "C_d", "C_w", "C_m", "J_d"]
        feat_df = build_har_features(train_slice)
        return feat_df, _HAR_COLS

    if feature_set == "garch_returns":
        from arch import arch_model as _am

        r_scaled = train_slice["log_ret"].values * 100
        garch = _am(r_scaled, vol="GARCH", p=1, q=1, dist="Normal")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = garch.fit(disp="off", show_warning=False)
        sigma = res.conditional_volatility / 100
        feat_df = train_slice.copy()
        feat_df["sigma"] = sigma
        # Cache GARCH params for filter() at non-retrain prediction steps
        feat_df.attrs["_garch_params"] = res.params
        return feat_df, ["log_ret", "sigma"]

    raise ValueError(f"Unknown feature_set: {feature_set}")


def _build_sequences(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, "_StandardScaler"]:
    """Build normalized sequences for LSTM input.

    Returns:
        X: (n_samples, seq_len, n_features) float32, standardized.
        y: (n_samples,) float32 — raw (unscaled) next-day log_ret.
        scaler: fitted _StandardScaler for inverse/prediction use.
    """
    raw = feat_df[feature_cols].values.astype(np.float32)
    scaler = _StandardScaler().fit(raw)
    data = scaler.transform(raw)
    returns = feat_df["log_ret"].values.astype(np.float32)

    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])
        y.append(returns[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler


class _VaRLSTM(nn.Module):
    """1-layer LSTM + FC head for single-quantile VaR prediction.

    Architecture based on GARCHNet (Buczynski & Chlebus, 2023):
    Input -> LSTM(hidden_size) -> FC(fc_units) -> ReLU -> FC(1) -> Output
    """

    def __init__(
        self, input_size: int, hidden_size: int, fc_units: int, dropout: float
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _grid_search(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    hidden_grid: list[int] = [32, 64, 100],
    fc_grid: list[int] = [16, 32, 64],
    seq_len_grid: list[int] = [10, 20],
    dropout_grid: list[float] = [0.0, 0.2],
    lr_grid: list[float] = [3e-4, 1e-3, 1e-2],
    wd_grid: list[float] = [0.0, 1e-4],
    epochs: int = 300,
    patience: int = 10,
    batch_size: int = 32,
) -> tuple[dict, float, int, list[dict]]:
    """Grid search over LSTM hyperparameters for a single quantile.

    Note: seq_len_grid requires that X was built with the LARGEST seq_len
    in the grid. For smaller seq_len values, X is truncated: X[:, -seq_len:, :].

    Returns (best_params, best_val_loss, best_epochs, search_log).
    """
    import itertools

    best_loss = float("inf")
    best_params: dict = {}
    best_epochs: int = 0
    search_log: list[dict] = []

    for hs, fc, sl, do, lr, wd in itertools.product(
        hidden_grid, fc_grid, seq_len_grid, dropout_grid, lr_grid, wd_grid
    ):
        t0 = time.time()
        # Truncate sequences if seq_len < max
        X_sl = X[:, -sl:, :] if sl < X.shape[1] else X

        factory = lambda _hs=hs, _fc=fc, _do=do: _VaRLSTM(
            input_size=X_sl.shape[2], hidden_size=_hs, fc_units=_fc, dropout=_do
        )
        model, epoch_count, val_loss = train_single_quantile(
            X_sl, y, tau, model_factory=factory,
            lr=lr, weight_decay=wd,
            epochs=epochs, patience=patience, batch_size=batch_size,
        )
        elapsed = time.time() - t0

        search_log.append({
            "tau": tau,
            "hidden_size": hs,
            "fc_units": fc,
            "seq_len": sl,
            "dropout": do,
            "lr": lr,
            "weight_decay": wd,
            "val_loss": val_loss,
            "train_epochs": epoch_count,
            "time_sec": round(elapsed, 3),
        })

        if not np.isnan(val_loss) and val_loss < best_loss:
            best_loss = val_loss
            best_epochs = epoch_count
            best_params = {
                "hidden_size": hs,
                "fc_units": fc,
                "seq_len": sl,
                "dropout": do,
                "lr": lr,
                "weight_decay": wd,
            }

    if not best_params:
        best_params = {
            "hidden_size": hidden_grid[0],
            "fc_units": fc_grid[0],
            "seq_len": seq_len_grid[0],
            "dropout": dropout_grid[0],
            "lr": lr_grid[0],
            "weight_decay": wd_grid[0],
        }
        best_epochs = epochs

    return best_params, best_loss, best_epochs, search_log


def predict_var_rolling(
    df: pd.DataFrame,
    window: int = 500,
    retrain_freq: int = 20,
    confidence_levels: list[float] = [0.01, 0.05, 0.10],
    feature_set: str = "returns",
    hparams: dict | None = None,
    seq_len: int = 20,
    epochs: int = 300,
    patience: int = 10,
    batch_size: int = 32,
    save_name: str | None = None,
    results_dir: str = "results",
) -> pd.DataFrame:
    """Rolling-window LSTM VaR prediction with periodic retraining.

    Args:
        df: Raw DataFrame with columns [log_ret, rv5, bv].
        window: Training window size.
        retrain_freq: Retrain every N steps.
        confidence_levels: List of tau values.
        feature_set: One of 'returns', 'har', 'garch_returns'.
        hparams: Dict with keys: hidden_size, fc_units, dropout, lr, weight_decay.
        seq_len: LSTM sequence length.
        epochs: Max epochs for early stopping.
        patience: Early stopping patience.
        batch_size: Training batch size.
        save_name: If provided, save meta JSON and checkpoints.
        results_dir: Directory for saving outputs.
    """
    if hparams is None:
        hparams = {
            "hidden_size": 64, "fc_units": 32, "dropout": 0.2,
            "lr": 3e-4, "weight_decay": 0.0,
        }
    if seq_len >= window:
        raise ValueError("seq_len must be smaller than window")

    total_t0 = time.time()
    n_total = len(df)
    n_steps = n_total - window
    pred_dates = df.index[window:]

    results = {tau: np.empty(n_steps, dtype=np.float32) for tau in confidence_levels}

    # Cache per retrain step: training sequences + scaler + feat_df
    # Shared across taus to avoid redundant feature builds / GARCH fits
    feature_cache: dict[int, tuple[np.ndarray, np.ndarray, "_StandardScaler", list[str], pd.DataFrame]] = {}

    # Track last best_epoch per tau for metadata
    tau_best_epochs: dict[float, int] = {}

    for tau in confidence_levels:
        model = None
        scaler = None
        feature_cols = None
        cached_feat_df = None

        for step in range(n_steps):
            if step % retrain_freq == 0:
                if step not in feature_cache:
                    train_slice = df.iloc[step : step + window]
                    feat_df, cols = _build_features(train_slice, feature_set)
                    X_all, y_all, sc = _build_sequences(feat_df, cols, seq_len)
                    feature_cache[step] = (X_all, y_all, sc, cols, feat_df)

                X_all, y_all, scaler, feature_cols, cached_feat_df = feature_cache[step]

                factory = lambda: _VaRLSTM(
                    input_size=X_all.shape[2],
                    hidden_size=hparams["hidden_size"],
                    fc_units=hparams["fc_units"],
                    dropout=hparams["dropout"],
                )

                # Early stopping to find best_epoch
                _, best_ep, _ = train_single_quantile(
                    X_all, y_all, tau, model_factory=factory,
                    lr=hparams["lr"], weight_decay=hparams["weight_decay"],
                    epochs=epochs, patience=patience, batch_size=batch_size,
                )

                # Retrain with fixed epochs
                model, _, _ = train_single_quantile(
                    X_all, y_all, tau, model_factory=factory,
                    lr=hparams["lr"], weight_decay=hparams["weight_decay"],
                    batch_size=batch_size, fixed_epochs=max(best_ep, 1),
                )
                tau_best_epochs[tau] = best_ep

            # Predict: need last seq_len rows of features, scaled
            # Rebuild feat_df_pred every step so prediction uses current data.
            # For garch_returns, reuse cached GARCH params via filter()
            # to avoid expensive re-estimation while keeping sigma fresh.
            train_slice = df.iloc[step : step + window]
            if feature_set in ("returns", "har"):
                feat_df_pred, _ = _build_features(train_slice, feature_set)
            else:
                # garch_returns: compute sigma with cached GARCH params
                # via fix() — avoids expensive re-estimation
                from arch import arch_model as _am

                r_scaled = train_slice["log_ret"].values * 100
                garch = _am(r_scaled, vol="GARCH", p=1, q=1, dist="Normal")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fixed_res = garch.fix(
                        cached_feat_df.attrs["_garch_params"]
                    )
                sigma = fixed_res.conditional_volatility / 100
                feat_df_pred = train_slice.copy()
                feat_df_pred["sigma"] = sigma

            raw_seq = feat_df_pred[feature_cols].values[-seq_len:].astype(np.float32)
            scaled_seq = scaler.transform(raw_seq)
            x_tensor = torch.from_numpy(scaled_seq).unsqueeze(0)

            model.eval()
            with torch.no_grad():
                results[tau][step] = model(x_tensor).squeeze().item()

        # Save checkpoint after all steps for this tau
        if save_name:
            ckpt_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{save_name}_tau{tau}.pt")
            torch.save(model.state_dict(), ckpt_path)

    total_time = time.time() - total_t0

    if save_name:
        os.makedirs(results_dir, exist_ok=True)
        meta = {
            "window": window,
            "retrain_freq": retrain_freq,
            "seq_len": seq_len,
            "feature_set": feature_set,
            "hparams": hparams,
            "seed": SEED,
            "n_steps": n_steps,
            "pred_date_range": [
                str(pred_dates[0].date()),
                str(pred_dates[-1].date()),
            ],
            "last_best_epoch_per_tau": {
                str(t): tau_best_epochs.get(t, 0) for t in confidence_levels
            },
            "total_time_sec": round(total_time, 2),
        }
        meta_path = os.path.join(results_dir, f"{save_name}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    return pd.DataFrame(results, index=pred_dates)
