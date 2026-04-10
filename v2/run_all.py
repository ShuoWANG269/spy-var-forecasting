"""Unified runner for all VaR models.

Usage:
    python run_all.py                      # Run all groups
    python run_all.py --group traditional   # 250-day traditional methods
    python run_all.py --group garch500      # 500-day GARCH + QR
    python run_all.py --group mlp           # MLP (fixed window + 500-day rolling)
    python run_all.py --group lstm          # LSTM 500-day (3 feature sets + tuned)
    python run_all.py --group search        # LSTM hyperparameter search
"""

import argparse
import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import pandas as pd

from models import garch, historical, lstm, nn, parametric, quantile_reg
from utils.data import load_data

CLS = [0.01, 0.05, 0.10]
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "spy_data.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# MLP fixed hyperparams from Step 4 dense grid search
MLP_FIXED_PARAMS = {
    0.01: {"n_layers": 1, "n_units": 1, "lr": 0.0001, "dropout": 0.0, "weight_decay": 0.0001},
    0.05: {"n_layers": 1, "n_units": 4, "lr": 0.01, "dropout": 0.0, "weight_decay": 0.0001},
    0.10: {"n_layers": 1, "n_units": 1, "lr": 0.01, "dropout": 0.0, "weight_decay": 0.0001},
}

# LSTM default hyperparams (GARCHNet literature)
LSTM_DEFAULT_HPARAMS = {
    "hidden_size": 64, "fc_units": 32, "dropout": 0.2,
    "lr": 3e-4, "weight_decay": 0.0,
}


def _save(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    df.to_csv(path)
    print(f"  Saved {df.shape} -> {path}")


def run_traditional(df: pd.DataFrame) -> None:
    """250-day window: historical, parametric, GARCH, QR."""
    configs = [
        ("historical", lambda: historical.predict_var(df, 250, CLS)),
        ("historical_weighted", lambda: historical.predict_var_weighted(df, 250, CLS)),
        ("parametric_normal", lambda: parametric.predict_var(df, 250, CLS, dist="normal")),
        ("parametric_t", lambda: parametric.predict_var(df, 250, CLS, dist="t")),
        ("garch_normal", lambda: garch.predict_var(df, 250, CLS, "GARCH", "normal")),
        ("garch_t", lambda: garch.predict_var(df, 250, CLS, "GARCH", "t")),
        ("gjr_garch_t", lambda: garch.predict_var(df, 250, CLS, "GJR", "t")),
        ("quantile_reg", lambda: quantile_reg.predict_var(df, 250, CLS)),
    ]
    for name, fn in configs:
        print(f"Running {name}...")
        _save(fn(), name)


def run_garch500(df: pd.DataFrame) -> pd.Series | None:
    """500-day window: GARCH variants + QR. Returns garch_normal sigma."""
    garch_sigma = None
    for name, mtype, dist in [
        ("garch_normal_500", "GARCH", "normal"),
        ("garch_t_500", "GARCH", "t"),
        ("gjr_garch_t_500", "GJR", "t"),
    ]:
        print(f"Running {name}...")
        var_df, sigma_s = garch.predict_var(
            df, 500, CLS, mtype, dist, return_sigma=True,
        )
        _save(var_df, name)
        if name == "garch_normal_500":
            garch_sigma = sigma_s

    print("Running quantile_reg_500...")
    _save(quantile_reg.predict_var(df, 500, CLS), "quantile_reg_500")

    return garch_sigma


def run_mlp(df: pd.DataFrame, garch_sigma: pd.Series | None = None) -> None:
    """MLP: fixed window experiments + 500-day rolling."""
    # Fixed window experiments
    for fs, name in [
        ("garch", "mlp_garch"),
        ("har", "mlp_har"),
        ("riskmetrics", "mlp_riskmetrics"),
        ("caviar_sv", "mlp_caviar_sv"),
        ("caviar_asv", "mlp_caviar_asv"),
    ]:
        print(f"Running {name}...")
        _save(nn.predict_var(df, CLS, feature_set=fs, save_name=name,
                             results_dir=RESULTS_DIR), name)

    # 500-day rolling
    if garch_sigma is None:
        print("Computing garch_normal_500 sigma for MLP rolling...")
        _, garch_sigma = garch.predict_var(
            df, 500, CLS, "GARCH", "normal", return_sigma=True,
        )

    print("Running mlp_garch_500...")
    _save(
        nn.predict_var_rolling(
            df, sigma_series=garch_sigma, window=500, retrain_freq=20,
            confidence_levels=CLS, fixed_params=MLP_FIXED_PARAMS,
            save_name="mlp_garch_500", results_dir=RESULTS_DIR,
        ),
        "mlp_garch_500",
    )


def run_lstm(df: pd.DataFrame) -> None:
    """LSTM 500-day rolling: 3 feature sets with default hparams + tuned."""
    for fs in ["returns", "har", "garch_returns"]:
        name = f"lstm_{fs}_500"
        print(f"Running {name}...")
        result = lstm.predict_var_rolling(
            df, window=500, retrain_freq=20,
            confidence_levels=CLS, feature_set=fs,
            hparams=LSTM_DEFAULT_HPARAMS, seq_len=20,
            epochs=300, patience=10, batch_size=32,
            save_name=name, results_dir=RESULTS_DIR,
        )
        _save(result, name)

    # Tuned run (per-tau best params from search)
    save_name = "lstm_returns_tuned_500"
    print(f"Running {save_name}...")
    best_params_per_tau = {
        0.01: {"hidden_size": 100, "fc_units": 16, "seq_len": 10,
                "dropout": 0.0, "lr": 0.01, "weight_decay": 0.0},
        0.05: {"hidden_size": 32, "fc_units": 32, "seq_len": 10,
                "dropout": 0.0, "lr": 0.01, "weight_decay": 0.0},
        0.10: {"hidden_size": 100, "fc_units": 64, "seq_len": 20,
                "dropout": 0.2, "lr": 0.01, "weight_decay": 0.0},
    }
    results_dfs = []
    for tau in CLS:
        p = best_params_per_tau[tau]
        hparams = {k: p[k] for k in ["hidden_size", "fc_units", "dropout", "lr", "weight_decay"]}
        result = lstm.predict_var_rolling(
            df, window=500, retrain_freq=20,
            confidence_levels=[tau], feature_set="returns",
            hparams=hparams, seq_len=p["seq_len"],
            epochs=300, patience=10, batch_size=32,
        )
        results_dfs.append(result)
    merged = pd.concat(results_dfs, axis=1)[CLS]
    _save(merged, save_name)

    # Save metadata for reproducibility
    meta = {
        "feature_set": "returns",
        "window": 500,
        "retrain_freq": 20,
        "epochs": 300,
        "patience": 10,
        "batch_size": 32,
        "best_params_per_tau": {str(t): best_params_per_tau[t] for t in CLS},
    }
    meta_path = os.path.join(RESULTS_DIR, f"{save_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Saved meta -> {meta_path}")


def run_search(df: pd.DataFrame) -> None:
    """LSTM hyperparameter grid search on first 500 days."""
    from models.lstm import _build_features, _build_sequences, _grid_search

    train_slice = df.iloc[:500]
    feat_df, cols = _build_features(train_slice, feature_set="returns")
    X_all, y_all, _ = _build_sequences(feat_df, cols, seq_len=20)
    print(f"Grid search data: X={X_all.shape}, y={y_all.shape}")

    all_logs = []
    best_per_tau = {}
    for tau in CLS:
        print(f"Grid search for tau={tau}...")
        t0 = time.time()
        best_params, best_loss, best_ep, logs = _grid_search(
            X_all, y_all, tau,
            hidden_grid=[32, 64, 100], fc_grid=[16, 32, 64],
            seq_len_grid=[10, 20], dropout_grid=[0.0, 0.2],
            lr_grid=[3e-4, 1e-3, 1e-2], wd_grid=[0.0, 1e-4],
            epochs=300, patience=10, batch_size=32,
        )
        all_logs.extend(logs)
        best_per_tau[tau] = best_params
        print(f"  Best: {best_params}, val_loss={best_loss:.8f}, time={time.time()-t0:.1f}s")

    pd.DataFrame(all_logs).to_csv(
        os.path.join(RESULTS_DIR, "lstm_returns_500_search_log.csv"), index=False
    )
    with open(os.path.join(RESULTS_DIR, "lstm_returns_500_search_meta.json"), "w") as f:
        json.dump({
            "feature_set": "returns", "search_window": 500,
            "best_params_per_tau": {str(t): best_per_tau[t] for t in CLS},
        }, f, indent=2, ensure_ascii=False)
    print("Search complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VaR models")
    parser.add_argument(
        "--group",
        choices=["traditional", "garch500", "mlp", "lstm", "search"],
        default=None,
        help="Run a specific model group (default: run all)",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Data shape: {df.shape}, from {df.index[0].date()} to {df.index[-1].date()}")

    if args.group is None:
        run_traditional(df)
        garch_sigma = run_garch500(df)
        run_mlp(df, garch_sigma)
        run_lstm(df)
    elif args.group == "traditional":
        run_traditional(df)
    elif args.group == "garch500":
        run_garch500(df)
    elif args.group == "mlp":
        run_mlp(df)
    elif args.group == "lstm":
        run_lstm(df)
    elif args.group == "search":
        run_search(df)

    print("Done.")


if __name__ == "__main__":
    main()
