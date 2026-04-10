"""Generate visualization plots for VaR forecasting results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
DATA = Path(__file__).parent / "spy_data.csv"

# Final 4 models
MODELS = {
    "GJR-GARCH-t": "gjr_garch_t_500.csv",
    "Quantile Regression": "quantile_reg_500.csv",
    "MLP": "mlp_garch_500.csv",
    "LSTM": "lstm_returns_tuned_500.csv",
}

COLORS = {
    "GJR-GARCH-t": "#1f77b4",
    "Quantile Regression": "#ff7f0e",
    "MLP": "#2ca02c",
    "LSTM": "#d62728",
}

TAU_LABELS = {"0.01": "1%", "0.05": "5%", "0.1": "10%"}


def load_data():
    """Load SPY returns and model predictions."""
    spy = pd.read_csv(DATA, index_col=0, parse_dates=True)
    preds = {}
    for name, fname in MODELS.items():
        df = pd.read_csv(RESULTS / fname, index_col=0, parse_dates=True)
        df.columns = [str(c) for c in df.columns]
        preds[name] = df
    return spy, preds


def _robust_ylim(series_list, margin=0.15):
    """Compute y-axis limits using percentiles to ignore extremes."""
    all_vals = pd.concat([s.dropna() for s in series_list])
    lo = all_vals.quantile(0.003)
    hi = all_vals.quantile(0.997)
    span = hi - lo
    return lo - span * margin, hi + span * margin


# ── Figure 1: EDA ──────────────────────────────────────────────────────

def plot_eda(spy):
    """EDA: returns time series + distribution histogram."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [2, 1]})

    # Panel A: time series
    ax = axes[0]
    ax.plot(spy.index, spy["log_ret"], color="steelblue", linewidth=0.4, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.3, alpha=0.5)
    # Mark 2008 crisis
    crisis_start = pd.Timestamp("2007-07-01")
    crisis_end = pd.Timestamp("2009-06-30")
    ax.axvspan(crisis_start, crisis_end, color="red", alpha=0.08, label="2007-2009 Financial Crisis")
    ax.set_title("SPY Daily Log Returns", fontsize=14)
    ax.set_ylabel("Log Return")
    ax.legend(loc="lower left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel B: histogram + normal overlay
    ax = axes[1]
    ret = spy["log_ret"].dropna()
    ax.hist(ret, bins=150, density=True, color="steelblue", alpha=0.6, edgecolor="none")
    # Normal overlay
    x_range = np.linspace(ret.min(), ret.max(), 500)
    from scipy.stats import norm
    mu, sigma = ret.mean(), ret.std()
    ax.plot(x_range, norm.pdf(x_range, mu, sigma), "r-", linewidth=1.2, label="Normal fit")
    ax.set_title("Return Distribution (heavy tails visible)", fontsize=14)
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)

    # Add descriptive stats as text box
    skew = ret.skew()
    kurt = ret.kurtosis()
    stats_text = (f"N = {len(ret)}\n"
                  f"Mean = {mu:.5f}\n"
                  f"Std = {sigma:.4f}\n"
                  f"Skew = {skew:.2f}\n"
                  f"Kurtosis = {kurt:.2f}")
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7))

    plt.tight_layout()
    out = RESULTS / "eda_returns.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2: Violation rates bar chart ────────────────────────────────

def plot_violation_rates(spy, preds):
    """Bar chart comparing violation rates across models and taus."""
    taus = ["0.01", "0.05", "0.1"]
    theoretical = [1.0, 5.0, 10.0]
    model_names = list(preds.keys())

    vr = {}
    for name, df in preds.items():
        rates = []
        for tau in taus:
            common = df.index.intersection(spy.index)
            ret = spy.loc[common, "log_ret"]
            pred = df.loc[common, tau]
            rates.append((ret < pred).mean() * 100)
        vr[name] = rates

    x = np.arange(len(taus))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, name in enumerate(model_names):
        bars = ax.bar(x + i * width, vr[name], width, label=name, color=COLORS[name], alpha=0.85)
        for bar, val in zip(bars, vr[name]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    # Theoretical markers with labels
    for j, th in enumerate(theoretical):
        line = ax.hlines(th, x[j] - 0.1, x[j] + len(model_names) * width,
                         colors="black", linestyles="--", linewidth=1, alpha=0.6)
        ax.text(x[j] + len(model_names) * width + 0.02, th,
                f"  {th:.0f}%", va="center", fontsize=8, alpha=0.6)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"τ = {TAU_LABELS[t]}" for t in taus])
    ax.set_ylabel("Violation Rate (%)")
    ax.set_title("Violation Rates vs Theoretical Values", fontsize=14)
    ax.legend(loc="upper left")
    plt.tight_layout()

    out = RESULTS / "violation_rates.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: Per-model VaR time series ────────────────────────────────

def plot_var_timeseries(spy, preds, tau="0.05"):
    """Plot VaR predictions vs actual returns for all models at given tau."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(
        f"VaR Predictions vs Actual Returns (τ = {TAU_LABELS[tau]})",
        fontsize=16, y=0.98,
    )

    for ax, (name, df) in zip(axes, preds.items()):
        dates = df.index
        returns = spy.loc[dates, "log_ret"]
        var_pred = df[tau]

        # Violations
        violations = returns < var_pred
        vr = violations.mean() * 100

        ax.plot(dates, returns, color="gray", alpha=0.35, linewidth=0.5, label="Log Returns")
        ax.plot(dates, var_pred, color=COLORS[name], linewidth=1.2, label=f"VaR ({name})")
        ax.scatter(
            dates[violations], returns[violations],
            color="red", s=8, alpha=0.6, zorder=5, label=f"Violations ({vr:.2f}%)",
        )

        ylo, yhi = _robust_ylim([returns, var_pred])
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel("Return")
        ax.legend(loc="lower left", fontsize=9)
        ax.set_title(name, fontsize=12, loc="left")
        ax.axhline(0, color="black", linewidth=0.3, alpha=0.5)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = RESULTS / f"var_timeseries_tau{tau}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 4: VaR overlay (tau=1%) ────────────────────────────────────

def plot_var_overlay(spy, preds, tau="0.01"):
    """Overlay all 4 models' VaR on one plot for direct comparison."""
    common_dates = preds["GJR-GARCH-t"].index
    for df in preds.values():
        common_dates = common_dates.intersection(df.index)
    common_dates = common_dates.sort_values()

    fig, ax = plt.subplots(figsize=(14, 6))
    returns = spy.loc[common_dates, "log_ret"]
    ax.plot(common_dates, returns, color="gray", alpha=0.4, linewidth=0.5, label="Log Returns")

    all_series = [returns]
    non_lstm_series = [returns]
    for name, df in preds.items():
        s = df.loc[common_dates, tau]
        ax.plot(common_dates, s, color=COLORS[name], linewidth=0.8, alpha=0.85, label=name)
        all_series.append(s)
        if name != "LSTM":
            non_lstm_series.append(s)

    # Use non-LSTM series for y-axis to keep other models distinguishable
    ylo, yhi = _robust_ylim(non_lstm_series, margin=0.3)
    ax.set_ylim(ylo, yhi)

    ax.set_title(f"VaR Comparison — All Models (τ = {TAU_LABELS[tau]})", fontsize=14)
    ax.set_ylabel("Return / VaR")
    ax.set_xlabel("Date")
    ax.axhline(0, color="black", linewidth=0.3, alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()

    out = RESULTS / f"var_overlay_tau{tau}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 5: Quantile crossing (4 models) ────────────────────────────

def plot_quantile_crossing(preds):
    """Visualize quantile crossing for all 4 models."""
    model_order = ["GJR-GARCH-t", "Quantile Regression", "MLP", "LSTM"]
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    fig.suptitle("Quantile Crossing Analysis", fontsize=16, y=0.98)

    for ax, name in zip(axes, model_order):
        df = preds[name]
        dates = df.index
        q01 = df["0.01"].values
        q05 = df["0.05"].values
        q10 = df["0.1"].values

        cross_01_05 = q01 > q05
        cross_05_10 = q05 > q10
        any_cross = cross_01_05 | cross_05_10

        ax.plot(dates, q01, color="#d62728", linewidth=0.6, alpha=0.8, label="τ=1%")
        ax.plot(dates, q05, color="#ff7f0e", linewidth=0.6, alpha=0.8, label="τ=5%")
        ax.plot(dates, q10, color="#2ca02c", linewidth=0.6, alpha=0.8, label="τ=10%")

        ylo, yhi = _robust_ylim([pd.Series(q01), pd.Series(q05), pd.Series(q10)])
        ax.set_ylim(ylo, yhi)

        # Shade crossing regions
        ax.fill_between(
            dates, ylo, yhi,
            where=any_cross, color="red", alpha=0.12,
            label=f"Crossing ({any_cross.mean()*100:.1f}%)",
        )

        n_cross = any_cross.sum()
        total = len(dates)
        ax.set_title(
            f"{name} — {n_cross}/{total} days ({n_cross/total*100:.1f}%) with crossing",
            fontsize=12, loc="left",
        )
        ax.set_ylabel("VaR")
        ax.legend(loc="lower left", fontsize=9)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = RESULTS / "quantile_crossing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    spy, preds = load_data()

    # Fig 1: EDA (§2)
    plot_eda(spy)

    # Fig 2: Violation rates bar chart (§5.1)
    plot_violation_rates(spy, preds)

    # Fig 3: Per-model VaR time series, tau=5% (§5.2)
    plot_var_timeseries(spy, preds, tau="0.05")

    # Fig 4: VaR overlay, tau=1% (§5.2, LSTM extreme discussion)
    plot_var_overlay(spy, preds, tau="0.01")

    # Fig 5: Quantile crossing, all 4 models (§5.3)
    plot_quantile_crossing(preds)

    print("\nAll plots generated.")
