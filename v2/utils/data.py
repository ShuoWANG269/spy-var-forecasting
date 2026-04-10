import pandas as pd
import numpy as np
from typing import Iterator, Tuple


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    df = df[["log_ret", "rv5", "bv"]].dropna()
    return df


def rolling_windows(
    df: pd.DataFrame, window: int = 250
) -> Iterator[Tuple[pd.DataFrame, pd.Timestamp, float]]:
    for i in range(window, len(df)):
        train = df.iloc[i - window : i]
        test_date = df.index[i]
        test_return = float(df["log_ret"].iloc[i])
        yield train, test_date, test_return


def build_garch_features(
    df: pd.DataFrame, fit_end: int | None = None
) -> pd.DataFrame:
    """Build GARCH(1,1) conditional volatility feature.

    If fit_end is given, fits GARCH only on df.iloc[:fit_end] to avoid
    look-ahead bias, then filters the full series using those parameters.
    Otherwise fits on the full series (legacy behavior for tests).
    Returns a copy of df with an added 'sigma' column.
    """
    import warnings
    from arch import arch_model

    out = df.copy()
    r_full = df["log_ret"].values * 100  # scale for numerical stability

    if fit_end is not None:
        # Fit only on training portion
        r_train = r_full[:fit_end]
        am = arch_model(r_train, vol="GARCH", p=1, q=1, dist="Normal", mean="Zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)
        # Filter full series using trained parameters
        omega = res.params["omega"]
        alpha = res.params["alpha[1]"]
        beta = res.params["beta[1]"]
        sigma2 = np.empty(len(r_full), dtype=np.float64)
        sigma2[0] = omega / (1 - alpha - beta) if (alpha + beta) < 1 else r_full[0] ** 2
        for t in range(1, len(r_full)):
            sigma2[t] = omega + alpha * r_full[t - 1] ** 2 + beta * sigma2[t - 1]
        out["sigma"] = np.sqrt(sigma2) / 100
    else:
        am = arch_model(r_full, vol="GARCH", p=1, q=1, dist="Normal", mean="Zero")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)
        out["sigma"] = res.conditional_volatility / 100

    return out


def build_riskmetrics_features(df: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """Build RiskMetrics EWMA conditional volatility feature.

    sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_{t-1}
    Returns a copy of df with an added 'sigma' column.
    """
    out = df.copy()
    r = df["log_ret"].values
    n = len(r)
    var = np.empty(n, dtype=np.float64)
    var[0] = r[0] ** 2  # initialize with first squared return
    for t in range(1, n):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    out["sigma"] = np.sqrt(var)
    return out


def _linear_caviar_init(
    returns: np.ndarray, tau: float, spec: str,
    fit_end: int | None = None,
) -> np.ndarray:
    """Estimate initial VaR series using linear CAViaR via QuantReg.

    spec: 'sv' for symmetric absolute value, 'asv' for asymmetric slope.
    If fit_end is given, fits QuantReg only on returns[:fit_end] to avoid
    look-ahead bias, then recursively generates VaR for the full sequence.
    Returns VaR array of length len(returns).
    """
    import warnings
    from statsmodels.regression.quantile_regression import (
        IterationLimitWarning,
        QuantReg,
    )

    n = len(returns)
    fit_n = fit_end if fit_end is not None else n
    var_init = np.full(n, np.quantile(returns[:fit_n], tau))

    for iteration in range(5):
        # Fit only on training portion
        y = returns[1:fit_n]
        if spec == "sv":
            X = np.column_stack([
                np.ones(fit_n - 1),
                var_init[:fit_n - 1],
                np.abs(returns[:fit_n - 1]),
            ])
        else:  # asv
            X = np.column_stack([
                np.ones(fit_n - 1),
                var_init[:fit_n - 1],
                np.maximum(returns[:fit_n - 1], 0),
                -np.minimum(returns[:fit_n - 1], 0),
            ])
        model = QuantReg(y, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IterationLimitWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                res = model.fit(q=tau, max_iter=2000, p_tol=1e-6)
            except np.linalg.LinAlgError:
                break  # SVD non-convergence; use current var_init
        var_init[0] = np.quantile(returns[:fit_n], tau)
        # Clamp bounds to prevent divergence
        lo = np.min(returns[:fit_n]) * 3
        hi = np.max(returns[:fit_n]) * 3
        # Recursively generate VaR for FULL sequence using trained params
        for t in range(1, n):
            if spec == "sv":
                x_t = np.array([1.0, var_init[t - 1], abs(returns[t - 1])])
            else:
                x_t = np.array([
                    1.0,
                    var_init[t - 1],
                    max(returns[t - 1], 0),
                    -min(returns[t - 1], 0),
                ])
            var_init[t] = np.clip(float(res.params @ x_t), lo, hi)

    return var_init


def build_caviar_sv_features(
    df: pd.DataFrame, tau: float, fit_end: int | None = None,
) -> pd.DataFrame:
    """Build CAViaR Symmetric Absolute Value features.

    Covariates: (VaR_{t-1}, |r_{t-1}|).
    Uses linear CAViaR to initialize VaR sequence.
    If fit_end is given, fits only on data[:fit_end] to avoid look-ahead bias.
    Returns DataFrame starting from index 1 (first row has no lag).
    Columns added: 'var_lag', 'abs_ret'.
    """
    out = df.copy()
    returns = df["log_ret"].values
    var_seq = _linear_caviar_init(returns, tau, "sv", fit_end=fit_end)

    out["var_lag"] = np.nan
    out["abs_ret"] = np.nan
    out.iloc[1:, out.columns.get_loc("var_lag")] = var_seq[:-1]
    out.iloc[1:, out.columns.get_loc("abs_ret")] = np.abs(returns[:-1])
    return out.dropna(subset=["var_lag", "abs_ret"]).copy()


def build_caviar_asv_features(
    df: pd.DataFrame, tau: float, fit_end: int | None = None,
) -> pd.DataFrame:
    """Build CAViaR Asymmetric Slope Value features.

    Covariates: (VaR_{t-1}, r^+_{t-1}, r^-_{t-1}).
    Uses linear CAViaR to initialize VaR sequence.
    If fit_end is given, fits only on data[:fit_end] to avoid look-ahead bias.
    Returns DataFrame starting from index 1.
    Columns added: 'var_lag', 'ret_pos', 'ret_neg'.
    """
    out = df.copy()
    returns = df["log_ret"].values
    var_seq = _linear_caviar_init(returns, tau, "asv", fit_end=fit_end)

    out["var_lag"] = np.nan
    out["ret_pos"] = np.nan
    out["ret_neg"] = np.nan
    out.iloc[1:, out.columns.get_loc("var_lag")] = var_seq[:-1]
    out.iloc[1:, out.columns.get_loc("ret_pos")] = np.maximum(returns[:-1], 0)
    out.iloc[1:, out.columns.get_loc("ret_neg")] = -np.minimum(returns[:-1], 0)
    return out.dropna(subset=["var_lag", "ret_pos", "ret_neg"]).copy()


def build_har_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["C_d"] = df["bv"]
    out["J_d"] = (df["rv5"] - df["bv"]).clip(lower=0)
    out["C_w"] = out["C_d"].rolling(5).mean()
    out["C_m"] = out["C_d"].rolling(22).mean()
    return out.dropna(subset=["C_w", "C_m", "J_d"]).copy()
