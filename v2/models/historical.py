import numpy as np
import pandas as pd
from collections.abc import Sequence

from utils.data import rolling_windows


DEFAULT_CONFIDENCE_LEVELS = (0.01, 0.05, 0.10)


def predict_var(
    df: pd.DataFrame,
    window: int = 250,
    confidence_levels: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Historical simulation VaR using empirical return quantiles."""
    levels = DEFAULT_CONFIDENCE_LEVELS if confidence_levels is None else confidence_levels
    records = {cl: [] for cl in levels}
    dates = []

    for train, test_date, _ in rolling_windows(df, window):
        dates.append(test_date)
        returns = train["log_ret"].to_numpy()
        for cl in levels:
            records[cl].append(float(np.quantile(returns, cl)))

    return pd.DataFrame(records, index=dates)


def predict_var_weighted(
    df: pd.DataFrame,
    window: int = 250,
    confidence_levels: Sequence[float] | None = None,
    lam: float = 0.99,
) -> pd.DataFrame:
    """Exponentially weighted historical simulation VaR."""
    levels = DEFAULT_CONFIDENCE_LEVELS if confidence_levels is None else confidence_levels
    records = {cl: [] for cl in levels}
    dates = []

    for train, test_date, _ in rolling_windows(df, window):
        dates.append(test_date)
        returns = train["log_ret"].to_numpy()
        n_obs = len(returns)
        raw_weights = np.array([lam ** (n_obs - 1 - idx) for idx in range(n_obs)])
        weights = raw_weights / raw_weights.sum()

        sorted_idx = np.argsort(returns)
        sorted_returns = returns[sorted_idx]
        cumulative_weights = np.cumsum(weights[sorted_idx])

        for cl in levels:
            quantile_idx = np.searchsorted(cumulative_weights, cl)
            records[cl].append(float(sorted_returns[min(quantile_idx, n_obs - 1)]))

    return pd.DataFrame(records, index=dates)
