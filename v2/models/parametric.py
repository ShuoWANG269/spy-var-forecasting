import pandas as pd
from collections.abc import Sequence
from scipy import stats

from utils.data import rolling_windows


def predict_var(
    df: pd.DataFrame,
    window: int = 250,
    confidence_levels: Sequence[float] | None = None,
    dist: str = "normal",
) -> pd.DataFrame:
    """
    Parametric VaR.
    dist='normal': normal MLE; dist='t': Student-t MLE (captures fat tails).
    """
    if dist not in {"normal", "t"}:
        raise ValueError(f"Unsupported dist: {dist}")

    levels = (0.01, 0.05, 0.10) if confidence_levels is None else confidence_levels
    records = {cl: [] for cl in levels}
    dates = []
    for train, test_date, _ in rolling_windows(df, window):
        dates.append(test_date)
        r = train["log_ret"].values

        if dist == "normal":
            mu = r.mean()
            sigma = r.std(ddof=1)
        else:
            df_fit, loc, scale = stats.t.fit(r)

        for cl in levels:
            if dist == "normal":
                var = mu + sigma * stats.norm.ppf(cl)
            else:
                var = stats.t.ppf(cl, df_fit, loc, scale)
            records[cl].append(float(var))
    return pd.DataFrame(records, index=dates)
