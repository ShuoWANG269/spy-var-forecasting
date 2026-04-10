import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from utils.data import rolling_windows


def predict_var(
    df: pd.DataFrame,
    window: int = 250,
    confidence_levels: Sequence[float] | None = None,
    model_type: str = "GARCH",
    dist: str = "normal",
    return_sigma: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """GARCH VaR prediction.

    model_type: 'GARCH' or 'GJR'
    dist: 'normal' or 't'
    return_sigma: If True, also return conditional volatility series.
    """
    if model_type not in {"GARCH", "GJR"}:
        raise ValueError(f"Unsupported model_type: {model_type}")
    if dist not in {"normal", "t"}:
        raise ValueError(f"Unsupported dist: {dist}")

    levels = (0.01, 0.05, 0.10) if confidence_levels is None else confidence_levels
    records = {cl: [] for cl in levels}
    sigmas = [] if return_sigma else None
    dates = []
    arch_dist = "Normal" if dist == "normal" else "StudentsT"

    for train, test_date, _ in rolling_windows(df, window):
        dates.append(test_date)
        r = train["log_ret"].values * 100

        o = 1 if model_type == "GJR" else 0
        am = arch_model(r, vol="GARCH", p=1, o=o, q=1, dist=arch_dist)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)

        fcast = res.forecast(horizon=1)
        mu_scaled = float(fcast.mean.iloc[-1, 0])
        sigma_scaled = float(np.sqrt(fcast.variance.iloc[-1, 0]))
        mu = mu_scaled / 100
        sigma = sigma_scaled / 100

        if return_sigma:
            sigmas.append(float(res.conditional_volatility[-1]) / 100)

        nu = None
        if dist == "t":
            nu = float(res.params.get("nu", 5.0))
            nu = max(nu, 2.01)
            scale = np.sqrt((nu - 2.0) / nu)

        for cl in levels:
            if dist == "normal":
                q = stats.norm.ppf(cl)
            else:
                q = stats.t.ppf(cl, nu) * scale
            records[cl].append(float(mu + sigma * q))

    var_df = pd.DataFrame(records, index=dates)

    if return_sigma:
        sigma_series = pd.Series(sigmas, index=dates, name="sigma")
        return var_df, sigma_series

    return var_df
