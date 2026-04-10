import warnings

import numpy as np
import pandas as pd

from statsmodels.regression.quantile_regression import IterationLimitWarning, QuantReg

from utils.data import build_har_features, rolling_windows

_FEATURE_COLS = ["C_d", "C_w", "C_m", "J_d"]


def predict_var(
    df: pd.DataFrame,
    window: int = 250,
    confidence_levels: list[float] = [0.01, 0.05, 0.10],
) -> pd.DataFrame:
    feat_df = build_har_features(df)
    records = {cl: [] for cl in confidence_levels}
    dates = []

    for train, test_date, _ in rolling_windows(feat_df, window):
        y = train["log_ret"].values[1:]
        X_raw = train[_FEATURE_COLS].values[:-1]
        X = np.column_stack([np.ones(len(X_raw)), X_raw])

        x_pred = np.array([1.0] + list(train.iloc[-1][_FEATURE_COLS].values))
        dates.append(test_date)

        for cl in confidence_levels:
            model = QuantReg(y, X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", IterationLimitWarning)
                res = model.fit(q=cl, max_iter=2000, p_tol=1e-6)
            records[cl].append(float(res.params @ x_pred))

    result = pd.DataFrame(records, index=dates)
    return result.dropna()
