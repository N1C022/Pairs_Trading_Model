"""
pairs_trading.py

Implements a cointegration-based pairs trading backtest:
- tests cointegration with statsmodels.tsa.stattools.coint
- fits OLS to get beta and spread
- computes z-score of spread
- entry/exit rules: enter when z > entry_z or z < -entry_z, exit when |z| < exit_z
- beta-neutral position sizing (dollar-neutral)
- returns performance metrics and simple plot when run as script
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt

def fit_pair(series_x: pd.Series, series_y: pd.Series):
    """
    Regress y on x (y = a + b*x + eps) and return b (beta), intercept, residuals, pvalue of coint.
    """
    x = sm.add_constant(series_x)
    model = sm.OLS(series_y, x).fit()
    intercept = model.params[0]
    beta = model.params[1]
    resid = model.resid
    coint_t, pvalue, _ = ts.coint(series_y, series_x)
    return {"beta": beta, "intercept": intercept, "resid": resid, "coint_pvalue": pvalue}

def backtest_pairs(series_x: pd.Series, series_y: pd.Series,
                   entry_z: float = 2.0, exit_z: float = 0.5,
                   lookback: int = 252, fee: float = 0.0):
    """
    Rolling backtest:
    - Use expanding or rolling estimation for beta; here we use rolling OLS over 'lookback'
      (if insufficient history, we skip trading)
    - Returns DataFrame with positions and pnl
    """
    n = len(series_x)
    df = pd.DataFrame({"x": series_x, "y": series_y}).dropna()
    df["beta"] = np.nan
    df["intercept"] = np.nan
    df["spread"] = np.nan
    df["zscore"] = np.nan
    df["pos_x"] = 0.0  # position in shares of x (positive = long)
    df["pos_y"] = 0.0  # position in shares of y
    df["pnl"] = 0.0

    for t in range(lookback, len(df)):
        window = df.iloc[t-lookback:t]
        res = fit_pair(window["x"], window["y"])
        beta = res["beta"]
        intercept = res["intercept"]
        spread = df["y"].iloc[t] - (intercept + beta * df["x"].iloc[t])
        # compute rolling mean/std of residuals using window
        resid_window = window["y"] - (res["intercept"] + res["beta"] * window["x"])
        m = resid_window.mean()
        s = resid_window.std(ddof=0) if resid_window.std(ddof=0) > 0 else 1e-9
        z = (spread - m) / s

        df.at[df.index[t], "beta"] = beta
        df.at[df.index[t], "intercept"] = intercept
        df.at[df.index[t], "spread"] = spread
        df.at[df.index[t], "zscore"] = z

        # simple strategy: if z > entry -> short y, long x to be beta-neutral (dollar neutral)
        # dollar-neutral sizing: size positions so dollar exposure equal
        # assume we invest $1 total when we enter (or scale by volatility if you prefer)
        pos_x = df["pos_x"].iloc[t-1]
        pos_y = df["pos_y"].iloc[t-1]
        px = df["x"].iloc[t]
        py = df["y"].iloc[t]

        # exit condition
        if abs(z) < exit_z:
            pos_x, pos_y = 0.0, 0.0
        else:
            if z > entry_z:
                # expect y to drop relative to x: short y, long x
                # set dollar neutral: w_x * px = w_y * py
                # choose weights w_x and w_y such that w_x + w_y = 1 (dollar invested)
                # solve: w_x * px = w_y * py -> w_x * px = (1 - w_x) * py -> w_x = py / (px + py)
                w_x = py / (px + py)
                w_y = 1 - w_x
                # but to keep signs: x long, y short
                pos_x = w_x / px  # shares of x (long)
                pos_y = -w_y / py  # shares of y (short)
            elif z < -entry_z:
                # expect y to rise relative to x: long y, short x
                w_x = py / (px + py)
                w_y = 1 - w_x
                pos_x = -w_x / px
                pos_y = w_y / py

        df.at[df.index[t], "pos_x"] = pos_x
        df.at[df.index[t], "pos_y"] = pos_y

        # PnL from previous day's held positions (mark-to-market)
        prev_px = df["x"].iloc[t-1]
        prev_py = df["y"].iloc[t-1]
        prev_pos_x = df["pos_x"].iloc[t-1]
        prev_pos_y = df["pos_y"].iloc[t-1]

        pnl = prev_pos_x * (px - prev_px) + prev_pos_y * (py - prev_py)
        # subtract fees: round-trip on change in positions (very simplified)
        trade_cost = fee * (abs(df["pos_x"].iloc[t] - prev_pos_x) * px + abs(df["pos_y"].iloc[t] - prev_pos_y) * py)
        df.at[df.index[t], "pnl"] = pnl - trade_cost

    df["cum_pnl"] = df["pnl"].cumsum()
    # simple performance
    returns = df["pnl"].dropna()
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0
    return df, {"sharpe": sharpe, "total_pnl": df["cum_pnl"].iloc[-1]}

if __name__ == "__main__":
    # Example using synthetic cointegrated series
    np.random.seed(42)
    T = 1000
    x = np.cumsum(np.random.normal(0, 1, T)) + 50
    # y is x * beta + stationary noise
    true_beta = 1.3
    y = true_beta * x + np.random.normal(0, 1.5, T)

    dates = pd.date_range("2020-01-01", periods=T)
    sx = pd.Series(x, index=dates)
    sy = pd.Series(y, index=dates)

    df, perf = backtest_pairs(sx, sy, entry_z=2.0, exit_z=0.5, lookback=200, fee=0.0)
    print("Performance:", perf)
    df["cum_pnl"].plot(title="Pairs Trading Cumulative PnL")
    plt.show()
