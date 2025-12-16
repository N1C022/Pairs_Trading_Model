import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


#Download Price Data
def load_data(ticker_x, ticker_y, start="2015-01-01"):
    df_x = yf.download(ticker_x, start=start)["Adj Close"]
    df_y = yf.download(ticker_y, start=start)["Adj Close"]
    df = pd.DataFrame({ticker_x: df_x, ticker_y: df_y}).dropna()
    return df



# Cointegration + Hedge Ratio

def estimate_hedge_ratio(y, x):
    #OLS regression: y = a + b*x
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    return model.params[1]  # slope = hedge ratio


def check_cointegration(series_x, series_y):
    score, pvalue, _ = coint(series_x, series_y)
    return pvalue


# Trading Signals

def compute_spread(df, ticker_x, ticker_y, hedge_ratio):
    return df[ticker_y] - hedge_ratio * df[ticker_x]


def zscore(series):
    return (series - series.mean()) / series.std()


def generate_signals(z, entry_z=2.0, exit_z=0.5):
    long_signal = (z > entry_z).astype(int)
    short_signal = (z < -entry_z).astype(int)
    exit_signal = (abs(z) < exit_z).astype(int)

    positions = []
    pos = 0

    for i in range(len(z)):
        if pos == 0:
            if long_signal[i]:
                pos = -1  # short spread
            elif short_signal[i]:
                pos = 1   # long spread
        else:
            if exit_signal[i]:
                pos = 0

        positions.append(pos)

    return pd.Series(positions, index=z.index)


# backtest

def backtest_pairs(df, ticker_x, ticker_y, hedge_ratio, positions):
    spread = compute_spread(df, ticker_x, ticker_y, hedge_ratio)
    spread_ret = spread.diff()

    pnl = positions.shift(1) * spread_ret
    pnl = pnl.fillna(0)

    equity = pnl.cumsum()

    return pnl, equity
