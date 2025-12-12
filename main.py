"""
pairs_trading.py

Implements a cointegration-based pairs trading backtest using a class-based structure.
Features:
- Rolling OLS for dynamic beta and spread calculation
- Z-score based entry/exit signals
- Dollar-neutral position sizing
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt

class PairsTrader:
    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.5, lookback: int = 252, fee: float = 0.0):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
        self.fee = fee
        self.results = None

    def backtest(self, series_x: pd.Series, series_y: pd.Series) -> pd.DataFrame:
        """
        Run the pairs trading backtest.
        :param series_x: Price series for asset X (independent variable)
        :param series_y: Price series for asset Y (dependent variable)
        :return: DataFrame with full backtest results
        """
        # Align data
        df = pd.DataFrame({"x": series_x, "y": series_y}).dropna()
        
        # 1. Rolling OLS for Beta and Intercept
        # y = alpha + beta * x
        exog = sm.add_constant(df["x"])
        # RollingOLS requires pandas >= 0.24 and statsmodels >= 0.11
        rols = RollingOLS(df["y"], exog, window=self.lookback)
        rres = rols.fit()
        params = rres.params
        
        df["intercept"] = params["const"]
        df["beta"] = params["x"]
        
        # 2. Calculate Spread and Z-Score
        # Spread = y - (alpha + beta * x)
        df["spread"] = df["y"] - (df["intercept"] + df["beta"] * df["x"])
        
        # Rolling stats for spread Z-score
        # We use a rolling window of the spread itself to normalize it. 
        # Alternatively, we could use the residuals' std from OLS, but normalizing the spread 
        # based on recent history is also common. Let's stick to the previous logic:
        # The previous logic calculated mean/std of the *residuals* over the lookback window.
        # RollingOLS residuals are accessible, but let's recalculate rolling z-score of the spread
        # to match the logic of "observing the spread distribution".
        rolling_mean = df["spread"].rolling(window=self.lookback).mean()
        rolling_std = df["spread"].rolling(window=self.lookback).std()
        
        df["zscore"] = (df["spread"] - rolling_mean) / rolling_std
        
        # 3. Generate Signals and Positions
        df["pos_x"] = 0.0
        df["pos_y"] = 0.0
        
        # We need to iterate for state-dependent logic (exit requires checks against current position)
        # To make this faster, we convert necessary columns to numpy arrays
        z_values = df["zscore"].values
        px_values = df["x"].values
        py_values = df["y"].values
        
        pos_x = np.zeros(len(df))
        pos_y = np.zeros(len(df))
        
        current_pos = 0 # 0: flat, 1: long spread (long y, short x), -1: short spread (short y, long x)
        # Note: Long spread usually means betting spread increases. 
        # If Spread = Y - Beta*X. 
        # If Z < -entry: Spread is too low, expect rise -> Buy Spread -> Buy Y, Sell X.
        # If Z > entry: Spread is too high, expect drop -> Sell Spread -> Sell Y, Buy X.
        
        # We start loop from 'lookback' because z-score is NaN before that
        for t in range(self.lookback, len(df)):
            z = z_values[t]
            px = px_values[t]
            py = py_values[t]
            
            # Carry forward previous position (will be overwritten if signal changes)
            prev_x_shares = pos_x[t-1]
            prev_y_shares = pos_y[t-1]
            
            # Check exit first if we are in a position
            if current_pos != 0:
                if abs(z) < self.exit_z:
                    current_pos = 0
                    pos_x[t] = 0.0
                    pos_y[t] = 0.0
                else:
                    # Hold position
                    pos_x[t] = prev_x_shares
                    pos_y[t] = prev_y_shares
            
            # Check entry if we are flat
            else: # current_pos == 0
                if z > self.entry_z:
                    # Spread too high -> Short Spread -> Short Y, Long X
                    # Dollar neutral weighting
                    # w_x * px = w_y * py = 0.5 (assuming $1 total exposure) -> w_x = 0.5, w_y = 0.5
                    # OR w_x = py / (px + py) for fully balanced
                    w_x = py / (px + py)
                    w_y = 1 - w_x
                    
                    pos_x[t] = w_x / px   # Long X
                    pos_y[t] = -w_y / py  # Short Y
                    current_pos = -1
                    
                elif z < -self.entry_z:
                    # Spread too low -> Long Spread -> Long Y, Short X
                    w_x = py / (px + py)
                    w_y = 1 - w_x
                    
                    pos_x[t] = -w_x / px  # Short X
                    pos_y[t] = w_y / py   # Long Y
                    current_pos = 1
                else:
                    # Remain flat
                    pos_x[t] = 0.0
                    pos_y[t] = 0.0

        df["pos_x"] = pos_x
        df["pos_y"] = pos_y
        
        # 4. Calculate PnL
        # PnL = pos_x * change_in_x + pos_y * change_in_y
        # Shift positions by 1 to represent "position held at start of day" applied to "price change today"
        # The previous code used pos[t-1] * (price[t] - price[t-1]), which is correct.
        
        df["pnl"] = (df["pos_x"].shift(1) * df["x"].diff()) + (df["pos_y"].shift(1) * df["y"].diff())
        
        # Transaction Costs
        # trade_size * price * fee
        # pos change for x
        pos_change_x = df["pos_x"].diff().abs().fillna(0)
        pos_change_y = df["pos_y"].diff().abs().fillna(0)
        
        costs = self.fee * (pos_change_x * df["x"] + pos_change_y * df["y"])
        df["pnl"] -= costs
        
        df["cum_pnl"] = df["pnl"].cumsum()
        self.results = df
        return df

    def summarize_performance(self):
        if self.results is None:
            return "No results. Run backtest() first."
        
        returns = self.results["pnl"].dropna()
        total_pnl = self.results["cum_pnl"].iloc[-1]
        if returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
            
        return {
            "Total PnL": total_pnl,
            "Sharpe Ratio": sharpe
        }
    
    def plot_performance(self):
        if self.results is None:
            print("No results to plot.")
            return
        
        plt.figure(figsize=(12, 8))
        
        ax1 = plt.subplot(2, 1, 1)
        self.results["cum_pnl"].plot(ax=ax1, title="Pairs Trading Cumulative PnL", color='green')
        ax1.set_ylabel("PnL ($)")
        ax1.grid(True)
        
        ax2 = plt.subplot(2, 1, 2)
        self.results["zscore"].plot(ax=ax2, title="Spread Z-Score", color='blue', alpha=0.6)
        ax2.axhline(self.entry_z, color='red', linestyle='--', label='Entry')
        ax2.axhline(-self.entry_z, color='red', linestyle='--')
        ax2.axhline(self.exit_z, color='black', linestyle=':', label='Exit')
        ax2.axhline(-self.exit_z, color='black', linestyle=':')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    T = 1000
    x = np.cumsum(np.random.normal(0, 1, T)) + 50
    # Cointegrated y
    true_beta = 1.3
    y = true_beta * x + np.random.normal(0, 1.5, T)

    dates = pd.date_range("2020-01-01", periods=T)
    sx = pd.Series(x, index=dates)
    sy = pd.Series(y, index=dates)

    trader = PairsTrader(entry_z=2.0, exit_z=0.5, lookback=20, fee=0.0) # Reduced lookback for example
    df = trader.backtest(sx, sy)
    
    print("Performance:", trader.summarize_performance())
    trader.plot_performance()
