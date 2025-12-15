import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats
import matplotlib.pyplot as plt
class PairsTrader:
    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.5, lookback: int = 252, fee: float = 0.0,
                 use_kalman: bool = True, stop_loss_z: float = 4.0, max_holding_period: int = 20, kalman_delta: float = 1e-5):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
        self.fee = fee
        self.use_kalman = use_kalman
        self.stop_loss_z = stop_loss_z
        self.max_holding_period = max_holding_period
        self.kalman_delta = kalman_delta
        self.results = None

    def _run_kalman_filter(self, x, y):
        # Kalman Filter estimates regression coefficients (slope and intercept).
        # y = alpha + beta * x + noise
        
        # theta = [alpha, beta]
        n = len(x)
        state_means = np.zeros((n, 2))
        
        # Initial estimates
        theta = np.zeros(2)
        P = np.eye(2) * 1.0 # Initial covariance
        
        # transition covariance (process noise)
        # Using a small noise for random walk
        Q = self.kalman_delta * np.eye(2)
        
        # Measurement variance
        R = 0.001 
        
        for t in range(n):
            # Prediction
            # theta(t|t-1) = theta(t-1)
            # P(t|t-1) = P(t-1) + Q
            P = P + Q
            
            # Update
            xt = x[t]
            yt = y[t]
            H = np.array([1.0, xt])
            
            # Innovation
            y_pred = np.dot(H, theta)
            error = yt - y_pred
            
            # Innovation variance: S = H P H.T + R
            # can explode if x is large and P is large.
            S = np.dot(H, np.dot(P, H.T)) + R
            
            # Kalman Gain: K = P H.T / S
            K = np.dot(P, H.T) / S
            
            # Update State
            theta = theta + K * error
            
            # Update Covariance
            # Stabilised: P = (I - KH)P(I - KH)' + KRK' 
            # simple: P = (I - KH)P
            # this uses simple form but forces symmetry (xtx maybe use stabilised)
            I_KH = np.eye(2) - np.outer(K, H)
            P = np.dot(I_KH, P)
            
            state_means[t] = theta
            
        return state_means

    def backtest(self, series_x: pd.Series, series_y: pd.Series) -> pd.DataFrame:
        # :return: DataFrame with full backtest results
        # Align data
        df = pd.DataFrame({"x": series_x, "y": series_y}).dropna()
        
        # Beta and intercept estimation
        if self.use_kalman:
            # Kalman filter
            state_means = self._run_kalman_filter(df["x"].values, df["y"].values)
            df["intercept"] = state_means[:, 0]
            df["beta"] = state_means[:, 1]
            
            # avoid look ahead bias by using yesterday's trades for todays
            # The KF state_means[t] includes the observation at t. 
            # We must not use it to calculate spread[t] for the signal at t.
            df["intercept"] = df["intercept"].shift(1)
            df["beta"] = df["beta"].shift(1)
            
        else:
            # Rolling OLS
            exog = sm.add_constant(df["x"])
            rols = RollingOLS(df["y"], exog, window=self.lookback)
            rres = rols.fit()
            params = rres.params
            df["intercept"] = params["const"]
            df["beta"] = params["x"]
            
            # RollingOLS params at index t are based on window ending at t.
            # lag them 1 day to be OOS safe though some momentum strategies trade "on close" with parameters fitted on that close. consistency with KF so we shift.
            df["intercept"] = df["intercept"].shift(1)
            df["beta"] = df["beta"].shift(1)
        
        #Calculate spread and z-Score
        df["spread"] = df["y"] - (df["intercept"] + df["beta"] * df["x"])
        
        # For z-score, use rolling window of spread to normalise
        # implicitly assumes the spread mean reverts around a local mean (often 0 if KF works well)
        # min_periods=lookback ensures we don't get z-scores until we have enough data 
        # if use_kalman is true, faster signals may be better so min_periods=1
        min_periods = self.lookback if not self.use_kalman else 5
        rolling_mean = df["spread"].rolling(window=self.lookback, min_periods=min_periods).mean().shift(1)
        rolling_std = df["spread"].rolling(window=self.lookback, min_periods=min_periods).std().shift(1)
        
        # avoid division by zero
        rolling_std = rolling_std.fillna(1e-9).replace(0.0, 1e-9)
        
        df["zscore"] = (df["spread"] - rolling_mean) / rolling_std
        df["zscore"] = df["zscore"].fillna(0.0) # Fill initial NaNs with 0
        
        # Generate Signals and Positions
        df["pos_x"] = 0.0
        df["pos_y"] = 0.0
        
        z_values = df["zscore"].values
        px_values = df["x"].values
        py_values = df["y"].values
        
        pos_x = np.zeros(len(df))
        pos_y = np.zeros(len(df))
        
        current_pos = 0 # 0: flat, 1: long spread (long y, short x), -1: short spread (short y, long x)
        days_held = 0
        
        # Start loop
        start_idx = self.lookback if not self.use_kalman else min_periods 
        for t in range(start_idx, len(df)):
            z = z_values[t]
            px = px_values[t]
            py = py_values[t]
            
            prev_x_shares = pos_x[t-1]
            prev_y_shares = pos_y[t-1]
            
            # Risk Management Checks
            forced_exit = False
            
            if current_pos != 0:
                days_held += 1
                
                # Stop Loss
                if abs(z) > self.stop_loss_z:
                    forced_exit = True
                
                # Max Holding Period
                if days_held > self.max_holding_period:
                    forced_exit = True
            
            # Logic
            if current_pos != 0:
                if forced_exit or abs(z) < self.exit_z:
                    # Exit
                    current_pos = 0
                    pos_x[t] = 0.0
                    pos_y[t] = 0.0
                    days_held = 0
                else:
                    # Hold
                    pos_x[t] = prev_x_shares
                    pos_y[t] = prev_y_shares
            
            else: # Flat
                if z > self.entry_z and abs(z) <= self.stop_loss_z: # Don't enter if already past stop loss
                    # Short Spread
                    w_x = py / (px + py)
                    w_y = 1 - w_x
                    pos_x[t] = w_x / px
                    pos_y[t] = -w_y / py
                    current_pos = -1
                    days_held = 0
                    
                elif z < -self.entry_z and abs(z) <= self.stop_loss_z:
                    # Long Spread
                    w_x = py / (px + py)
                    w_y = 1 - w_x
                    pos_x[t] = -w_x / px
                    pos_y[t] = w_y / py
                    current_pos = 1
                    days_held = 0
                else:
                    # Remain Flat
                    pos_x[t] = 0.0
                    pos_y[t] = 0.0

        df["pos_x"] = pos_x
        df["pos_y"] = pos_y
        
        # Calculate PnL
        df["pnl"] = (df["pos_x"].shift(1) * df["x"].diff()) + (df["pos_y"].shift(1) * df["y"].diff())
        
        # Transaction Costs
        pos_change_x = df["pos_x"].diff().abs().fillna(0)
        pos_change_y = df["pos_y"].diff().abs().fillna(0)
        costs = self.fee * (pos_change_x * df["x"] + pos_change_y * df["y"])
        df["pnl"] -= costs
        
        df["cum_pnl"] = df["pnl"].cumsum()
        self.results = df
        return df

    def summarize_performance(self):
        return self.calculate_metrics()

    def calculate_metrics(self):
        if self.results is None:
            return {}
            
        df = self.results
        returns = df["pnl"] # Daily PnL in dollars. 
        # Note: capital base needed to get % returns properly 
        # assume we allocate 1.0 (or capital C) to the strategy.
        # assumes hat the positions (weights) sum to ~1.0 gross (0.5 long, 0.5 short).
        
        # Risk/Return Metrics
        total_pnl = df["pnl"].sum()
        avg_daily_pnl = returns.mean()
        std_daily_pnl = returns.std()
        
        annual_factor = 252
        
        if std_daily_pnl > 0:
            sharpe = (avg_daily_pnl / std_daily_pnl) * np.sqrt(annual_factor)
            # Sortino
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                sortino = (avg_daily_pnl / downside_std) * np.sqrt(annual_factor)
            else:
                sortino = np.inf
        else:
            sharpe = 0.0
            sortino = 0.0
            
        # Max drawdown
        cum_pnl_series = df["cum_pnl"]
        running_max = cum_pnl_series.cummax()
        drawdown = cum_pnl_series - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown * 100.0 # Assuming 1.0 base
        
        cum_pnl_pct = total_pnl * 100.0
        
        # Annualised Return (Simple compounding approximation)
        n_days = len(df)
        if n_days > 0:
            ann_return = (total_pnl / n_days) * 252 * 100 # Arithmetic annualisation
        else:
            ann_return = 0.0

        # Skew / Kurtosis
        clean_returns = returns.dropna()
        ret_skew = stats.skew(clean_returns)
        ret_kurt = stats.kurtosis(clean_returns)

        #Trade Statistics
        trade_stats = self._calculate_trade_stats(df)
        
        #Spread & Model Stats
        spread = df["spread"].dropna()
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        # ADF Test
        try:
            adf_res = adfuller(spread)
            adf_pvalue = adf_res[1]
        except:
            adf_pvalue = np.nan
            
        # Half-life
        halflife = self._calculate_halflife(spread)
        
        # Time in market
        days_in_market = (df["pos_x"] != 0).sum()
        pct_time_in_market = (days_in_market / n_days) * 100
        
        return {
            "Annualised Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Cumulative PnL (%)": round(cum_pnl_pct, 2),
            "Annualised Return (%)": round(ann_return, 2),
            "Max Drawdown (%)": round(max_drawdown_pct, 2),
            "Number of Trades": trade_stats["count"],
            "Win Rate (%)": round(trade_stats["win_rate"] * 100, 2),
            "Average Holding Period": round(trade_stats["avg_holding"], 2),
            "Mean Spread": round(spread_mean, 5),
            "Std Spread": round(spread_std, 5),
            "Half-life": round(halflife, 2) if halflife else None,
            "ADF p-value": round(adf_pvalue, 4),
            "Time in Market (%)": round(pct_time_in_market, 2),
            "Skewness": round(ret_skew, 2),
            "Kurtosis": round(ret_kurt, 2),
            "Entry Z": self.entry_z,
            "Exit Z": self.exit_z,
        }

    def _calculate_trade_stats(self, df):
        # Identify trades based on pos_x changing from 0 to something or flipping..
        # A trade is defined from opening a position (0 -> non-0) to closing it (non-0 -> 0).
        # Note: Reversals (Long -> Short directly)treated as a close and open
        
        trades = []
        in_trade = False
        start_entry_idx = 0
        entry_pnl = 0.0
        
        # Positions
        pos = df["pos_x"].values
        pnl = df["pnl"].values # Daily PnL
        
        current_trade_pnl = 0.0
        current_trade_days = 0
        
        for t in range(1, len(df)):
            if pos[t] != 0:
                if not in_trade:
                    # New trade opened
                    in_trade = True
                    start_entry_idx = t
                    current_trade_pnl = pnl[t] # Includes transaction cost of entry
                    current_trade_days = 1
                else:
                    # Continuing trade
                    # Check if position flipped sign (Long <-> Short)
                    if np.sign(pos[t]) != np.sign(pos[t-1]):
                        # Reverse: Close current, Open new
                        trades.append({"pnl": current_trade_pnl, "days": current_trade_days})
                        
                        # New trade starts
                        # pnl[t] contains cost of flipping.
                        # Ideally splitting pnl[t] is hard without more data.
                        # pnl[t] treated as belonging to the new position for simplicity as it pays the cost for the new direction.
                        current_trade_pnl = pnl[t]
                        current_trade_days = 1
                    else:
                        current_trade_pnl += pnl[t]
                        current_trade_days += 1
            else:
                if in_trade:
                    # Trade closed (pos becomes 0)
                    # pnl[t] is the PnL of the closing day (usually just cost or 0 return from flat)
                    # Use pnl[t] which captures exit cost
                    current_trade_pnl += pnl[t]
                    trades.append({"pnl": current_trade_pnl, "days": current_trade_days})
                    in_trade = False
                    current_trade_pnl = 0.0
                    current_trade_days = 0
        
        # If still open at end, count as trade?
        if in_trade:
            trades.append({"pnl": current_trade_pnl, "days": current_trade_days})
            
        if not trades:
            return {"count": 0, "win_rate": 0.0, "avg_holding": 0.0}
            
        pnls = [t["pnl"] for t in trades]
        # Win rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(trades)
        
        # Avg holding
        days = [t["days"] for t in trades]
        avg_holding = np.mean(days)
        
        return {"count": len(trades), "win_rate": win_rate, "avg_holding": avg_holding}

    def _calculate_halflife(self, spread):
        # Ornstein-Uhlenbeck: dy = -theta * y * dt + sigma * dW
        # Discrete: y[t] - y[t-1] = -theta * y[t-1] + epsilon
        # Regress (y[t] - y[t-1]) on y[t-1]
        
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Align
        idx = spread_lag.index.intersection(spread_diff.index)
        spread_lag = spread_lag.loc[idx]
        spread_diff = spread_diff.loc[idx]
        
        if len(spread_lag) < 10:
            return None
        
        model = sm.OLS(spread_diff, spread_lag)
        res = model.fit()
        
        theta = -res.params.iloc[0]
        if theta <= 0:
            return np.inf # Non-mean-reverting
            
        halflife = np.log(2) / theta
        return halflife
    
    def plot_performance(self):
        if self.results is None:
            print("No results to plot.")
            return
        
        plt.figure(figsize=(12, 10))
        
        ax1 = plt.subplot(3, 1, 1)
        self.results["cum_pnl"].plot(ax=ax1, title="Pairs Trading Cumulative PnL", color='green')
        ax1.set_ylabel("PnL ($)")
        ax1.grid(True)
        
        ax2 = plt.subplot(3, 1, 2)
        self.results["zscore"].plot(ax=ax2, title="Spread Z-Score", color='blue', alpha=0.6)
        ax2.axhline(self.entry_z, color='red', linestyle='--', label='Entry')
        ax2.axhline(-self.entry_z, color='red', linestyle='--')
        ax2.axhline(self.exit_z, color='black', linestyle=':', label='Exit')
        ax2.axhline(-self.exit_z, color='black', linestyle=':')
        ax2.axhline(self.stop_loss_z, color='orange', linestyle='-', label='Stop Loss')
        ax2.axhline(-self.stop_loss_z, color='orange', linestyle='-')
        ax2.legend()
        ax2.grid(True)

        ax3 = plt.subplot(3, 1, 3)
        self.results["beta"].plot(ax=ax3, title="Beta (Hedge Ratio)", color='purple')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    np.random.seed(67)
    T = 1000
    x = np.cumsum(np.random.normal(0, 1, T)) + 50
    # Cointegrated y with time-varying beta
    # Create a beta that shifts
    true_beta = np.linspace(1.0, 1.5, T)
    y = true_beta * x + np.random.normal(0, 1.5, T)

    dates = pd.date_range("2020-01-01", periods=T)
    sx = pd.Series(x, index=dates)
    sy = pd.Series(y, index=dates)

    # Enable Kalman and Risk Management
    trader = PairsTrader(
        entry_z=2.0, 
        exit_z=0.5, 
        lookback=50, 
        fee=0.0, 
        use_kalman=True,
        stop_loss_z=4.0,
        max_holding_period=30
    )
    df = trader.backtest(sx, sy)
    
    print("Performance:", trader.summarize_performance())
    trader.plot_performance()
