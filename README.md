# Pairs Trading Backtester

This repository implements a cointegration-based pairs trading strategy using Python. It includes:

- Engle–Granger cointegration testing
- OLS hedge ratio estimation
- Rolling spread and z-score calculation
- Entry and exit trading rules
- Beta-neutral position sizing
- PnL and cumulative equity tracking
- Optional data loading from yfinance

The main implementation is in `pairs_trading.py`. A simplified version using yfinance is also included.

## Installation

Install the required dependencies:
pip install numpy pandas statsmodels matplotlib yfinance

## Running the Backtest

This will:

- Generate synthetic cointegrated price series
- Estimate the hedge ratio using rolling OLS
- Construct a spread and compute a rolling z-score
- Apply long and short entry signals
- Calculate PnL and cumulative PnL
- Display a cumulative PnL chart

## Example Output

Running the script produces output similar to:
Performance: {'Annualised Sharpe Ratio': np.float64(1.1), 'Sortino Ratio': np.float64(0.63), 'Cumulative PnL (%)': np.float64(61.0), 'Annualised Return (%)': np.float64(15.37), 'Max Drawdown (%)': np.float64(-6.99), 'Number of Trades': 37, 'Win Rate (%)': 83.78, 'Average Holding Period': np.float64(3.27), 'Mean Spread': np.float64(0.02187), 'Std Spread': np.float64(2.09241), 'Half-life': np.float64(0.47), 'ADF p-value': np.float64(0.0), 'Time in Market (%)': np.float64(12.1), 'Skewness': np.float64(2.65), 'Kurtosis': np.float64(45.16), 'Entry Z': 2.0, 'Exit Z': 0.5}

A plot of cumulative PnL and z-score spread is shown below.
<img width="1280" height="612" alt="Figure_1" src="https://github.com/user-attachments/assets/4fef43c8-9f29-4375-b290-ef4423ce3532" />


