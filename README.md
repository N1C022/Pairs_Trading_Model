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
Performance: {'sharpe': 2.41, 'total_pnl': 56.72}

A plot of cumulative PnL is also shown.
