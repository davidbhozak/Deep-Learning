# Deep Learning for Financial Time Series

Built and compared an LSTM and Transformer model to predict next-day AAPL return direction, then backtested the signal.

## What it does
- Engineers 15 features: RSI, MACD, Bollinger Bands, momentum, volume ratio etc.
- Trains a 2-layer LSTM and a Transformer on 60-day lookback windows
- Predicts next-day up/down direction
- Backtests: go long when model predicts up, stay flat otherwise

## Results (out-of-sample 2022-2024)
| Model | Accuracy | Sharpe |
|---|---|---|
| LSTM | 49% | 0.76 |
| Transformer | 49% | 0.40 |
| Buy & Hold | — | 0.84 |

49% accuracy sounds bad but the LSTM still generates a 0.76 Sharpe — it's right on the bigger moves and wrong on the smaller ones, which is what matters.

## Stack
Python, PyTorch, yfinance, sklearn, matplotlib