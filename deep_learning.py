import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# STEP 1: Config
SYMBOL      = "AAPL"
START       = "2010-01-01"
END         = "2024-12-31"
LOOKBACK    = 60
TRAIN_SPLIT = 0.8
EPOCHS      = 50
BATCH_SIZE  = 32
HIDDEN_SIZE = 64
LR          = 0.001
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# STEP 2: Fetch data
print(f"Fetching {SYMBOL} data...")
raw     = yf.download(SYMBOL, start=START, end=END, auto_adjust=True, progress=False)
prices  = raw["Close"].squeeze()
volume  = raw["Volume"].squeeze()
high    = raw["High"].squeeze()
low     = raw["Low"].squeeze()

# STEP 3: Feature engineering
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast).mean()
    ema_slow   = series.ewm(span=slow).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    return macd_line, signal_line

returns     = prices.pct_change()
log_returns = np.log(prices / prices.shift(1))
vol_20      = returns.rolling(20).std()
vol_5       = returns.rolling(5).std()
ma_10       = prices.rolling(10).mean()
ma_50       = prices.rolling(50).mean()
ma_200      = prices.rolling(200).mean()
rsi         = compute_rsi(prices)
macd, macd_signal = compute_macd(prices)
bb_mid      = prices.rolling(20).mean()
bb_std      = prices.rolling(20).std()
bb_upper    = bb_mid + 2 * bb_std
bb_lower    = bb_mid - 2 * bb_std
bb_pct      = (prices - bb_lower) / (bb_upper - bb_lower)
vol_ratio   = volume / volume.rolling(20).mean()
price_range = (high - low) / prices
momentum_5  = returns.rolling(5).sum()
momentum_20 = returns.rolling(20).sum()

df = pd.DataFrame({
    "returns":     returns,
    "log_returns": log_returns,
    "vol_20":      vol_20,
    "vol_5":       vol_5,
    "ma_ratio_10": prices / ma_10,
    "ma_ratio_50": prices / ma_50,
    "ma_ratio_200":prices / ma_200,
    "rsi":         rsi,
    "macd":        macd,
    "macd_signal": macd_signal,
    "bb_pct":      bb_pct,
    "vol_ratio":   vol_ratio,
    "price_range": price_range,
    "momentum_5":  momentum_5,
    "momentum_20": momentum_20,
}).dropna()

# STEP 4: Target — next day direction (1=up, 0=down)
df["target"] = (df["returns"].shift(-1) > 0).astype(int)
df = df.dropna()

print(f"Dataset: {len(df)} days, {df.shape[1]-1} features")
print(f"Class balance: {df['target'].mean():.1%} up days")

# STEP 5: Build sequences
feature_cols = [c for c in df.columns if c != "target"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols].values)
y = df["target"].values

def build_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = build_sequences(X_scaled, y, LOOKBACK)
split        = int(len(X_seq) * TRAIN_SPLIT)

X_train = torch.FloatTensor(X_seq[:split])
y_train = torch.FloatTensor(y_seq[:split])
X_test  = torch.FloatTensor(X_seq[split:])
y_test  = torch.FloatTensor(y_seq[split:])

train_ds     = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# STEP 6: LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# STEP 7: Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x   = self.input_proj(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :]).squeeze()

# STEP 8: Training function
def train_model(model, loader, epochs, lr, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    losses    = []

    print(f"\nTraining {name}...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for Xb, yb in loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f}")
    return losses

# STEP 9: Evaluation function
def evaluate_model(model, X_test, y_test, name):
    model.eval()
    with torch.no_grad():
        probs = model(X_test).numpy()
    preds = (probs > 0.5).astype(int)
    acc   = accuracy_score(y_test.numpy(), preds)
    print(f"\n{name} Test Accuracy: {acc:.2%}")
    print(classification_report(y_test.numpy(), preds,
                                 target_names=["Down", "Up"], digits=3))
    return probs, preds, acc

# STEP 10: Backtesting function
def backtest(probs, y_test, returns_test, name):
    signals  = (probs > 0.5).astype(float)
    strategy = signals * returns_test
    hold     = returns_test

    cum_strat = pd.Series((1 + strategy).cumprod())
    cum_hold  = pd.Series((1 + hold).cumprod())

    ann_ret  = strategy.mean() * 252
    ann_vol  = strategy.std() * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd   = (cum_strat / cum_strat.cummax() - 1).min()

    bh_ret   = hold.mean() * 252
    bh_vol   = hold.std() * np.sqrt(252)
    bh_sharpe = bh_ret / bh_vol if bh_vol > 0 else 0

    print(f"\n{name} Backtest:")
    print(f"  Strategy  — Ann Return: {ann_ret:.1%} | Sharpe: {sharpe:.2f} | Max DD: {max_dd:.1%}")
    print(f"  Buy&Hold  — Ann Return: {bh_ret:.1%}  | Sharpe: {bh_sharpe:.2f}")

    return cum_strat, cum_hold, sharpe, ann_ret

# STEP 11: Train both models
n_features  = X_train.shape[2]
lstm_model  = LSTMModel(n_features, HIDDEN_SIZE)
trans_model = TransformerModel(n_features)

lstm_losses  = train_model(lstm_model,  train_loader, EPOCHS, LR, "LSTM")
trans_losses = train_model(trans_model, train_loader, EPOCHS, LR, "Transformer")

# STEP 12: Evaluate both
lstm_probs,  lstm_preds,  lstm_acc  = evaluate_model(lstm_model,  X_test, y_test, "LSTM")
trans_probs, trans_preds, trans_acc = evaluate_model(trans_model, X_test, y_test, "Transformer")

# STEP 13: Backtest both
test_returns = df["returns"].values[LOOKBACK + split:]
test_returns = test_returns[:len(lstm_probs)]

lstm_cum,  hold_cum,  lstm_sharpe,  lstm_ret  = backtest(lstm_probs,  y_test, test_returns, "LSTM")
trans_cum, _,         trans_sharpe, trans_ret = backtest(trans_probs, y_test, test_returns, "Transformer")

# STEP 14: Visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Training loss
axes[0,0].plot(lstm_losses,  label="LSTM",        color="blue",   linewidth=2)
axes[0,0].plot(trans_losses, label="Transformer", color="orange", linewidth=2)
axes[0,0].set_title("Training Loss")
axes[0,0].set_xlabel("Epoch")
axes[0,0].set_ylabel("BCE Loss")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Equity curves
axes[0,1].plot(lstm_cum.values,  label=f"LSTM (Sharpe: {lstm_sharpe:.2f})",
               color="blue",   linewidth=2)
axes[0,1].plot(trans_cum.values, label=f"Transformer (Sharpe: {trans_sharpe:.2f})",
               color="orange", linewidth=2)
axes[0,1].plot(hold_cum.values,  label="Buy & Hold",
               color="gray",   linewidth=1.5, linestyle="--")
axes[0,1].set_title("Out-of-Sample Equity Curves")
axes[0,1].set_ylabel("Cumulative Return")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Prediction confidence histogram
axes[1,0].hist(lstm_probs,  bins=50, alpha=0.6, color="blue",   label="LSTM",        density=True)
axes[1,0].hist(trans_probs, bins=50, alpha=0.6, color="orange", label="Transformer", density=True)
axes[1,0].axvline(0.5, color="red", linestyle="--", linewidth=2, label="Decision boundary")
axes[1,0].set_title("Prediction Confidence Distribution")
axes[1,0].set_xlabel("Predicted Probability (Up)")
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Summary bar chart
models    = ["LSTM", "Transformer", "Buy & Hold"]
sharpes   = [lstm_sharpe, trans_sharpe,
             (test_returns.mean() * 252) / (test_returns.std() * np.sqrt(252))]
colors    = ["blue", "orange", "gray"]
bars      = axes[1,1].bar(models, sharpes, color=colors, alpha=0.7, edgecolor="black")
axes[1,1].axhline(0, color="black", linewidth=1)
axes[1,1].set_title("Sharpe Ratio Comparison")
axes[1,1].set_ylabel("Sharpe Ratio")
for bar, val in zip(bars, sharpes):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f"{val:.2f}", ha="center", fontsize=11)
axes[1,1].grid(True, alpha=0.3, axis="y")

plt.suptitle(f"Deep Learning for Financial Time Series — {SYMBOL}", fontsize=14)
plt.tight_layout()
plt.savefig("/Users/davidhozak/python_algo/Deep Learning/deep_learning_results.png",
            dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print("\nDone! Saved to Deep Learning/deep_learning_results.png")