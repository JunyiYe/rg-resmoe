"""Shared walk-forward training core used by every neural model.

Implements the paper's protocol: 16 stock-level features (6 base + 10 raw
return lags), rolling 504-trading-day development windows with an 85/15
chronological train/validation split, patience-based early stopping on
validation MSE, and a 63-trading-day test period per window.
"""
import os
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROCESSED_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"

# Per-seed, per-window predictions land under results/predictions/<model>/
# (or results/predictions/jp/<model>/ when MARKET=jp, for the Japan
# cross-market replication -- keeps JP runs from overwriting the US ones).
_MARKET = os.environ.get("MARKET", "us")
_RESULTS_BASE = pathlib.Path(__file__).resolve().parent.parent / "results" / "predictions"
RESULTS_DIR = _RESULTS_BASE if _MARKET == "us" else _RESULTS_BASE / _MARKET

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_LAGS = 10
BASE_FEATURES = ["vol_5", "vol_20", "vol_60", "ret_5", "ret_20", "rsi_14"]
LAG_FEATURES = [f"ret_lag_{k}" for k in range(1, N_LAGS + 1)]
FEATURES = BASE_FEATURES + LAG_FEATURES
# TARGET_COL override supports the horizon sweep (target_fwd_vol_{21,63}
# panels built with build_panel.py TARGET_HORIZON=...); default unchanged.
TARGET = os.environ.get("TARGET_COL", "target_fwd_vol_5")

N_EXPERTS = 4
TRAIN_WINDOW_DAYS = int(os.environ.get("TRAIN_WINDOW_DAYS", "504"))
STEP_DAYS = int(os.environ.get("STEP_DAYS", "63"))
LR = 1e-2
LB_WEIGHT = 0.05
SEED = 0

VAL_FRAC = 0.15
CHECK_EVERY = 5
PATIENCE_CHECKS = 10
MAX_EPOCHS = 2000


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    if all(f in panel.columns for f in LAG_FEATURES):
        return panel
    panel = panel.sort_values(["ticker", "Date"]).copy()
    for k in range(1, N_LAGS + 1):
        panel[f"ret_lag_{k}"] = panel.groupby("ticker")["log_ret"].shift(k - 1)
    panel = panel.dropna(subset=LAG_FEATURES).reset_index(drop=True)
    return panel


class ExpertMLP(nn.Module):
    def __init__(self, in_dim, hidden=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GatingNet(nn.Module):
    def __init__(self, in_dim, n_experts, hidden=16, temperature=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, n_experts))
        self.temperature = temperature

    def forward(self, x):
        return torch.softmax(self.net(x) / self.temperature, dim=-1)


def _split_train_val(df: pd.DataFrame):
    dates_sorted = np.sort(df["Date"].unique())
    split_idx = int(len(dates_sorted) * (1 - VAL_FRAC))
    val_split_date = dates_sorted[split_idx]
    return df[df["Date"] < val_split_date], df[df["Date"] >= val_split_date]
