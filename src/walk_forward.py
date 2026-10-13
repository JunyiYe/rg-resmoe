"""Shared walk-forward training core used by every neural model.

Implements the paper's protocol: 16 stock-level features (6 base + 10 raw
return lags), rolling 504-trading-day development windows with an 85/15
chronological train/validation split, patience-based early stopping on
validation MSE, and a 63-trading-day test period per window.
"""
import copy
import os
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.preprocessing import StandardScaler

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
TARGET_VOL = 0.15
COST_BP = 5
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


def train_explicit_es(train_X, train_y, train_bucket, val_X, val_y, val_bucket, in_dim, n_experts=N_EXPERTS, seed=SEED, lr=LR):
    torch.manual_seed(seed)
    experts = nn.ModuleList([ExpertMLP(in_dim) for _ in range(n_experts)]).to(DEVICE)
    opt = torch.optim.Adam(experts.parameters(), lr=lr)
    Xt = torch.tensor(train_X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(train_y, dtype=torch.float32, device=DEVICE)
    bt = torch.tensor(train_bucket, dtype=torch.long, device=DEVICE)
    Xv = torch.tensor(val_X, dtype=torch.float32, device=DEVICE)

    best_val, best_state, no_improve = np.inf, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        opt.zero_grad()
        loss = 0.0
        for i in range(n_experts):
            mask = bt == i
            if mask.sum() < 5:
                continue
            loss = loss + nn.functional.mse_loss(experts[i](Xt[mask]), yt[mask])
        loss.backward()
        opt.step()

        if epoch % CHECK_EVERY == 0:
            with torch.no_grad():
                pred = np.zeros(len(val_X))
                for i in range(n_experts):
                    mask = val_bucket == i
                    if mask.sum() > 0:
                        pred[mask] = experts[i](Xv[mask]).cpu().numpy()
            val_mse = ((val_y - pred) ** 2).mean()
            if val_mse < best_val:
                best_val, best_state, no_improve = val_mse, copy.deepcopy(experts.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE_CHECKS:
                    break
    experts.load_state_dict(best_state)
    return experts


def train_datadriven_es(train_X, train_y, val_X, val_y, in_dim, n_experts=N_EXPERTS, seed=SEED, lr=LR):
    torch.manual_seed(seed)
    experts = nn.ModuleList([ExpertMLP(in_dim) for _ in range(n_experts)]).to(DEVICE)
    gate = GatingNet(in_dim, n_experts).to(DEVICE)
    opt = torch.optim.Adam(list(experts.parameters()) + list(gate.parameters()), lr=lr)
    Xt = torch.tensor(train_X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(train_y, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(val_X, dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(val_y, dtype=torch.float32, device=DEVICE)

    best_val, best_state, no_improve = np.inf, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        opt.zero_grad()
        gate_w = gate(Xt)
        expert_out = torch.stack([e(Xt) for e in experts], dim=-1)
        pred = (gate_w * expert_out).sum(-1)
        mse = nn.functional.mse_loss(pred, yt)
        usage = gate_w.mean(0)
        lb_loss = ((usage - 1.0 / n_experts) ** 2).sum()
        (mse + LB_WEIGHT * lb_loss).backward()
        opt.step()

        if epoch % CHECK_EVERY == 0:
            with torch.no_grad():
                gate_w_v = gate(Xv)
                expert_out_v = torch.stack([e(Xv) for e in experts], dim=-1)
                pred_v = (gate_w_v * expert_out_v).sum(-1)
                val_mse = nn.functional.mse_loss(pred_v, yv).item()
            if val_mse < best_val:
                best_val = val_mse
                best_state = (copy.deepcopy(experts.state_dict()), copy.deepcopy(gate.state_dict()))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE_CHECKS:
                    break
    experts.load_state_dict(best_state[0])
    gate.load_state_dict(best_state[1])
    return experts, gate


def predict_explicit(experts, X, bucket, n_experts=N_EXPERTS):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    bucket = np.asarray(bucket)
    pred = np.zeros(len(X))
    with torch.no_grad():
        for i in range(n_experts):
            mask = bucket == i
            if mask.sum() > 0:
                pred[mask] = experts[i](X_t[mask]).cpu().numpy()
    return pred


def predict_datadriven(experts, gate, X):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        gate_w = gate(X_t)
        expert_out = torch.stack([e(X_t) for e in experts], dim=-1)
        pred = (gate_w * expert_out).sum(-1).cpu().numpy()
        dominant = gate_w.argmax(dim=-1).cpu().numpy()
    return pred, dominant, gate_w.cpu().numpy()


def run_walk_forward(panel: pd.DataFrame, seed: int = SEED, n_experts: int = N_EXPERTS, lr: float = LR):
    dates = np.sort(panel["Date"].unique())
    rebalance_dates = dates[TRAIN_WINDOW_DAYS::STEP_DAYS]

    records, dominant_log = [], []
    for i in range(len(rebalance_dates) - 1):
        cutoff, next_cutoff = rebalance_dates[i], rebalance_dates[i + 1]
        window_start = dates[max(0, np.searchsorted(dates, cutoff) - TRAIN_WINDOW_DAYS)]
        full_train_df = panel[(panel["Date"] >= window_start) & (panel["Date"] < cutoff)]
        test_df = panel[(panel["Date"] >= cutoff) & (panel["Date"] < next_cutoff)]
        if len(test_df) == 0 or len(full_train_df) < 500:
            continue

        train_df, val_df = _split_train_val(full_train_df)
        scaler = StandardScaler().fit(train_df[FEATURES])
        train_X, val_X, test_X = scaler.transform(train_df[FEATURES]), scaler.transform(val_df[FEATURES]), scaler.transform(test_df[FEATURES])
        train_y, val_y = train_df[TARGET].values, val_df[TARGET].values

        quantile_pts = np.linspace(0, 1, n_experts + 1)[1:-1]
        thresholds = train_df["vol_20"].quantile(quantile_pts).values
        train_bucket = np.digitize(train_df["vol_20"].values, thresholds)
        val_bucket = np.digitize(val_df["vol_20"].values, thresholds)
        test_bucket = np.digitize(test_df["vol_20"].values, thresholds)

        explicit_experts = train_explicit_es(train_X, train_y, train_bucket, val_X, val_y, val_bucket, len(FEATURES), n_experts, seed, lr)
        explicit_pred = predict_explicit(explicit_experts, test_X, test_bucket, n_experts)

        dd_experts, dd_gate = train_datadriven_es(train_X, train_y, val_X, val_y, len(FEATURES), n_experts, seed, lr)
        dd_pred, dd_dominant, _ = predict_datadriven(dd_experts, dd_gate, test_X)

        top_bucket = n_experts - 1
        high_vol_mask = test_bucket == top_bucket
        dominant_high_vol = int(np.bincount(dd_dominant[high_vol_mask], minlength=n_experts).argmax()) if high_vol_mask.sum() > 0 else -1
        dominant_log.append({"window_end": cutoff, "dominant_high_vol_expert": dominant_high_vol})

        rec = test_df[["Date", "ticker", "sector", TARGET, "fwd_ret_1"]].copy() if "sector" in test_df.columns else test_df[["Date", "ticker", TARGET, "fwd_ret_1"]].copy()
        rec["explicit_pred"], rec["datadriven_pred"] = explicit_pred, dd_pred
        records.append(rec)

    return pd.concat(records, ignore_index=True), pd.DataFrame(dominant_log)


def strategy_returns(results, pred_col):
    df = results.sort_values(["ticker", "Date"]).copy()
    df["position"] = (TARGET_VOL / df[pred_col].clip(lower=0.01)).clip(upper=3)
    df["position_prev"] = df.groupby("ticker")["position"].shift(1).fillna(0)
    df["turnover"] = (df["position"] - df["position_prev"]).abs()
    df["strategy_ret"] = df["position"] * df["fwd_ret_1"] - (COST_BP / 10000) * df["turnover"]
    return df.groupby("Date")["strategy_ret"].mean()


def newey_west_mean_var(d: np.ndarray, max_lag: int) -> float:
    t = len(d)
    d_c = d - d.mean()
    var = (d_c**2).sum() / t
    for lag in range(1, max_lag + 1):
        cov = (d_c[:-lag] * d_c[lag:]).sum() / t
        var += 2 * (1 - lag / (max_lag + 1)) * cov
    return var / t


def diebold_mariano(loss_a, loss_b, max_lag=4):
    d = (loss_a - loss_b).dropna().values
    dm_stat = d.mean() / np.sqrt(newey_west_mean_var(d, max_lag))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value, len(d)


def block_bootstrap_sharpe_diff(ret_a, ret_b, n_boot=5000, block_size=20, seed=0):
    rng = np.random.default_rng(seed)
    a, b = ret_a.values, ret_b.values
    t = len(a)
    n_blocks = t // block_size
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, t - block_size, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
        sa, sb = a[idx], b[idx]
        diffs[i] = sa.mean() / sa.std() * np.sqrt(252) - sb.mean() / sb.std() * np.sqrt(252)
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5])
