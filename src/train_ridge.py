"""Pooled linear (Ridge) baselines.

Two variants, mirroring the HAR conventions of
train_classical_baselines.py exactly (same walk-forward windows,
pooled fit on the full 504-day train window, Ridge(alpha=1.0),
deterministic single run):

  ridge        — the same 16 stock-level features the neural models use
  ridge_state  — 16 features + the 2 regime state variables (input
                 pathway, the linear analogue of MLP+state)

Unlike HAR (3 same-scale vol features), the 16-feature set mixes
scales, so features are z-scored with train-window statistics, same
convention as the neural models. alpha in {0.1, 1, 10} is evaluated
and printed for a flatness check; alpha=1.0 (HAR's value) is shipped.

Outputs: results/predictions/classical/ridge.parquet (ridge_pred),
         ridge_state.parquet (ridge_state_pred).
Run (from the repo root): python src/train_ridge.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from walk_forward import FEATURES, add_lag_features  # noqa: E402

DATA_PATH = "data/processed/panel_systemic.parquet"
OUT_DIR = "results/predictions/classical"
TARGET = "target_fwd_vol_5"
STATE = ["market_vol_20", "idio_vol_20"]
TRAIN_WINDOW_DAYS = 504
STEP_DAYS = 63
ALPHAS = [0.1, 1.0, 10.0]
SHIP_ALPHA = 1.0


def walk_forward_windows(dates):
    rebalance = dates[TRAIN_WINDOW_DAYS::STEP_DAYS]
    for cutoff in rebalance:
        idx = np.searchsorted(dates, cutoff)
        window_start = dates[max(0, idx - TRAIN_WINDOW_DAYS)]
        test_dates = dates[idx: idx + STEP_DAYS]
        if len(test_dates) == 0:
            continue
        yield window_start, cutoff, test_dates[0], test_dates[-1]


def run_variant(panel, feats, pred_col):
    dates = np.sort(panel["Date"].unique())
    records = {a: [] for a in ALPHAS}
    for w_idx, (wstart, cutoff, tstart, tend) in enumerate(walk_forward_windows(dates)):
        train_df = panel[(panel["Date"] >= wstart) & (panel["Date"] < cutoff)].dropna(subset=feats + [TARGET])
        test_df = panel[(panel["Date"] >= tstart) & (panel["Date"] <= tend)].dropna(subset=feats).copy()
        scaler = StandardScaler().fit(train_df[feats])
        X_tr = scaler.transform(train_df[feats])
        X_te = scaler.transform(test_df[feats])
        y_tr = train_df[TARGET].values
        for a in ALPHAS:
            m = Ridge(alpha=a, fit_intercept=True).fit(X_tr, y_tr)
            rec = test_df[["Date", "ticker", TARGET, "fwd_ret_1"]].copy()
            rec[pred_col] = m.predict(X_te)
            records[a].append(rec)
        print(f"  [{pred_col}] window {w_idx + 1} done", flush=True)
    return {a: pd.concat(r, ignore_index=True) for a, r in records.items()}


def pooled_stats(df, pred_col):
    ic = (df.groupby("Date")
            .apply(lambda g: g[pred_col].corr(g[TARGET], method="spearman"),
                   include_groups=False).mean())
    rmse = np.sqrt(((df[TARGET] - df[pred_col]) ** 2).mean())
    p = np.maximum(df[pred_col].values, 1e-4)
    t = np.maximum(df[TARGET].values, 1e-4)
    r = (t / p) ** 2
    ql = (r - np.log(r) - 1.0).mean()
    return ic, rmse, ql


panel = add_lag_features(pd.read_parquet(DATA_PATH))
os.makedirs(OUT_DIR, exist_ok=True)

for feats, pred_col, fname in [
    (FEATURES, "ridge_pred", "ridge.parquet"),
    (FEATURES + STATE, "ridge_state_pred", "ridge_state.parquet"),
]:
    out = run_variant(panel, feats, pred_col)
    print(f"\n== {pred_col}: alpha flatness check ==")
    for a, df in out.items():
        ic, rmse, ql = pooled_stats(df, pred_col)
        ship = "  <- shipped" if a == SHIP_ALPHA else ""
        print(f"  alpha={a:<5}  IC={ic:.4f}  RMSE={rmse:.4f}  QLIKE={ql:.3f}  "
              f"minpred={df[pred_col].min():.4f}{ship}")
    out[SHIP_ALPHA].to_parquet(f"{OUT_DIR}/{fname}", index=False)
    print(f"saved {OUT_DIR}/{fname}  ({len(out[SHIP_ALPHA]):,} rows)")
