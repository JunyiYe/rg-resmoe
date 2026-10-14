"""Diebold-Mariano significance tests for the Japan replication (Section 8).

Same seed-mean-first methodology as dm_tests.py: average the per-day metric
difference across seeds first, then one DM test (Newey-West, lag 4) on the
daily series. Compares RG-ResMoE against MLP-L on IC, RMSE, and QLIKE.

Usage (from the repo root, after training the JP models):
    N_SEEDS=30 python src/dm_tests_jp.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

BASE    = "results/predictions/jp"
TARGET  = "target_fwd_vol_5"
CLIP    = 1e-4
N_SEEDS = int(os.environ.get("N_SEEDS", "30"))

MODELS = {
    "RG-ResMoE": (f"{BASE}/rg_resmoe/seed{{s}}.parquet",   "residual_pred"),
    "MLP-L":     (f"{BASE}/mlp_l/seed{{s}}.parquet",       "plain_pred"),
}
PAIRS = [("RG-ResMoE", "MLP-L")]
METRICS = ["IC", "RMSE", "QLIKE"]


def fast_ic_by_date(df, col, min_n=20):
    df = df.copy()
    df["_rp"] = df.groupby("Date")[col].rank()
    df["_rt"] = df.groupby("Date")[TARGET].rank()
    cnt = df.groupby("Date")["_rp"].transform("count")
    df  = df[cnt >= min_n]
    mp  = df.groupby("Date")["_rp"].transform("mean")
    mt  = df.groupby("Date")["_rt"].transform("mean")
    df["_cp"] = df["_rp"] - mp
    df["_ct"] = df["_rt"] - mt
    num = df.groupby("Date").apply(lambda g: (g["_cp"]*g["_ct"]).sum(), include_groups=False)
    den = df.groupby("Date").apply(
        lambda g: np.sqrt((g["_cp"]**2).sum()*(g["_ct"]**2).sum()), include_groups=False)
    return (num/den).replace([np.inf, -np.inf], np.nan)


def per_day_metrics(df, col):
    ic = fast_ic_by_date(df, col)
    err2 = (df[TARGET] - df[col])**2
    rmse_t = np.sqrt(err2.groupby(df["Date"]).mean())
    p = np.maximum(df[col].values, CLIP)
    t = np.maximum(df[TARGET].values, CLIP)
    r = (t / p)**2
    ql = pd.Series(r - np.log(r) - 1.0, index=df.index).groupby(df["Date"]).mean()
    return pd.DataFrame({"IC": ic, "RMSE": rmse_t, "QLIKE": ql})


def load_daily(pat, col, n_seeds):
    per_seed = []
    for s in range(n_seeds):
        p = pat.format(s=s)
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        per_seed.append(per_day_metrics(df, col))
    return per_seed


def nw_var(d, max_lag=4):
    t = len(d); dc = d - d.mean(); v = (dc**2).sum() / t
    for lag in range(1, max_lag + 1):
        v += 2*(1 - lag/(max_lag+1))*(dc[:-lag]*dc[lag:]).sum()/t
    return max(v/t, 1e-30)


def dm_test(diff):
    diff = diff.dropna()
    stat = diff.mean() / np.sqrt(nw_var(diff.values))
    return stat, 2*(1 - norm.cdf(abs(stat))), len(diff)


daily = {name: load_daily(pat, col, N_SEEDS) for name, (pat, col) in MODELS.items()}
for name, series in daily.items():
    print(f"{name}: {len(series)}/{N_SEEDS} seeds loaded")

print(f"\n{'A':12s} {'B':8s} {'metric':6s} {'seed-mean diff':>16s} {'DM stat':>10s} {'p':>10s} {'n':>6s}")
for a, b in PAIRS:
    if not daily[a] or not daily[b]:
        missing = a if not daily[a] else b
        print(f"{a} vs {b}: skipped, no seeds loaded for {missing}")
        continue
    for metric in METRICS:
        a_df = pd.concat([s[metric] for s in daily[a]], axis=1).mean(axis=1)
        b_df = pd.concat([s[metric] for s in daily[b]], axis=1).mean(axis=1)
        d = (a_df - b_df).dropna()
        stat, p, n = dm_test(d)
        print(f"{a:12s} {b:8s} {metric:6s} {d.mean():>16.5f} {stat:>10.2f} {p:>10.2e} {n:>6d}")
