"""Stress-slice IC comparison for Table 8: where RG-ResMoE's edge over
MLP-L concentrates across market conditions.

Compares RG-ResMoE against MLP-L on five date slices of the test period:
full sample, the top decile of days by market_vol_20, the COVID crash
(2020-02 to 2020-06), the 2022 bear market (full calendar year), and
windows +/-10 trading days around a high/low market_vol_20 regime flip
(median split). For each slice, reports the number of days, the
seed-mean-first IC advantage (Delta IC), that slice's gain relative to
the full-sample gain, and a Diebold-Mariano significance test (Newey-West,
lag 4), matching the paper's Section 4.4 testing protocol.

Usage (from the repo root, after training rg_resmoe and mlp_l):
    N_SEEDS=30 python src/stress_slices.py
Output:
    results/summary/stress_slices.txt  (also printed to stdout)
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT_FILE = "results/summary/stress_slices.txt"
BASE     = "results/predictions"
PROCESSED_DIR = "data/processed"
TARGET   = "target_fwd_vol_5"
N_SEEDS  = int(os.environ.get("N_SEEDS", "30"))

REF, REF_COL = f"{BASE}/rg_resmoe/seed{{s}}.parquet", "residual_pred"
CMP, CMP_COL = f"{BASE}/mlp_l/seed{{s}}.parquet", "plain_pred"


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


def nw_var(d, max_lag=4):
    t = len(d); dc = d - d.mean(); v = (dc**2).sum() / t
    for lag in range(1, min(max_lag, t - 1) + 1):
        v += 2*(1 - lag/(max_lag+1))*(dc[:-lag]*dc[lag:]).sum()/t
    return max(v/t, 1e-30)


def dm_test(diff):
    diff = diff.dropna().values
    if len(diff) < 2:
        return np.nan, np.nan
    stat = diff.mean() / np.sqrt(nw_var(diff))
    return stat, 2*(1 - norm.cdf(abs(stat)))


def build_slices():
    panel = pd.read_parquet(f"{PROCESSED_DIR}/panel_systemic.parquet", columns=["Date", "market_vol_20"])
    mvol = panel.groupby("Date")["market_vol_20"].first().sort_index()
    dates = mvol.index.values

    top_decile = set(mvol[mvol >= mvol.quantile(0.90)].index)

    median = mvol.median()
    regime = mvol >= median
    flips = regime.index[regime.ne(regime.shift(1)).fillna(False)]
    flip_window = set()
    for fd in flips:
        idx = mvol.index.searchsorted(fd)
        lo, hi = max(0, idx - 10), min(len(dates), idx + 11)
        flip_window.update(dates[lo:hi])

    covid = set(mvol.index[(mvol.index >= "2020-02-01") & (mvol.index <= "2020-06-30")])
    y2022 = set(mvol.index[(mvol.index >= "2022-01-01") & (mvol.index <= "2022-12-31")])

    return {
        "Full sample":              set(dates),
        "Top market-vol decile":    top_decile,
        "COVID crisis":             covid,
        "2022 bear market":         y2022,
        "Regime-flip windows":      flip_window,
    }


def daily_ic_per_seed(pat, col, n_seeds):
    per_seed = []
    for s in range(n_seeds):
        p = pat.format(s=s)
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        per_seed.append(fast_ic_by_date(df, col))
    return per_seed


os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
_fh = open(OUT_FILE, "w")
def _print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=_fh)

slices = build_slices()
ref_daily = daily_ic_per_seed(REF, REF_COL, N_SEEDS)
cmp_daily = daily_ic_per_seed(CMP, CMP_COL, N_SEEDS)
_print(f"RG-ResMoE: {len(ref_daily)}/{N_SEEDS} seeds loaded, MLP-L: {len(cmp_daily)}/{N_SEEDS} seeds loaded")

ref_mean = pd.concat(ref_daily, axis=1).mean(axis=1)
cmp_mean = pd.concat(cmp_daily, axis=1).mean(axis=1)
diff = (ref_mean - cmp_mean).dropna()

full_sample_delta = None
_print(f"\n{'Slice':24s}  {'Days':>6}  {'Delta IC':>10}  {'Rel. gain':>10}  {'p':>10}")
for name, slice_dates in slices.items():
    d = diff[diff.index.isin(slice_dates)]
    if len(d) == 0:
        continue
    delta = d.mean()
    if full_sample_delta is None:
        full_sample_delta = delta
    rel_gain = delta / full_sample_delta if full_sample_delta else np.nan
    stat, p = dm_test(d)
    _print(f"{name:24s}  {len(d):>6}  {delta:>+10.4f}  {rel_gain:>9.1f}x  {p:>10.2e}")

_fh.close()
print(f"\nSaved to {OUT_FILE}")
