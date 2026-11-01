"""Diebold-Mariano significance tests for the headline table (Table 3).

For each model pair and metric, averages the per-day metric difference
across the 30 training seeds first,

    d_t = (1/30) * sum_s [ X_{A,s}(t) - X_{B,s}(t) ],

then runs one DM test on the resulting ~1,890-day series with Newey-West
(lag 4) standard errors, as described in the paper's Section 4.4.

Run (from the repo root): python src/dm_tests.py
Env: N_SEEDS (default 30).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

BASE    = "results/predictions"
TARGET  = "target_fwd_vol_5"
RET     = "fwd_ret_1"
CLIP    = 1e-4
N_SEEDS = int(os.environ.get("N_SEEDS", "30"))

REF = ("RG-ResMoE",
       f"{BASE}/rg_resmoe/seed{{s}}.parquet",
       "residual_pred", N_SEEDS)

BASELINES = {
    "MLP-L":        (f"{BASE}/mlp_l/seed{{s}}.parquet", "plain_pred", N_SEEDS),
    "MLP-S":        (f"{BASE}/mlp_s/seed{{s}}.parquet", "plain_pred", N_SEEDS),
    "Standard MoE": (f"{BASE}/standard_moe/seed{{s}}.parquet", "residual_pred", N_SEEDS),
    "HAR":          (f"{BASE}/classical/har.parquet",         "har_pred",         1),
    "Ridge":        (f"{BASE}/classical/ridge.parquet",       "ridge_pred",       1),
    "Ridge(+z)":  (f"{BASE}/classical/ridge_state.parquet", "ridge_state_pred", 1),
    "GARCH(1,1)":   (f"{BASE}/classical/garch.parquet",       "garch_pred",       1),
    "Persistence":  (f"{BASE}/classical/persistence.parquet", "hist_vol_pred",    1),
}

METRICS = ["IC", "RMSE", "MAE", "R2", "QLIKE", "IVolSR"]


def nw_var(d, max_lag=4):
    t = len(d); dc = d - d.mean(); v = (dc ** 2).sum() / t
    for lag in range(1, max_lag + 1):
        v += 2 * (1 - lag / (max_lag + 1)) * (dc[:-lag] * dc[lag:]).sum() / t
    return max(v / t, 1e-30)


def dm_stat(diff):
    stat = diff.mean() / np.sqrt(nw_var(diff))
    return stat, 2 * (1 - norm.cdf(abs(stat)))


def daily_ic(m, col, min_n=20):
    """Per-date Spearman IC as a Date-indexed Series."""
    df = m.copy()
    df["_rp"] = df.groupby("Date")[col].rank()
    df["_rt"] = df.groupby("Date")[TARGET].rank()
    cnt = df.groupby("Date")["_rp"].transform("count")
    df = df[cnt >= min_n]
    mp = df.groupby("Date")["_rp"].transform("mean")
    mt = df.groupby("Date")["_rt"].transform("mean")
    df["_cp"] = df["_rp"] - mp
    df["_ct"] = df["_rt"] - mt
    num = df.groupby("Date").apply(lambda g: (g["_cp"] * g["_ct"]).sum(), include_groups=False)
    den = df.groupby("Date").apply(
        lambda g: np.sqrt((g["_cp"] ** 2).sum() * (g["_ct"] ** 2).sum()), include_groups=False)
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def daily_series(m, col, metric):
    """One value per Date for the given metric, as a Date-indexed Series."""
    if metric == "IC":
        return daily_ic(m, col)
    if metric == "RMSE":
        return m.groupby("Date").apply(
            lambda g: np.sqrt(((g[TARGET] - g[col]) ** 2).mean()), include_groups=False)
    if metric == "MAE":
        return m.groupby("Date").apply(
            lambda g: (g[TARGET] - g[col]).abs().mean(), include_groups=False)
    if metric == "R2":
        return m.groupby("Date").apply(
            lambda g: 1 - ((g[TARGET] - g[col]) ** 2).sum()
            / max(((g[TARGET] - g[TARGET].mean()) ** 2).sum(), 1e-10), include_groups=False)
    if metric == "QLIKE":
        def q(g):
            p = np.maximum(g[col].values, CLIP)
            t = np.maximum(g[TARGET].values, CLIP)
            r = (t / p) ** 2
            return (r - np.log(r) - 1.0).mean()
        return m.groupby("Date").apply(q, include_groups=False)
    if metric == "IVolSR":
        def pr(g):
            if len(g) < 20:
                return np.nan
            w = 1.0 / np.maximum(g[col].values, 1e-6)
            w = w / w.sum()
            return (w * g[RET].values).sum()
        return m.groupby("Date").apply(pr, include_groups=False).dropna()
    raise ValueError(metric)


def load(pat, n):
    out = []
    for s in range(n):
        p = pat.format(s=s)
        if os.path.exists(p):
            out.append(pd.read_parquet(p))
    return out


# daily_series() reads only Date, its prediction column, TARGET and RET, and
# an inner merge keeps the reference's rows in their original order, so the
# same per-day series gets rebuilt again and again: the reference's once per
# (baseline, metric), and a single-seed classical baseline's once per
# reference seed. Fingerprint exactly the columns the series depends on and
# compute each distinct one once -- over 8 baselines x 6 metrics that is 6
# passes over the reference instead of 48. Nothing is assumed about which
# baselines cover the full panel: GARCH drops ~1.3k reference rows, so its
# merge fingerprints differently and is computed on its own, as before.
_series_cache = {}


def cached_series(m, col, metric, sig):
    key = (metric, col) + sig
    out = _series_cache.get(key)
    if out is None:
        out = _series_cache[key] = daily_series(m, col, metric)
    return out


def col_sig(m, col, shared):
    return shared + (hash(m[col].values.tobytes()),)


# Mirror the report to results/summary/ as well as stdout, same convention
# as summarize_results.py / var_backtest.py / stress_slices.py.
OUT_FILE = "results/summary/dm_tests.txt"
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
_fh = open(OUT_FILE, "w")


def _print(*args, **kwargs):
    print(*args, **kwargs)
    kwargs.pop("file", None)
    print(*args, **kwargs, file=_fh)


_print(f"Loading reference {REF[0]} ({REF[3]} seeds)...")
ref_dfs = load(REF[1], REF[3])
_print(f"  {len(ref_dfs)} loaded")

hdr = f"{'Baseline':18s} {'metric':7s} {'delta':>10} {'DM':>8} {'p':>8} {'sig':4}"
_print(hdr)
_print("-" * len(hdr))

for bname, (bpat, bcol, bn) in BASELINES.items():
    b_dfs = load(bpat, bn)
    if not b_dfs:
        _print(f"{bname}: no files, skipped")
        continue
    rcol = REF[2]
    bc_m = bcol + "_b" if bcol == rcol else bcol
    per_seed_diffs = {metric: [] for metric in METRICS}
    for i, rdf in enumerate(ref_dfs):
        bdf = b_dfs[min(i, len(b_dfs) - 1)]
        m = rdf[["Date", "ticker", rcol]].merge(
            bdf[["Date", "ticker", bcol, TARGET, RET]].rename(columns={bcol: bc_m}),
            on=["Date", "ticker"])
        shared = (len(m),) + tuple(
            hash(m[c].values.tobytes()) for c in ("Date", TARGET, RET))
        ref_sig, cmp_sig = col_sig(m, rcol, shared), col_sig(m, bc_m, shared)
        for metric in METRICS:
            dq = cached_series(m, rcol, metric, ref_sig)
            db = cached_series(m, bc_m, metric, cmp_sig)
            per_seed_diffs[metric].append((dq - db).dropna())
    for metric in METRICS:
        # Average the per-day differences across seeds, then one DM test.
        d_t = pd.concat(per_seed_diffs[metric], axis=1).mean(axis=1).dropna().values
        stat, p = dm_stat(d_t)
        delta = float(d_t.mean())
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        _print(f"{bname:18s} {metric:7s} {delta:>+10.5f} {stat:>+8.3f} {p:>8.4f} {sig:4}")
    _print()

_fh.close()
print(f"\nSaved to {OUT_FILE}")
