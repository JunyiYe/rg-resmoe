"""Headline summary table from the persisted per-seed predictions.

For every model: IC, ICIR, RMSE, MAE, R2, QLIKE, and inverse-volatility
Sharpe, reported mean +/- std across training seeds (classical baselines
are deterministic, single run). This reproduces the paper's main
forecasting-results table; significance shading comes from
src/dm_tests.py and the collapse column from src/collapse_rates.py.

Usage (from the repo root, after training):
    N_SEEDS=30 python src/summarize_results.py
Output:
    results/summary/final_metrics.txt  (also printed to stdout)
"""
import os, datetime
import numpy as np
import pandas as pd
from scipy.stats import norm

OUT_FILE = "results/summary/final_metrics.txt"
BASE    = "results/predictions"
TARGET  = "target_fwd_vol_5"
RET     = "fwd_ret_1"
CLIP    = 1e-4
N_SEEDS = int(os.environ.get("N_SEEDS", "30"))  # neural models; classical baselines are deterministic

MODELS = {
    "RG-ResMoE":      (f"{BASE}/rg_resmoe/seed{{s}}.parquet",            "residual_pred", N_SEEDS),
    "RG-ResMoE-K2":   (f"{BASE}/rg_resmoe_k2_matched/seed{{s}}.parquet", "residual_pred", N_SEEDS),
    "RG-ResMoE-K6":   (f"{BASE}/rg_resmoe_k6_matched/seed{{s}}.parquet", "residual_pred", N_SEEDS),
    "MLP-L":          (f"{BASE}/mlp_l/seed{{s}}.parquet",                "plain_pred",    N_SEEDS),
    "MLP-S":          (f"{BASE}/mlp_s/seed{{s}}.parquet",                "plain_pred",    N_SEEDS),
    "Standard MoE":   (f"{BASE}/standard_moe/seed{{s}}.parquet",         "residual_pred", N_SEEDS),
    "HAR":            (f"{BASE}/classical/har.parquet",         "har_pred",         1),
    "Ridge":          (f"{BASE}/classical/ridge.parquet",       "ridge_pred",       1),
    "Ridge+state":    (f"{BASE}/classical/ridge_state.parquet", "ridge_state_pred", 1),
    "GARCH(1,1)":     (f"{BASE}/classical/garch.parquet",       "garch_pred",       1),
    "Persistence":    (f"{BASE}/classical/persistence.parquet", "hist_vol_pred",    1),
}

# ── metric functions ──────────────────────────────────────────────────────────
def fast_ic(df, col, min_n=20):
    df = df.copy()
    df["_rp"] = df.groupby("Date")[col].rank()
    df["_rt"] = df.groupby("Date")[TARGET].rank()
    cnt = df.groupby("Date")["_rp"].transform("count")
    df  = df[cnt >= min_n]
    mp  = df.groupby("Date")["_rp"].transform("mean")
    mt  = df.groupby("Date")["_rt"].transform("mean")
    df["_cp"] = df["_rp"] - mp
    df["_ct"] = df["_rt"] - mt
    num = df.groupby("Date").apply(lambda g: (g["_cp"]*g["_ct"]).sum(),
                                   include_groups=False)
    den = df.groupby("Date").apply(
        lambda g: np.sqrt((g["_cp"]**2).sum()*(g["_ct"]**2).sum()),
        include_groups=False)
    return (num/den).replace([np.inf,-np.inf], np.nan).dropna().values

def decile_ic(df, col, q_lo, q_hi, min_n=5):
    lo  = df.groupby("Date")[col].transform(lambda x: x.quantile(q_lo))
    hi  = df.groupby("Date")[col].transform(lambda x: x.quantile(q_hi))
    sub = df[(df[col] >= lo) & (df[col] <= hi)].copy()
    cnt = sub.groupby("Date")[col].transform("count")
    sub = sub[cnt >= min_n]
    if sub.empty:
        return np.array([np.nan])
    return fast_ic(sub, col, min_n=min_n)

def qlike(pred, target):
    p = np.maximum(pred,   CLIP)
    t = np.maximum(target, CLIP)
    r = (t / p) ** 2
    return (r - np.log(r) - 1.0).mean()

def ivol_sharpe(df, col, min_n=20):
    daily = []
    for _, g in df.groupby("Date"):
        if len(g) < min_n:
            continue
        w = 1.0 / np.maximum(g[col].values, 1e-6)
        w = w / w.sum()
        daily.append((w * g[RET].values).sum())
    d = np.array(daily)
    if len(d) < 10 or d.std() < 1e-10:
        return np.nan, np.nan, d
    return d.mean() / d.std() * np.sqrt(252), d.mean() * 252, d

def nw_var(d, max_lag=4):
    t = len(d); dc = d - d.mean(); v = (dc**2).sum() / t
    for lag in range(1, max_lag + 1):
        v += 2*(1-lag/(max_lag+1))*(dc[:-lag]*dc[lag:]).sum()/t
    return max(v/t, 1e-30)

def dm(diff):
    stat = diff.mean() / np.sqrt(nw_var(diff))
    return stat, 2*(1-norm.cdf(abs(stat)))

# ── dual output ───────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
_fh = open(OUT_FILE, "w")
def _print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=_fh)

SEP = "="*92

# ── load files ────────────────────────────────────────────────────────────────
_print(f"\nRun: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_rows(pat, col, ns):
    rows = []
    for s in range(ns):
        p = pat.format(s=s) if "{s}" in pat else pat
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        ic_arr = fast_ic(df, col)
        sr, ann, d_daily = ivol_sharpe(df, col)
        rows.append({
            "ic":       ic_arr,
            "rmse":     np.sqrt(((df[TARGET]-df[col])**2).mean()),
            "mae":      (df[TARGET]-df[col]).abs().mean(),
            "r2":       1 - ((df[TARGET]-df[col])**2).sum()/((df[TARGET]-df[TARGET].mean())**2).sum(),
            "qlike":    qlike(df[col].values, df[TARGET].values),
            "ivol_sr":  sr,
            "daily":    d_daily,
            "df": df, "col": col,
        })
    return rows

cache = {}
for name, (pat, col, ns) in MODELS.items():
    cache[name] = load_rows(pat, col, ns)
    _print(f"  {name}: {len(cache[name])}/{ns} files loaded")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Summary table
# ═══════════════════════════════════════════════════════════════════════════════
_print(f"\n{SEP}")
_print("  SECTION 1 — SUMMARY TABLE  (mean +/- std across seeds; single-seed models show mean only)")
_print(SEP)
W = 16  # width for mean +/- std columns
hdr = (f"  {'Model':22s}  {'IC':>{W}}  {'ICIR':>{W}}  {'IC t':>8}  {'IC>0%':>7}"
       f"  {'RMSE':>{W}}  {'MAE':>{W}}  {'R2':>{W}}  {'QLIKE':>{W}}  {'IVolSR':>{W}}  n")
_print(hdr)
_print("  " + "-"*(len(hdr)-2))

def _fmt(vals, fmt=".4f"):
    arr = np.array([v for v in vals if np.isfinite(v)])
    if len(arr) == 0: return "—".rjust(W)
    if len(arr) == 1: return ("{:" + str(W) + fmt + "}").format(arr[0])
    s = format(arr.mean(), fmt) + " +/-" + format(arr.std(), fmt)
    return (s + " " * W)[:W]

for name, rows in cache.items():
    if not rows:
        _print(f"  {name:22s}  (no data)")
        continue
    ic_means = np.array([r["ic"].mean() for r in rows])
    ic_stds  = np.array([r["ic"].std()  for r in rows])
    icir_vals= ic_means / ic_stds
    ic_tstat = ic_means.mean() / (ic_stds.mean() / np.sqrt(len(rows[0]["ic"])))
    ic_pos   = np.mean([(r['ic']>0).mean()*100 for r in rows])
    _print(f"  {name:22s}"
           f"  {_fmt(ic_means)}"
           f"  {_fmt(icir_vals, '.3f')}"
           f"  {ic_tstat:>8.2f}"
           f"  {ic_pos:>7.1f}"
           f"  {_fmt([r['rmse'] for r in rows])}"
           f"  {_fmt([r['mae'] for r in rows])}"
           f"  {_fmt([r['r2'] for r in rows])}"
           f"  {_fmt([r['qlike'] for r in rows], '.2f')}"
           f"  {_fmt([r['ivol_sr'] for r in rows], '.4f')}"
           f"  {len(rows)}")

_print(SEP)
_fh.close()
print(f"Saved to {OUT_FILE}")
