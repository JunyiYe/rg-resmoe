"""Headline summary table for the Japan TSE Prime replication (Table 9).

Same metrics as summarize_results.py (IC, RMSE, QLIKE), restricted to the
four models in the paper's cross-market replication: MLP-L, MLP-L(+z),
Standard MoE, RG-ResMoE, all trained with MARKET=jp (see README).

Usage (from the repo root, after training the JP models):
    N_SEEDS=30 python src/summarize_results_jp.py
Output:
    results/summary/final_metrics_jp.txt  (also printed to stdout)
"""
import os, datetime
import numpy as np
import pandas as pd

OUT_FILE = "results/summary/final_metrics_jp.txt"
BASE     = "results/predictions/jp"
TARGET   = "target_fwd_vol_5"
CLIP     = 1e-4
N_SEEDS  = int(os.environ.get("N_SEEDS", "30"))
COLLAPSE_THRESH = float(os.environ.get("COLLAPSE_THRESH", "2.0"))

MODELS = {
    "MLP-L":        (f"{BASE}/mlp_l/seed{{s}}.parquet",       "plain_pred",    N_SEEDS),
    "MLP-L(+z)":    (f"{BASE}/mlp_l_state/seed{{s}}.parquet", "plain_pred",    N_SEEDS),
    "Standard MoE": (f"{BASE}/standard_moe/seed{{s}}.parquet","residual_pred", N_SEEDS),
    "RG-ResMoE":    (f"{BASE}/rg_resmoe/seed{{s}}.parquet",   "residual_pred", N_SEEDS),
}


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
    num = df.groupby("Date").apply(lambda g: (g["_cp"]*g["_ct"]).sum(), include_groups=False)
    den = df.groupby("Date").apply(
        lambda g: np.sqrt((g["_cp"]**2).sum()*(g["_ct"]**2).sum()), include_groups=False)
    return (num/den).replace([np.inf, -np.inf], np.nan).dropna().values


def qlike(pred, target):
    p = np.maximum(pred, CLIP)
    t = np.maximum(target, CLIP)
    r = (t / p) ** 2
    return (r - np.log(r) - 1.0).mean()


def load_rows(pat, col, ns):
    rows = []
    for s in range(ns):
        p = pat.format(s=s)
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        rows.append({
            "ic":    fast_ic(df, col),
            "rmse":  np.sqrt(((df[TARGET]-df[col])**2).mean()),
            "qlike": qlike(df[col].values, df[TARGET].values),
        })
    return rows


def _fmt(vals, fmt=".4f", width=16):
    arr = np.array([v for v in vals if np.isfinite(v)])
    if len(arr) == 0:
        return "—".rjust(width)
    s = format(arr.mean(), fmt) + " +/-" + format(arr.std(), fmt)
    return (s + " " * width)[:width]


os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
_fh = open(OUT_FILE, "w")
def _print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=_fh)

_print(f"\nRun: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  (Japan TSE Prime replication)")

cache = {}
for name, (pat, col, ns) in MODELS.items():
    cache[name] = load_rows(pat, col, ns)
    _print(f"  {name}: {len(cache[name])}/{ns} files loaded")

W = 16
hdr = f"  {'Model':14s}  {'IC':>{W}}  {'RMSE':>{W}}  {'QLIKE':>{W}}  {'Collapse':>10}  n"
_print(f"\n{hdr}")
_print("  " + "-"*(len(hdr)-2))
for name, rows in cache.items():
    if not rows:
        _print(f"  {name:14s}  (no data)")
        continue
    n_collapsed = sum(1 for r in rows if r["qlike"] > COLLAPSE_THRESH)
    _print(f"  {name:14s}"
           f"  {_fmt([r['ic'].mean() for r in rows])}"
           f"  {_fmt([r['rmse'] for r in rows])}"
           f"  {_fmt([r['qlike'] for r in rows], '.2f')}"
           f"  {n_collapsed:>6d}/{len(rows):<3d}"
           f"  {len(rows)}")
_fh.close()
print(f"Saved to {OUT_FILE}")
