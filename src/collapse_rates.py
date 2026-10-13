"""Per-seed QLIKE collapse-rate table.

The stability exhibit behind the 30-seed narrative: for every model and
seed, compute mean QLIKE (CLIP=1e-4 as everywhere) and flag "collapse"
seeds (QLIKE > THRESH, default 2 ~ 3x the healthy level of ~0.74). For
each collapse seed, locate the month where the daily-QLIKE mass
concentrates, so the paper can say *when* each model breaks.

Run (from the repo root): python src/collapse_rates.py
Output: results/summary/collapse_rate_by_model.csv (per-seed) + printed summary.
Models whose prediction files are absent (e.g. ablations not trained) are skipped.
"""
import os
import pathlib

import numpy as np
import pandas as pd

TARGET = "target_fwd_vol_5"
CLIP = 1e-4
THRESH = float(os.environ.get("COLLAPSE_THRESH", "2.0"))
N_SEEDS = int(os.environ.get("N_SEEDS", "30"))

BASE = "results/predictions"

MODELS = {
    # label: (pattern, pred col)
    "RG-ResMoE":              (f"{BASE}/rg_resmoe/seed{{s}}.parquet", "residual_pred"),
    "RG-ResMoE-K2":           (f"{BASE}/rg_resmoe_k2_matched/seed{{s}}.parquet", "residual_pred"),
    "RG-ResMoE-K2-unmatched": (f"{BASE}/rg_resmoe_k2/seed{{s}}.parquet", "residual_pred"),
    "RG-ResMoE-K6":           (f"{BASE}/rg_resmoe_k6_matched/seed{{s}}.parquet", "residual_pred"),
    "learned top-1 (hard)":   (f"{BASE}/rg_resmoe_hard_routing/seed{{s}}.parquet", "residual_pred"),
    "RG-ResMoE (-z)":         (f"{BASE}/rg_resmoe_no_states/seed{{s}}.parquet", "residual_pred"),
    "market x idio split":    (f"{BASE}/rule_gate_market_idio/seed{{s}}.parquet", "residual_pred"),
    "volatility quantiles":   (f"{BASE}/rule_gate_vol_quantiles/seed{{s}}.parquet", "residual_pred"),
    "GICS sectors":           (f"{BASE}/rule_gate_sectors/seed{{s}}.parquet", "residual_pred"),
    "Standard MoE":           (f"{BASE}/standard_moe/seed{{s}}.parquet", "residual_pred"),
    "random init":            (f"{BASE}/rg_resmoe_random_init/seed{{s}}.parquet", "residual_pred"),
    "MLP-S":                  (f"{BASE}/mlp_s/seed{{s}}.parquet", "plain_pred"),
    "MLP-L":                  (f"{BASE}/mlp_l/seed{{s}}.parquet", "plain_pred"),
    "MLP-S(+z)":              (f"{BASE}/mlp_s_state/seed{{s}}.parquet", "plain_pred"),
    "MLP-L(+z)":              (f"{BASE}/mlp_l_state/seed{{s}}.parquet", "plain_pred"),
}


def qlike(pred, target):
    p = np.maximum(pred, CLIP)
    t = np.maximum(target, CLIP)
    r = (t / p) ** 2
    return r - np.log(r) - 1.0


rows = []
for label, (pat, col) in MODELS.items():
    for s in range(N_SEEDS):
        f = pathlib.Path(pat.format(s=s))
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["Date", col, TARGET])
        ql = qlike(df[col].values, df[TARGET].values)
        m = float(ql.mean())
        rec = {"model": label, "seed": s, "qlike": m,
               "min_pred": float(df[col].min())}
        if m > THRESH:
            daily = pd.Series(ql).groupby(df["Date"].values).mean()
            worst = daily.idxmax()
            top = daily.sort_values(ascending=False).head(60)
            rec["worst_day"] = str(pd.Timestamp(worst).date())
            rec["worst_month"] = str(pd.Timestamp(worst).to_period("M"))
            rec["top60_months"] = ";".join(
                pd.Series(pd.to_datetime(top.index)).dt.to_period("M").astype(str).value_counts().head(3).index)
        rows.append(rec)

os.makedirs("results/summary", exist_ok=True)
out = pd.DataFrame(rows)
out.to_csv("results/summary/collapse_rate_by_model.csv", index=False)

print(f"collapse threshold: QLIKE > {THRESH}\n")
print(f"{'model':24s} {'n':>3} {'collapse':>8} {'median_q':>9} {'max_q':>12}  collapse months")
print("-" * 95)
for label, g in out.groupby("model", sort=False):
    coll = g[g.qlike > THRESH]
    months = ";".join(sorted(set(coll.get("worst_month", pd.Series(dtype=str)).dropna()))) if len(coll) else ""
    print(f"{label:24s} {len(g):3d} {len(coll):8d} {g.qlike.median():9.3f} {g.qlike.max():12.1f}  {months}")

print("saved results/summary/collapse_rate_by_model.csv")
