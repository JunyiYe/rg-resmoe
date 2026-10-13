"""Diebold-Mariano significance tests for the pathway/routing/ablation
tables (Tables 4-6).

Same seed-mean-first methodology as dm_tests.py, applied to every
architecture-variant comparison: pathway variants (+z vs gate) vs the
headline RG-ResMoE, hard-routing variants, and the expert-count/residual
-design ablations (K2, K6, random init, Standard MoE).

Also prints each variant's mean IC +/- std across seeds.

Run (from the repo root): python src/dm_tests_variants.py
Env: N_SEEDS (default 30).
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

TARGET  = "target_fwd_vol_5"
CLIP    = 1e-4
N_SEEDS = int(os.environ.get("N_SEEDS", "30"))
US      = "results/predictions"

MODELS = {
    "RG-ResMoE":   (f"{US}/rg_resmoe/seed{{s}}.parquet", "residual_pred"),
    "RG-ResMoE-K2":      (f"{US}/rg_resmoe_k2_matched/seed{{s}}.parquet", "residual_pred"),
    "RG-ResMoE-K2-unmatched": (f"{US}/rg_resmoe_k2/seed{{s}}.parquet", "residual_pred"),
    "RG-ResMoE-K6":      (f"{US}/rg_resmoe_k6_matched/seed{{s}}.parquet", "residual_pred"),
    "learned top-1 (hard)":     (f"{US}/rg_resmoe_hard_routing/seed{{s}}.parquet", "residual_pred"),
    "random init":   (f"{US}/rg_resmoe_random_init/seed{{s}}.parquet", "residual_pred"),
    "RG-ResMoE (-z)":  (f"{US}/rg_resmoe_no_states/seed{{s}}.parquet", "residual_pred"),
    "Standard MoE":     (f"{US}/standard_moe/seed{{s}}.parquet", "residual_pred"),
    "market x idio split": (f"{US}/rule_gate_market_idio/seed{{s}}.parquet", "residual_pred"),
    "volatility quantiles":     (f"{US}/rule_gate_vol_quantiles/seed{{s}}.parquet", "residual_pred"),
    "GICS sectors":  (f"{US}/rule_gate_sectors/seed{{s}}.parquet", "residual_pred"),
    "MLP-L":    (f"{US}/mlp_l/seed{{s}}.parquet", "plain_pred"),
    "MLP-S":    (f"{US}/mlp_s/seed{{s}}.parquet", "plain_pred"),
    "MLP-L(+z)": (f"{US}/mlp_l_state/seed{{s}}.parquet", "plain_pred"),
    "MLP-S(+z)": (f"{US}/mlp_s_state/seed{{s}}.parquet", "plain_pred"),
}

# (A, B): DM on d_t = seed-mean[ X_A(t) - X_B(t) ]
PAIRS = [
    # tab:variants — everything vs the headline model
    ("RG-ResMoE", "RG-ResMoE (-z)"),
    ("RG-ResMoE", "learned top-1 (hard)"),
    ("RG-ResMoE", "volatility quantiles"),
    ("RG-ResMoE", "GICS sectors"),
    ("RG-ResMoE", "market x idio split"),
    ("RG-ResMoE", "RG-ResMoE-K2-unmatched"),
    ("RG-ResMoE", "RG-ResMoE-K6"),
    ("RG-ResMoE", "Standard MoE"),
    ("RG-ResMoE", "random init"),
    ("RG-ResMoE", "MLP-L(+z)"),
    # tab:pathway — each 18f MLP vs its own 16f baseline
    ("MLP-L(+z)", "MLP-L"),
    ("MLP-S(+z)", "MLP-S"),
    # ladder: unmatched K=2 (soft) vs each hard variant
    ("RG-ResMoE-K2-unmatched", "learned top-1 (hard)"),
    ("RG-ResMoE-K2-unmatched", "volatility quantiles"),
    ("RG-ResMoE-K2-unmatched", "GICS sectors"),
    ("RG-ResMoE-K2-unmatched", "market x idio split"),
    # Standard MoE vs the plain MLPs
    ("Standard MoE", "MLP-L"),
    ("Standard MoE", "MLP-S"),
]

METRICS = ["IC", "RMSE", "R2", "QLIKE"]


def nw_var(d, max_lag=4):
    t = len(d); dc = d - d.mean(); v = (dc ** 2).sum() / t
    for lag in range(1, max_lag + 1):
        v += 2 * (1 - lag / (max_lag + 1)) * (dc[:-lag] * dc[lag:]).sum() / t
    return max(v / t, 1e-30)


def dm_stat(diff):
    stat = diff.mean() / np.sqrt(nw_var(diff))
    return stat, 2 * (1 - norm.cdf(abs(stat)))


def daily_ic(df, col, min_n=20):
    d = df.copy()
    d["_rp"] = d.groupby("Date")[col].rank()
    d["_rt"] = d.groupby("Date")[TARGET].rank()
    cnt = d.groupby("Date")["_rp"].transform("count")
    d = d[cnt >= min_n]
    mp = d.groupby("Date")["_rp"].transform("mean")
    mt = d.groupby("Date")["_rt"].transform("mean")
    d["_cp"] = d["_rp"] - mp
    d["_ct"] = d["_rt"] - mt
    num = d.groupby("Date").apply(lambda g: (g["_cp"] * g["_ct"]).sum(), include_groups=False)
    den = d.groupby("Date").apply(
        lambda g: np.sqrt((g["_cp"] ** 2).sum() * (g["_ct"] ** 2).sum()), include_groups=False)
    return (num / den).replace([np.inf, -np.inf], np.nan).dropna()


def all_daily(df, col):
    out = {"IC": daily_ic(df, col)}
    g = df.groupby("Date")
    out["RMSE"] = g.apply(lambda x: np.sqrt(((x[TARGET] - x[col]) ** 2).mean()), include_groups=False)
    out["R2"] = g.apply(
        lambda x: 1 - ((x[TARGET] - x[col]) ** 2).sum()
        / max(((x[TARGET] - x[TARGET].mean()) ** 2).sum(), 1e-10), include_groups=False)

    def q(x):
        p = np.maximum(x[col].values, CLIP)
        t = np.maximum(x[TARGET].values, CLIP)
        r = (t / p) ** 2
        return (r - np.log(r) - 1.0).mean()
    out["QLIKE"] = g.apply(q, include_groups=False)
    return out


# ── pass 1: per-(model, seed) daily series ─────────────────────────────────
series = {}   # (model, seed) -> {metric: Series}
ic_means = {} # model -> list of per-seed time-mean IC
for name, (pat, col) in MODELS.items():
    ic_means[name] = []
    n_loaded = 0
    for s in range(N_SEEDS):
        p = pat.format(s=s)
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p, columns=["Date", "ticker", col, TARGET])
        series[(name, s)] = all_daily(df, col)
        ic_means[name].append(series[(name, s)]["IC"].mean())
        n_loaded += 1
        del df
    print(f"loaded {name}: {n_loaded}/{N_SEEDS} seeds", flush=True)

print("\n== per-variant IC over seeds (time-mean of daily IC) ==")
for name, v in ic_means.items():
    if v:
        print(f"  {name:14s} IC = {np.mean(v):.4f} +/- {np.std(v, ddof=1):.4f}  (n={len(v)})")

# ── pass 2: pairwise DM ─────────────────────────────────────────────────────
hdr = (f"\n{'A (ref)':14s} {'B':14s} {'metric':6s} {'delta':>10} "
       f"{'DM_new':>8} {'p_new':>8} {'sig':4} {'n_days':>6}")
print(hdr)
print("-" * len(hdr))
for a, b in PAIRS:
    seeds_a = sorted(s for (n, s) in series if n == a)
    seeds_b = sorted(s for (n, s) in series if n == b)
    if not seeds_a or not seeds_b:
        print(f"{a:14s} {b:14s} MISSING FILES, skipped")
        continue
    for metric in METRICS:
        diffs = []
        for i, sa in enumerate(seeds_a):
            sb = seeds_b[min(i, len(seeds_b) - 1)]
            diffs.append((series[(a, sa)][metric] - series[(b, sb)][metric]).dropna())
        d_t = pd.concat(diffs, axis=1).mean(axis=1).dropna()
        stat, p = dm_stat(d_t.values)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"{a:14s} {b:14s} {metric:6s} {d_t.mean():>+10.5f} "
              f"{stat:>+8.3f} {p:>8.4f} {sig:4} {len(d_t):>6}")
    print()
