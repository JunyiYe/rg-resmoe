"""Value-at-Risk / Kupiec coverage backtest for the VaR table (Table 7).

Converts each model's five-day forward volatility forecast into a daily
Value-at-Risk threshold, assuming the standardized residual
(next-day return / implied daily vol) follows a Student-t distribution
(fat-tailed, unlike a Normal assumption that would call every model
miscalibrated regardless of forecast quality). The t shape (degrees of
freedom and scale) is fit once per (model, seed) by matching the empirical
excess kurtosis and variance of the pooled test-period standardized
residuals -- a global nuisance parameter, not a per-sample forecast, so
this does not use any future information relative to each prediction.

For each ticker, each alpha in {5%, 1%}, and each market-volatility regime
(all dates / above-median / below-median market_vol_20), runs Kupiec's
(1995) proportion-of-failures test and reports the fraction of tickers
whose coverage is rejected at the 5% level. Significance of RG-ResMoE's
lower rejection rate against each baseline is a paired t-test across the
30 training seeds on that fraction, matching the paper's Section 4.4
testing protocol.

Usage (from the repo root, after training rg_resmoe/mlp_l/mlp_s):
    N_SEEDS=30 python src/var_backtest.py
Output:
    results/summary/var_backtest.txt  (also printed to stdout)
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

OUT_FILE = "results/summary/var_backtest.txt"
BASE     = "results/predictions"
PROCESSED_DIR = "data/processed"
TARGET   = "target_fwd_vol_5"
RET      = "fwd_ret_1"
N_SEEDS  = int(os.environ.get("N_SEEDS", "30"))
ALPHAS   = [0.05, 0.01]

REF = "RG-ResMoE"
MODELS = {
    "RG-ResMoE": (f"{BASE}/rg_resmoe/seed{{s}}.parquet", "residual_pred"),
    "MLP-L":     (f"{BASE}/mlp_l/seed{{s}}.parquet",     "plain_pred"),
    "MLP-S":     (f"{BASE}/mlp_s/seed{{s}}.parquet",     "plain_pred"),
}


def fit_t_dof_scale(standardized_resid: np.ndarray):
    """Method-of-moments (dof, scale) Student-t fit, matching the
    empirical excess kurtosis (-> dof) and variance (-> scale) of the
    standardized residuals. scipy's unscaled t has Var[t_df] = df/(df-2),
    which does not match empirically fat/thin-tailed residuals at scale=1."""
    r = standardized_resid[np.isfinite(standardized_resid)]
    var_r = np.var(r)
    m2, m4 = np.mean(r**2), np.mean(r**4)
    excess_kurt = m4 / m2**2 - 3.0
    dof = 30.0 if excess_kurt <= 0 else float(np.clip(6.0 / excess_kurt + 4.0, 4.1, 60.0))
    scale = float(np.sqrt(var_r * (dof - 2) / dof))
    return dof, scale


def kupiec_pof(violations: np.ndarray, alpha: float):
    n = len(violations)
    x = violations.sum()
    pi_hat = x / n
    if pi_hat in (0, 1):
        return np.nan
    ll_null = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
    ll_alt = x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)
    lr = -2 * (ll_null - ll_alt)
    return 1 - stats.chi2.cdf(lr, df=1)


def load_regime(df: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_parquet(f"{PROCESSED_DIR}/panel_systemic.parquet", columns=["Date", "market_vol_20"])
    daily = panel.groupby("Date")["market_vol_20"].first()
    median = daily.median()
    regime = (daily >= median).map({True: "high", False: "low"}).rename("regime").reset_index()
    return df.merge(regime, on="Date", how="left")


def backtest_one(df: pd.DataFrame, pred_col: str, alpha: float) -> dict:
    """Per-ticker Kupiec rejection fraction, split all/high/low regime."""
    d = df.copy()
    d["sigma_hat"] = d[pred_col].clip(lower=0.01) / np.sqrt(252)
    d["std_resid"] = d[RET] / d["sigma_hat"]
    dof, scale = fit_t_dof_scale(d["std_resid"].values)
    var = -d["sigma_hat"].values * scale * stats.t.ppf(alpha, dof)
    d["violation"] = (d[RET].values < -var).astype(int)

    out = {}
    for regime in ["all", "high", "low"]:
        sub = d if regime == "all" else d[d["regime"] == regime]
        rejects, n_tickers = 0, 0
        for _, g in sub.groupby("ticker"):
            if len(g) < 20:
                continue
            p = kupiec_pof(g["violation"].values, alpha)
            n_tickers += 1
            rejects += int(np.isfinite(p) and p < 0.05)
        out[regime] = rejects / n_tickers if n_tickers else np.nan
    return out


os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
_fh = open(OUT_FILE, "w")
def _print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=_fh)

rows = []
for model, (pat, col) in MODELS.items():
    for seed in range(N_SEEDS):
        path = pat.format(s=seed)
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)[["Date", "ticker", RET, col]]
        df = load_regime(df)
        for alpha in ALPHAS:
            frac = backtest_one(df, col, alpha)
            for regime, val in frac.items():
                rows.append({"model": model, "seed": seed, "alpha": alpha, "regime": regime, "frac_kupiec_reject": val})
    _print(f"  {model}: done")

full = pd.DataFrame(rows)

_print(f"\n{'Slice':20s}  {REF:>10}" + "".join(f"  {m:>18}" for m in MODELS if m != REF))
for alpha in ALPHAS:
    for regime in ["all", "high", "low"]:
        sub = full[(full["alpha"] == alpha) & (full["regime"] == regime)]
        ref_vals = sub[sub["model"] == REF]["frac_kupiec_reject"].dropna().values
        line = f"{f'{int(alpha*100)}% VaR, {regime}':20s}  {ref_vals.mean()*100:>9.1f}%"
        for model in MODELS:
            if model == REF:
                continue
            m_vals = sub[sub["model"] == model]["frac_kupiec_reject"].dropna().values
            if len(m_vals) == 0:
                line += f"  {'(no data)':>18}"
                continue
            n = min(len(ref_vals), len(m_vals))
            if n < 2:
                # The paired t-test needs >= 2 seeds (the paper uses 30).
                # Below that, still report the rejection level itself rather
                # than dropping the baseline column from the table.
                sig = "1 seed"
            else:
                t, p = stats.ttest_rel(ref_vals[:n], m_vals[:n])
                sig = "n.s." if p >= 0.05 else f"p={p:.4f}"
            line += f"  {m_vals.mean()*100:>6.1f}% ({sig:>7})"
        _print(line)

_fh.close()
print(f"\nSaved to {OUT_FILE}")
