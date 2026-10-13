# -*- coding: utf-8 -*-
"""Residual expert MoE + data sharing.

Hand-crafted (rule-based) hard-routing variants of RG-ResMoE: the learned
gate is replaced by a fixed regime taxonomy, with optional data sharing:

  - f_global is trained on ALL data and frozen (same as before).
  - Each correction expert r_k trains on ALL samples, but samples from its
    own bucket get weight=1 and samples from other buckets get weight=lam:

      loss_k = mean_i [ w(i,k) * (f_global(x_i) + r_k(x_i) - y_i)^2
                       + alpha * w(i,k) * r_k(x_i)^2 ]
      w(i,k) = 1.0  if sample i is in bucket k
               lam  otherwise

  - lam=0 : reduces to original residual_experts (no sharing)
  - lam=1 : all samples equal weight (experts see same data, no specialization)
  - lam in (0,1): data sharing, experts specialise but borrow from neighbours

Hyperparameter selection: sweep (lam x alpha) by validation MSE.

Usage:
  SEED=0 python src/train_rule_gate_moe.py {vol|systemic_idio|sector}

"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from walk_forward import (
    CHECK_EVERY,
    DEVICE,
    ExpertMLP,
    FEATURES,
    LR,
    MAX_EPOCHS,
    N_EXPERTS,
    PATIENCE_CHECKS,
    PROCESSED_DIR,
    RESULTS_DIR,
    STEP_DAYS,
    TARGET,
    TRAIN_WINDOW_DAYS,
    _split_train_val,
    diebold_mariano,
)

# Hand-crafted hard-routing gates (paper's regime-taxonomy variants) and
# their frozen, validation-selected configs:
#   vol           volatility-quantile assignment      (lr 0.01, lam 0.0)
#   systemic_idio market x idiosyncratic median split (lr 0.01, lam 0.0)
#   sector        GICS sectors, one expert per sector (lr 0.03, lam 0.1)
FROZEN = {"vol": (0.01, 0.0), "systemic_idio": (0.01, 0.0), "sector": (0.03, 0.1)}
GATE        = sys.argv[1] if len(sys.argv) > 1 else "vol"
RESIDUAL_LR = float(os.environ.get("EXPERT_LR", str(FROZEN[GATE][0])))
LAM         = float(os.environ.get("LAM", str(FROZEN[GATE][1])))
FROZEN_ALPHA = 1.0
ALPHAS      = [float(x) for x in os.environ.get("ALPHA", str(FROZEN_ALPHA)).split(",")]
SEED        = int(os.environ.get("SEED", "0"))
HIDDEN      = int(os.environ.get("EXPERT_HIDDEN", "16"))
GLOBAL_LR   = float(os.environ.get("BASE_LR", "0.03"))

_GATE_DIR = {"vol": "rule_gate_vol_quantiles", "sector": "rule_gate_sectors",
             "systemic_idio": "rule_gate_market_idio"}
RULE_GATE_RESULTS_DIR = RESULTS_DIR / _GATE_DIR[GATE]


def compute_val_ic(pred, target, dates, min_n=20):
    """Mean cross-sectional Spearman IC on the validation split, grouped by
    date. Used to select hyperparameters by the metric the paper actually
    reports (IC), not just validation MSE (the training loss) -- these two
    don't necessarily agree, especially near strong shrinkage (high alpha).
    """
    df = pd.DataFrame({"pred": pred, "target": target, "Date": dates})
    ics = []
    for _, g in df.groupby("Date"):
        if len(g) < min_n:
            continue
        ic = spearmanr(g["pred"], g["target"])[0]
        if np.isfinite(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan


def make_buckets(train_df, val_df, test_df, gate, sector_to_id=None):
    if gate == "vol":
        n_exp = int(os.environ.get("VOL_N_EXPERTS", N_EXPERTS))
        quantile_pts = np.linspace(0, 1, n_exp + 1)[1:-1]
        thresholds = train_df["vol_20"].quantile(quantile_pts).values
        train_bucket = np.digitize(train_df["vol_20"].values, thresholds)
        val_bucket   = np.digitize(val_df["vol_20"].values,   thresholds)
        test_bucket  = np.digitize(test_df["vol_20"].values,  thresholds)
        return train_bucket, val_bucket, test_bucket, n_exp
    elif gate == "systemic_idio":
        mkt_med  = train_df["market_vol_20"].median()
        idio_med = train_df["idio_vol_20"].median()
        def bucket_of(df):
            mkt_high  = (df["market_vol_20"].values >= mkt_med).astype(int)
            idio_high = (df["idio_vol_20"].values   >= idio_med).astype(int)
            return mkt_high * 2 + idio_high
        return bucket_of(train_df), bucket_of(val_df), bucket_of(test_df), 4
    elif gate == "sector":
        n_exp = len(sector_to_id)
        return (train_df["sector"].map(sector_to_id).values,
                val_df["sector"].map(sector_to_id).values,
                test_df["sector"].map(sector_to_id).values, n_exp)
    else:
        raise ValueError(gate)


def train_global(train_X, train_y, val_X, val_y, in_dim, seed=0, hidden=32, lr=LR):
    torch.manual_seed(seed)
    model = ExpertMLP(in_dim, hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(train_X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(train_y, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(val_X,   dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(val_y,   dtype=torch.float32, device=DEVICE)
    best_val, best_state, no_improve = np.inf, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        nn.functional.mse_loss(model(Xt), yt).backward()
        opt.step()
        if epoch % CHECK_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_mse = nn.functional.mse_loss(model(Xv), yv).item()
            if val_mse < best_val:
                best_val, best_state, no_improve = val_mse, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE_CHECKS:
                    break
    model.load_state_dict(best_state)
    es_tag = f"ES@{epoch}" if epoch < MAX_EPOCHS else f"FULL({MAX_EPOCHS})"
    print(f"    [global] {es_tag}  best_val={best_val:.5f}")
    return model


def train_residual_experts(f_global, train_X, train_y, train_bucket,
                                      val_X, val_y, val_bucket, val_dates,
                                      in_dim, n_experts, alpha, lam,
                                      seed=0, residual_lr=LR, hidden=32):
    torch.manual_seed(seed)
    experts = nn.ModuleList([ExpertMLP(in_dim, hidden=hidden) for _ in range(n_experts)]).to(DEVICE)
    for e in experts:
        nn.init.zeros_(e.net[-1].weight)
        nn.init.zeros_(e.net[-1].bias)
    opt = torch.optim.Adam(experts.parameters(), lr=residual_lr)

    Xt = torch.tensor(train_X,    dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(train_y,    dtype=torch.float32, device=DEVICE)
    bt = torch.tensor(train_bucket, dtype=torch.long,  device=DEVICE)
    Xv = torch.tensor(val_X,      dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        global_train = f_global(Xt)
        global_val   = f_global(Xv)

    best_val, best_state, no_improve = np.inf, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        experts.train()
        opt.zero_grad()
        loss = 0.0
        for i in range(n_experts):
            # weight=1 for own bucket, weight=lam for others
            w = torch.where(bt == i,
                            torch.ones(len(train_X),      device=DEVICE),
                            torch.full((len(train_X),), lam, device=DEVICE))
            r    = experts[i](Xt)
            pred = global_train + r
            loss = loss + (w * (pred - yt)**2).mean() + alpha * (w * r**2).mean()
        loss.backward()
        opt.step()

        if epoch % CHECK_EVERY == 0:
            experts.eval()
            with torch.no_grad():
                pred = global_val.clone()
                for i in range(n_experts):
                    mask = val_bucket == i
                    if mask.sum() > 0:
                        pred[mask] = global_val[mask] + experts[i](Xv[mask])
                val_mse = ((val_y - pred.cpu().numpy())**2).mean()
            if val_mse < best_val:
                best_val, best_state, no_improve = val_mse, copy.deepcopy(experts.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE_CHECKS:
                    break
    experts.load_state_dict(best_state)
    experts.eval()
    with torch.no_grad():
        final_pred = global_val.clone()
        for i in range(n_experts):
            mask = val_bucket == i
            if mask.sum() > 0:
                final_pred[mask] = global_val[mask] + experts[i](Xv[mask])
    best_val_ic = compute_val_ic(final_pred.cpu().numpy(), val_y, val_dates)
    es_tag = f"ES@{epoch}" if epoch < MAX_EPOCHS else f"FULL({MAX_EPOCHS})"
    print(f"    [residual+share lam={lam}] {es_tag}  best_val_mse={best_val:.5f}  best_val_ic={best_val_ic:.5f}")
    return experts, best_val, best_val_ic


def predict_residual(f_global, experts, X, bucket, n_experts):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    f_global.eval()
    experts.eval()
    with torch.no_grad():
        global_pred = f_global(X_t).cpu().numpy()
        pred = global_pred.copy()
        for i in range(n_experts):
            mask = bucket == i
            if mask.sum() > 0:
                pred[mask] = global_pred[mask] + experts[i](X_t[mask]).cpu().numpy()
    return pred


def run_walk_forward(panel, alpha, lam, gate, seed=0, residual_lr=LR,
                     sector_to_id=None, hidden=32):
    dates = np.sort(panel["Date"].unique())
    rebalance_dates = dates[TRAIN_WINDOW_DAYS::STEP_DAYS]

    records, val_mses, val_ics = [], [], []
    for i in range(len(rebalance_dates) - 1):
        cutoff, next_cutoff = rebalance_dates[i], rebalance_dates[i + 1]
        window_start = dates[max(0, np.searchsorted(dates, cutoff) - TRAIN_WINDOW_DAYS)]
        full_train_df = panel[(panel["Date"] >= window_start) & (panel["Date"] < cutoff)]
        test_df       = panel[(panel["Date"] >= cutoff)       & (panel["Date"] < next_cutoff)]
        if len(test_df) == 0 or len(full_train_df) < 500:
            continue

        train_df, val_df = _split_train_val(full_train_df)
        scaler = StandardScaler().fit(train_df[FEATURES])
        train_X  = scaler.transform(train_df[FEATURES])
        val_X    = scaler.transform(val_df[FEATURES])
        test_X   = scaler.transform(test_df[FEATURES])
        train_y  = train_df[TARGET].values
        val_y    = val_df[TARGET].values
        val_dates = val_df["Date"].values

        train_bucket, val_bucket, test_bucket, n_experts = make_buckets(
            train_df, val_df, test_df, gate, sector_to_id)

        f_global = train_global(train_X, train_y, val_X, val_y, len(FEATURES), seed, hidden, GLOBAL_LR)
        for p in f_global.parameters():
            p.requires_grad_(False)

        experts, best_val, best_val_ic = train_residual_experts(
            f_global, train_X, train_y, train_bucket,
            val_X, val_y, val_bucket, val_dates,
            len(FEATURES), n_experts, alpha, lam, seed, residual_lr, hidden)

        pred = predict_residual(f_global, experts, test_X, test_bucket, n_experts)
        val_mses.append(best_val)
        val_ics.append(best_val_ic)

        rec = test_df[["Date", "ticker", "vol_20", TARGET, "fwd_ret_1"]].copy()
        rec["residual_pred"] = pred
        rec["bucket"]        = test_bucket
        records.append(rec)

    return pd.concat(records, ignore_index=True), np.mean(val_mses), np.nanmean(val_ics)


def main():
    panel = pd.read_parquet(PROCESSED_DIR / "panel_systemic.parquet")
    print(f"gate={GATE}  global_lr={GLOBAL_LR}  residual_lr={RESIDUAL_LR}  lam={LAM}  alphas={ALPHAS}  hidden={HIDDEN}  seed={SEED}")

    RULE_GATE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sector_to_id = None
    if GATE == "sector":
        sectors = sorted(panel["sector"].unique())
        sector_to_id = {s: i for i, s in enumerate(sectors)}

    summary = []
    results_by_alpha = {}
    for alpha in ALPHAS:
        print(f"\n=== gate={GATE} lr={RESIDUAL_LR} lam={LAM} alpha={alpha} ===")
        results, mean_val_mse, mean_val_ic = run_walk_forward(
            panel, alpha=alpha, lam=LAM, gate=GATE, seed=SEED,
            residual_lr=RESIDUAL_LR, sector_to_id=sector_to_id, hidden=HIDDEN)
        tw_tag = f"_tw{TRAIN_WINDOW_DAYS}" if TRAIN_WINDOW_DAYS != 504 else ""
        alpha_tag = "" if alpha == FROZEN_ALPHA else f"_alpha{alpha}"
        fname = f"seed{SEED}{tw_tag}{alpha_tag}.parquet"
        results.to_parquet(RULE_GATE_RESULTS_DIR / fname, index=False)
        results_by_alpha[alpha] = results
        summary.append({"lam": LAM, "alpha": alpha, "val_mse": mean_val_mse, "val_ic": mean_val_ic})
        print(f"  mean_val_mse={mean_val_mse:.5f}  mean_val_ic={mean_val_ic:.5f}")

    df = pd.DataFrame(summary)
    best_alpha = df.loc[df["val_ic"].idxmax(), "alpha"]
    tw_tag = f"_tw{TRAIN_WINDOW_DAYS}" if TRAIN_WINDOW_DAYS != 504 else ""
    val_csv = RULE_GATE_RESULTS_DIR / f"val_scores_seed{SEED}{tw_tag}.csv"
    df.to_csv(val_csv, index=False)
    print(f"\n>>> best alpha={best_alpha} (by val_ic)  (saved {val_csv.name})")

    # Quick comparison vs plain MLP on test
    results = results_by_alpha[best_alpha]
    target  = results[TARGET]
    moe_mse = ((target - results['residual_pred'])**2).mean()
    print(f"\n=== TEST: gate={GATE} lam={LAM} alpha={best_alpha} ===")
    print(f"MoE  test_mse={moe_mse:.5f}")


if __name__ == "__main__":
    main()
