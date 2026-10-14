# -*- coding: utf-8 -*-
"""RG-ResMoE(-z): the main model's no-state-variables control.

Identical to train_rg_resmoe.py except the learned soft gate sees ONLY the
16 stock-level features -- the two regime state variables z never enter the
model anywhere. Comparing this against RG-ResMoE isolates the value of the
state information itself (the "-z" row of the pathway table).

Every other gate axis (vol, systemic_idio, sector) uses an EXPLICIT,
hard-coded bucket assignment (make_buckets() in
train_rule_gate_moe.py) to decide which correction expert
handles which sample, plus a lam-weighted data-sharing mechanism between
buckets. Here, instead, a GatingNet produces a continuous softmax blend
over the n_experts correction networks, trained jointly with them:

    gate_w(x)     = softmax(GatingNet(x))            # shape (n_experts,)
    correction(x) = sum_k gate_w_k(x) * r_k(x)
    pred(x)       = f_global(x) + correction(x)
    loss          = MSE(pred, y) + alpha * mean(correction(x)^2)
                    + LB_WEIGHT * load_balancing_penalty

There is no `lam` parameter here: the soft blend already lets every sample
partially train every expert in proportion to gate_w, which plays the same
role lam plays for the hard-bucket gates (partial cross-bucket sharing) --
adding a separate lam on top would double up the same mechanism.

Usage: SEED=0 python src/train_rg_resmoe_no_states.py
Output: results/predictions/rg_resmoe_no_states/seed{SEED}.parquet
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
    GatingNet,
    LB_WEIGHT,
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
)

RESIDUAL_LR = float(os.environ.get("EXPERT_LR", "0.01"))
FROZEN_ALPHA = 1.0
ALPHAS      = [float(x) for x in os.environ.get("ALPHA", str(FROZEN_ALPHA)).split(",")]
SEED        = int(os.environ.get("SEED", "0"))
HIDDEN      = int(os.environ.get("EXPERT_HIDDEN", "16"))
GLOBAL_LR   = float(os.environ.get("BASE_LR", "0.03"))
LB_WEIGHT   = float(os.environ.get("LB_WEIGHT", "0.5"))


DATADRIVEN_RESULTS_DIR = RESULTS_DIR / "rg_resmoe_no_states"


def compute_val_ic(pred, target, dates, min_n=20):
    """Mean cross-sectional Spearman IC on the validation split, grouped by
    date -- see train_rule_gate_moe.py for why this is tracked
    alongside val MSE (they don't necessarily agree on which hyperparameters
    are best)."""
    df = pd.DataFrame({"pred": pred, "target": target, "Date": dates})
    ics = []
    for _, g in df.groupby("Date"):
        if len(g) < min_n:
            continue
        ic = spearmanr(g["pred"], g["target"])[0]
        if np.isfinite(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan


def train_global(train_X, train_y, val_X, val_y, in_dim, seed=0, hidden=16, lr=LR):
    torch.manual_seed(seed)
    model = ExpertMLP(in_dim, hidden=hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(train_X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(train_y, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(val_X,   dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(val_y,   dtype=torch.float32, device=DEVICE)
    n = len(train_X)
    max_epochs = MAX_EPOCHS
    check_every = CHECK_EVERY
    patience = PATIENCE_CHECKS
    best_val, best_state, no_improve = np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        nn.functional.mse_loss(model(Xt), yt).backward()
        opt.step()
        if epoch % check_every == 0:
            model.eval()
            with torch.no_grad():
                val_mse = nn.functional.mse_loss(model(Xv), yv).item()
            if val_mse < best_val:
                best_val, best_state, no_improve = val_mse, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
    model.load_state_dict(best_state)
    es_tag = f"ES@{epoch}" if epoch < max_epochs else f"FULL({max_epochs})"
    print(f"    [global] {es_tag}  best_val={best_val:.5f}")
    return model


def train_residual_datadriven(f_global, train_X, train_y, val_X, val_y, val_dates,
                               in_dim, n_experts, alpha, seed=0, residual_lr=LR, hidden=16):
    torch.manual_seed(seed)
    experts = nn.ModuleList([ExpertMLP(in_dim, hidden=hidden) for _ in range(n_experts)]).to(DEVICE)
    for e in experts:
        nn.init.zeros_(e.net[-1].weight)
        nn.init.zeros_(e.net[-1].bias)
    gate = GatingNet(in_dim, n_experts).to(DEVICE)
    opt = torch.optim.Adam(list(experts.parameters()) + list(gate.parameters()), lr=residual_lr)

    Xt = torch.tensor(train_X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(train_y, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(val_X,   dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        global_train = f_global(Xt)
        global_val   = f_global(Xv)

    n = len(train_X)
    max_epochs = MAX_EPOCHS
    check_every = CHECK_EVERY
    patience = PATIENCE_CHECKS

    def step(idx=None):
        gt  = global_train if idx is None else global_train[idx]
        xt  = Xt if idx is None else Xt[idx]
        yt_ = yt if idx is None else yt[idx]
        gate_w = gate(xt)
        r_out  = torch.stack([e(xt) for e in experts], dim=-1)
        correction = (gate_w * r_out).sum(-1)
        pred = gt + correction
        mse  = nn.functional.mse_loss(pred, yt_)
        usage = gate_w.mean(0)
        lb_loss = ((usage - 1.0 / n_experts) ** 2).sum()
        loss = mse + alpha * (correction ** 2).mean() + LB_WEIGHT * lb_loss
        loss.backward()

    best_val, best_state, no_improve = np.inf, None, 0
    for epoch in range(1, max_epochs + 1):
        experts.train(); gate.train()
        opt.zero_grad()
        step()
        opt.step()

        if epoch % check_every == 0:
            experts.eval(); gate.eval()
            with torch.no_grad():
                gate_w_v = gate(Xv)
                r_out_v  = torch.stack([e(Xv) for e in experts], dim=-1)
                correction_v = (gate_w_v * r_out_v).sum(-1)
                pred_v = global_val + correction_v
                val_mse = ((val_y - pred_v.cpu().numpy()) ** 2).mean()
            if val_mse < best_val:
                best_val = val_mse
                best_state = (copy.deepcopy(experts.state_dict()), copy.deepcopy(gate.state_dict()))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
    experts.load_state_dict(best_state[0])
    gate.load_state_dict(best_state[1])
    experts.eval(); gate.eval()
    with torch.no_grad():
        gate_w_v = gate(Xv)
        r_out_v  = torch.stack([e(Xv) for e in experts], dim=-1)
        final_pred = global_val + (gate_w_v * r_out_v).sum(-1)
    best_val_ic = compute_val_ic(final_pred.cpu().numpy(), val_y, val_dates)
    es_tag = f"ES@{epoch}" if epoch < max_epochs else f"FULL({max_epochs})"
    print(f"    [residual+gate] {es_tag}  best_val_mse={best_val:.5f}  best_val_ic={best_val_ic:.5f}")
    return experts, gate, best_val, best_val_ic


def predict_residual_datadriven(f_global, experts, gate, X):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    f_global.eval(); experts.eval(); gate.eval()
    with torch.no_grad():
        global_pred = f_global(X_t)
        gate_w = gate(X_t)
        r_out  = torch.stack([e(X_t) for e in experts], dim=-1)
        correction = (gate_w * r_out).sum(-1)
        pred = (global_pred + correction).cpu().numpy()
        dominant = gate_w.argmax(dim=-1).cpu().numpy()
    return pred, dominant


def run_walk_forward(panel, alpha, seed=0, residual_lr=LR, hidden=16, n_experts=N_EXPERTS):
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

        f_global = train_global(train_X, train_y, val_X, val_y, len(FEATURES), seed, hidden, GLOBAL_LR)
        for p in f_global.parameters():
            p.requires_grad_(False)

        experts, gate, best_val, best_val_ic = train_residual_datadriven(
            f_global, train_X, train_y, val_X, val_y, val_dates,
            len(FEATURES), n_experts, alpha, seed, residual_lr, hidden)

        pred, dominant = predict_residual_datadriven(f_global, experts, gate, test_X)
        val_mses.append(best_val)
        val_ics.append(best_val_ic)

        rec = test_df[["Date", "ticker", "vol_20", TARGET, "fwd_ret_1"]].copy()
        rec["residual_pred"]    = pred
        rec["dominant_expert"]  = dominant
        records.append(rec)

    return pd.concat(records, ignore_index=True), np.mean(val_mses), np.nanmean(val_ics)


def main():
    panel = pd.read_parquet(PROCESSED_DIR / "panel_systemic.parquet")
    tw_tag = f"_tw{TRAIN_WINDOW_DAYS}" if TRAIN_WINDOW_DAYS != 504 else ""
    print(f"gate=datadriven  global_lr={GLOBAL_LR}  residual_lr={RESIDUAL_LR}  alphas={ALPHAS}  lb_weight={LB_WEIGHT}  hidden={HIDDEN}  seed={SEED}  train_window_days={TRAIN_WINDOW_DAYS}")

    DATADRIVEN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    results_by_alpha = {}
    for alpha in ALPHAS:
        print(f"\n=== gate=datadriven lr={RESIDUAL_LR} alpha={alpha} lb_weight={LB_WEIGHT} ===")
        results, mean_val_mse, mean_val_ic = run_walk_forward(
            panel, alpha=alpha, seed=SEED, residual_lr=RESIDUAL_LR, hidden=HIDDEN)
        alpha_tag = "" if alpha == FROZEN_ALPHA else f"_alpha{alpha}"
        fname = f"seed{SEED}{tw_tag}{alpha_tag}.parquet"
        results.to_parquet(DATADRIVEN_RESULTS_DIR / fname, index=False)
        results_by_alpha[alpha] = results
        summary.append({"alpha": alpha, "val_mse": mean_val_mse, "val_ic": mean_val_ic})
        print(f"  mean_val_mse={mean_val_mse:.5f}  mean_val_ic={mean_val_ic:.5f}")

    df = pd.DataFrame(summary)
    best_alpha = df.loc[df["val_ic"].idxmax(), "alpha"]
    val_csv = DATADRIVEN_RESULTS_DIR / f"val_scores_seed{SEED}{tw_tag}.csv"
    df.to_csv(val_csv, index=False)
    print(f"\n>>> best alpha={best_alpha} (by val_ic)  (saved {val_csv.name})")

    # Quick comparison vs plain MLP on test
    results = results_by_alpha[best_alpha]
    target  = results[TARGET]
    moe_mse = ((target - results['residual_pred']) ** 2).mean()
    print(f"\n=== TEST: gate=datadriven alpha={best_alpha} ===")
    print(f"MoE  test_mse={moe_mse:.5f}")
    print(f"dominant-expert usage: {results['dominant_expert'].value_counts(normalize=True).sort_index().to_dict()}")


if __name__ == "__main__":
    main()
