# -*- coding: utf-8 -*-
"""Standard (non-residual) MoE -- the residual-scaffold ablation.

pred(x) = sum_k gate_w_k(x) * expert_k(x) -- jointly trained from standard
init, NO frozen f_global backbone, NO correction-magnitude shrinkage
(alpha). Otherwise identical to train_rg_resmoe.py: same soft learned gate fed 16 shared FEATURES + 2
gate-only aux vars (market_vol_20, idio_vol_20), same load-balance penalty.

Tests whether the residual "recipe" (warm-started frozen global model +
zero-initialized correction experts + output-space shrinkage toward that
global model) is necessary, or whether a naive jointly-trained MoE
performs just as well on its own.

Removing f_global frees up its parameter budget, so each expert here uses a
larger hidden size than the residual recipe's correction experts (H=16) to
keep the total parameter count comparable -- if the pure MoE loses despite
equal or greater capacity, that is not a capacity artifact.

Usage: SEED=0 python src/train_standard_moe.py
Output: results/predictions/standard_moe/seed{SEED}.parquet
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

GATE_AUX_FEATURES = ["market_vol_20", "idio_vol_20"]

RLRS        = [float(x) for x in os.environ.get("LR", "0.03").split(",")]
FROZEN_LR   = 0.03
SEED        = int(os.environ.get("SEED", "0"))
HIDDEN      = int(os.environ.get("EXPERT_HIDDEN", "18"))
LB_WEIGHT   = float(os.environ.get("LB_WEIGHT", "0.5"))
N_EXPERTS_ABLATION = int(os.environ.get("N_EXPERTS", str(N_EXPERTS)))


DATADRIVEN_RESULTS_DIR = RESULTS_DIR / "standard_moe"


def compute_val_ic(pred, target, dates, min_n=20):
    df = pd.DataFrame({"pred": pred, "target": target, "Date": dates})
    ics = []
    for _, g in df.groupby("Date"):
        if len(g) < min_n:
            continue
        ic = spearmanr(g["pred"], g["target"])[0]
        if np.isfinite(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan


def gate_weights(gate, x):
    return torch.softmax(gate.net(x), dim=-1)


def train_standard_moe(train_X, train_y, train_gate_X, val_X, val_y, val_gate_X, val_dates,
                    in_dim, n_experts, seed=0, lr=LR, hidden=HIDDEN):
    torch.manual_seed(seed)
    # Standard init throughout -- unlike the residual recipe's correction
    # experts (zero-initialized output layer so the model starts exactly at
    # f_global's prediction), these experts must predict the full target
    # from scratch and a zero start would just mean "predict 0" initially.
    experts = nn.ModuleList([ExpertMLP(in_dim, hidden) for _ in range(n_experts)]).to(DEVICE)
    gate    = GatingNet(train_gate_X.shape[1], n_experts).to(DEVICE)
    opt = torch.optim.Adam(list(experts.parameters()) + list(gate.parameters()), lr=lr)

    Xt = torch.tensor(train_X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(train_y, dtype=torch.float32, device=DEVICE)
    Gt = torch.tensor(train_gate_X, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(val_X, dtype=torch.float32, device=DEVICE)
    yv_np = val_y
    Gv = torch.tensor(val_gate_X, dtype=torch.float32, device=DEVICE)

    n = len(train_X)
    max_epochs = MAX_EPOCHS
    check_every = CHECK_EVERY
    patience = PATIENCE_CHECKS

    def step(idx=None):
        xt  = Xt if idx is None else Xt[idx]
        gxt = Gt if idx is None else Gt[idx]
        yt_ = yt if idx is None else yt[idx]
        gate_w = gate_weights(gate, gxt)
        r_out  = torch.stack([e(xt) for e in experts], dim=-1)
        pred = (gate_w * r_out).sum(-1)
        mse  = nn.functional.mse_loss(pred, yt_)
        usage = gate_w.mean(0)
        lb_loss = ((usage - 1.0 / n_experts) ** 2).sum()
        # No alpha/correction-magnitude term -- there is no "correction",
        # this pred IS the prediction, shrinking it toward zero would be
        # actively wrong.
        loss = mse + LB_WEIGHT * lb_loss
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
                gate_w_v = gate_weights(gate, Gv)
                r_out_v  = torch.stack([e(Xv) for e in experts], dim=-1)
                pred_v = (gate_w_v * r_out_v).sum(-1)
                val_mse = ((yv_np - pred_v.cpu().numpy()) ** 2).mean()
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
        gate_w_v = gate_weights(gate, Gv)
        r_out_v  = torch.stack([e(Xv) for e in experts], dim=-1)
        final_pred = (gate_w_v * r_out_v).sum(-1)
    best_val_ic = compute_val_ic(final_pred.cpu().numpy(), yv_np, val_dates)
    es_tag = f"ES@{epoch}" if epoch < max_epochs else f"FULL({max_epochs})"
    print(f"    [standard_moe] {es_tag}  best_val_mse={best_val:.5f}  best_val_ic={best_val_ic:.5f}")
    return experts, gate, best_val, best_val_ic


def predict_standard_moe(experts, gate, X, gate_X):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    G_t = torch.tensor(gate_X, dtype=torch.float32, device=DEVICE)
    experts.eval(); gate.eval()
    with torch.no_grad():
        gate_w = gate_weights(gate, G_t)
        r_out  = torch.stack([e(X_t) for e in experts], dim=-1)
        pred = (gate_w * r_out).sum(-1).cpu().numpy()
        dominant = gate_w.argmax(dim=-1).cpu().numpy()
        gate_w_np = gate_w.cpu().numpy()
    return pred, dominant, gate_w_np


def run_walk_forward(panel, seed=0, lr=LR, hidden=HIDDEN, n_experts=N_EXPERTS_ABLATION):
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

        gate_scaler = StandardScaler().fit(train_df[GATE_AUX_FEATURES])
        train_aux = gate_scaler.transform(train_df[GATE_AUX_FEATURES])
        val_aux   = gate_scaler.transform(val_df[GATE_AUX_FEATURES])
        test_aux  = gate_scaler.transform(test_df[GATE_AUX_FEATURES])
        train_gate_X = np.concatenate([train_X, train_aux], axis=1)
        val_gate_X   = np.concatenate([val_X,   val_aux],   axis=1)
        test_gate_X  = np.concatenate([test_X,  test_aux],  axis=1)

        experts, gate, best_val, best_val_ic = train_standard_moe(
            train_X, train_y, train_gate_X, val_X, val_y, val_gate_X, val_dates,
            len(FEATURES), n_experts, seed, lr, hidden)

        pred, dominant, gate_w_np = predict_standard_moe(experts, gate, test_X, test_gate_X)
        val_mses.append(best_val)
        val_ics.append(best_val_ic)

        rec = test_df[["Date", "ticker", "vol_20", TARGET, "fwd_ret_1"] + GATE_AUX_FEATURES].copy()
        # Column named "residual_pred" (not "standard_moe_pred") on purpose --
        # this IS the final prediction here, and keeping the column name
        # matches the MODELS dict pattern in dm_tests.py / summarize_results.py
        # so this file drops straight into the existing aggregation tooling.
        rec["residual_pred"]   = pred
        rec["dominant_expert"] = dominant
        for k in range(gate_w_np.shape[1]):
            rec[f"gate_w{k}"] = gate_w_np[:, k]
        records.append(rec)

    return pd.concat(records, ignore_index=True), float(np.mean(val_mses)), float(np.nanmean(val_ics))


def main():

    panel = pd.read_parquet(PROCESSED_DIR / os.environ.get("PANEL_FILE", "panel_systemic.parquet"))
    tw_tag = f"_tw{TRAIN_WINDOW_DAYS}" if TRAIN_WINDOW_DAYS != 504 else ""
    tw_tag += f"_k{N_EXPERTS_ABLATION}" if N_EXPERTS_ABLATION != N_EXPERTS else ""
    print(f"standard_moe  lrs={RLRS}  lb_weight={LB_WEIGHT}  hidden={HIDDEN}  n_experts={N_EXPERTS_ABLATION}  "
          f"seed={SEED}  train_window_days={TRAIN_WINDOW_DAYS}  "
          f"gate_aux={GATE_AUX_FEATURES}")

    DATADRIVEN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for lr in RLRS:
        print(f"\n=== standard_moe lr={lr} lb_weight={LB_WEIGHT} ===")
        results, mean_val_mse, mean_val_ic = run_walk_forward(panel, seed=SEED, lr=lr, hidden=HIDDEN)
        lr_tag = "" if lr == FROZEN_LR else f"_lr{lr}"
        fname = f"seed{SEED}{tw_tag}{lr_tag}.parquet"
        results.to_parquet(DATADRIVEN_RESULTS_DIR / fname, index=False)
        summary.append({"lr": lr, "val_mse": mean_val_mse, "val_ic": mean_val_ic})
        print(f"  mean_val_mse={mean_val_mse:.5f}  mean_val_ic={mean_val_ic:.5f}")

    df = pd.DataFrame(summary)
    best_lr = df.loc[df["val_ic"].idxmax(), "lr"]
    val_csv = DATADRIVEN_RESULTS_DIR / f"val_scores_seed{SEED}{tw_tag}.csv"
    df.to_csv(val_csv, index=False)
    print(f"\n>>> best lr={best_lr} (by val_ic)  (saved {val_csv.name})")


if __name__ == "__main__":
    main()
