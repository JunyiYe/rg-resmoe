# -*- coding: utf-8 -*-
"""RG-ResMoE, the paper's main model.

A frozen base MLP predicts the volatility level; K zero-initialized
correction experts learn residual corrections that a learned soft gate
combines. The two regime state variables z (market_vol_20, idio_vol_20)
are fed to the gate ONLY -- the base network and the experts see just the
16 stock-level features, so the parameter-matched comparison against
MLP-S / MLP-L is unaffected.

Defaults reproduce the paper's frozen configuration; env vars select the
ablation variants (paper notation in parentheses):
  SEED           training seed (30 seeds in the paper)
  EXPERT_LR      expert/gate learning rate           (default 0.01)
  BASE_LR        stage-one base-network learning rate (default 0.03)
  ALPHA          correction-shrinkage strength alpha  (default 1.0)
  LB_WEIGHT      load-balance penalty lambda_LB       (default 0.5)
  EXPERT_HIDDEN  expert width H                       (default 16)
  BASE_HIDDEN    base-network width                   (default 16)
  N_EXPERTS      expert count K                       (default 4)
  HARD_ROUTING=1 straight-through top-1 routing ablation
  ZERO_INIT=0    random-initialization ablation

Usage: SEED=0 python src/train_rg_resmoe.py
Output: results/predictions/<variant>/seed{SEED}.parquet
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

RESIDUAL_LR = float(os.environ.get("EXPERT_LR", "0.01"))
FROZEN_ALPHA = 1.0
ALPHAS      = [float(x) for x in os.environ.get("ALPHA", str(FROZEN_ALPHA)).split(",")]
SEED        = int(os.environ.get("SEED", "0"))
HIDDEN      = int(os.environ.get("EXPERT_HIDDEN", "16"))
# BASE_HIDDEN decouples the base network's width from the experts' width.
# Needed for the parameter-matched K=2 ablation (2 experts x H=25 = 2202
# params vs 4 x H=16 = 2244): EXPERT_HIDDEN must grow to 25 while the base
# stays H=16 to remain the MLP-S-matched warm start.
GLOBAL_HIDDEN = int(os.environ.get("BASE_HIDDEN", "16"))
GLOBAL_LR   = float(os.environ.get("BASE_LR", "0.03"))
LB_WEIGHT   = float(os.environ.get("LB_WEIGHT", "0.5"))


# HARD_ROUTING=1: straight-through top-1 routing -- forward pass uses a
# one-hot vector at the softmax argmax (so only ONE expert's output reaches
# the prediction, like the hard vol/systemic_idio/sector gates), backward
# pass still flows the soft-softmax gradient (standard straight-through
# estimator), so the gate remains trainable. Isolates whether soft blending
# itself (not just "some data-driven signal") is doing the work: temperature
# sweeps couldn't answer this because the network partially compensates by
# shrinking logits, never reaching genuinely hard (>0.9 max-weight) routing.
HARD_ROUTING = os.environ.get("HARD_ROUTING", "0") == "1"

# N_EXPERTS override for the K=2 ablation: gate-weight diagnostics showed
# only 2 of 4 experts correlate with market_vol_20/idio_vol_20 (the other 2
# are ~uncorrelated), suggesting the learned gate found one high/low regime
# axis, not four genuinely distinct regimes.
N_EXPERTS_ABLATION = int(os.environ.get("N_EXPERTS", str(N_EXPERTS)))

# ZERO_INIT=0: skip zero-initializing the experts' output layers, using the
# framework default (Kaiming-uniform) init instead. Isolates whether
# zero-init specifically (vs. the frozen base or output shrinkage) is what
# stabilizes training -- the "necessary scaffold" finding so far only tests
# removing all three at once (the pure-MoE ablation).
ZERO_INIT = os.environ.get("ZERO_INIT", "1") == "1"

# Each ablation variant writes to its own model folder.
if HARD_ROUTING:
    _variant = "rg_resmoe_hard_routing"
elif N_EXPERTS_ABLATION != N_EXPERTS:
    _variant = (f"rg_resmoe_k{N_EXPERTS_ABLATION}_matched" if HIDDEN != 16
                else f"rg_resmoe_k{N_EXPERTS_ABLATION}")
elif not ZERO_INIT:
    _variant = "rg_resmoe_random_init"
else:
    _variant = "rg_resmoe"
DATADRIVEN_RESULTS_DIR = RESULTS_DIR / _variant


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
    """Soft softmax weights, or (if HARD_ROUTING) a straight-through top-1
    one-hot: forward pass sees only the argmax expert's output, backward
    pass still flows the soft-softmax gradient so the gate stays trainable."""
    logits = gate.net(x)
    soft = torch.softmax(logits, dim=-1)
    if not HARD_ROUTING:
        return soft
    idx = soft.argmax(dim=-1)
    hard = torch.zeros_like(soft).scatter_(-1, idx.unsqueeze(-1), 1.0)
    return hard + (soft - soft.detach())


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


def train_residual_datadriven(f_global, train_X, train_y, train_gate_X, val_X, val_y, val_gate_X, val_dates,
                               in_dim, n_experts, alpha, seed=0, residual_lr=LR, hidden=16):
    torch.manual_seed(seed)
    experts = nn.ModuleList([ExpertMLP(in_dim, hidden=hidden) for _ in range(n_experts)]).to(DEVICE)
    if ZERO_INIT:
        for e in experts:
            nn.init.zeros_(e.net[-1].weight)
            nn.init.zeros_(e.net[-1].bias)
    gate = GatingNet(train_gate_X.shape[1], n_experts).to(DEVICE)
    opt = torch.optim.Adam(list(experts.parameters()) + list(gate.parameters()), lr=residual_lr)

    Xt  = torch.tensor(train_X,      dtype=torch.float32, device=DEVICE)
    yt  = torch.tensor(train_y,      dtype=torch.float32, device=DEVICE)
    Gt  = torch.tensor(train_gate_X, dtype=torch.float32, device=DEVICE)
    Xv  = torch.tensor(val_X,        dtype=torch.float32, device=DEVICE)
    Gv  = torch.tensor(val_gate_X,   dtype=torch.float32, device=DEVICE)

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
        gxt = Gt if idx is None else Gt[idx]
        yt_ = yt if idx is None else yt[idx]
        gate_w = gate_weights(gate, gxt)
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
                gate_w_v = gate_weights(gate, Gv)
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
        gate_w_v = gate_weights(gate, Gv)
        r_out_v  = torch.stack([e(Xv) for e in experts], dim=-1)
        final_pred = global_val + (gate_w_v * r_out_v).sum(-1)
    best_val_ic = compute_val_ic(final_pred.cpu().numpy(), val_y, val_dates)
    es_tag = f"ES@{epoch}" if epoch < max_epochs else f"FULL({max_epochs})"
    print(f"    [residual+gate] {es_tag}  best_val_mse={best_val:.5f}  best_val_ic={best_val_ic:.5f}")
    return experts, gate, best_val, best_val_ic


def predict_residual_datadriven(f_global, experts, gate, X, gate_X):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    G_t = torch.tensor(gate_X, dtype=torch.float32, device=DEVICE)
    f_global.eval(); experts.eval(); gate.eval()
    with torch.no_grad():
        global_pred = f_global(X_t)
        gate_w = gate_weights(gate, G_t)
        r_out  = torch.stack([e(X_t) for e in experts], dim=-1)
        correction = (gate_w * r_out).sum(-1)
        pred = (global_pred + correction).cpu().numpy()
        dominant = gate_w.argmax(dim=-1).cpu().numpy()
        gate_w_np = gate_w.cpu().numpy()
    return pred, dominant, gate_w_np


def run_walk_forward(panel, alpha, seed=0, residual_lr=LR, hidden=16, n_experts=N_EXPERTS_ABLATION):
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

        f_global = train_global(train_X, train_y, val_X, val_y, len(FEATURES), seed, GLOBAL_HIDDEN, GLOBAL_LR)
        for p in f_global.parameters():
            p.requires_grad_(False)

        experts, gate, best_val, best_val_ic = train_residual_datadriven(
            f_global, train_X, train_y, train_gate_X, val_X, val_y, val_gate_X, val_dates,
            len(FEATURES), n_experts, alpha, seed, residual_lr, hidden)

        pred, dominant, gate_w_np = predict_residual_datadriven(f_global, experts, gate, test_X, test_gate_X)
        val_mses.append(best_val)
        val_ics.append(best_val_ic)

        rec = test_df[["Date", "ticker", "vol_20", TARGET, "fwd_ret_1"] + GATE_AUX_FEATURES].copy()
        rec["residual_pred"]    = pred
        rec["dominant_expert"]  = dominant
        for k in range(gate_w_np.shape[1]):
            rec[f"gate_w{k}"] = gate_w_np[:, k]
        records.append(rec)

    return pd.concat(records, ignore_index=True), np.mean(val_mses), np.nanmean(val_ics)


def main():
    panel = pd.read_parquet(PROCESSED_DIR / os.environ.get("PANEL_FILE", "panel_systemic.parquet"))
    # Non-default overrides are tagged in the filename; the ablation variant
    # itself is already encoded in the output folder name.
    tw_tag = f"_tw{TRAIN_WINDOW_DAYS}" if TRAIN_WINDOW_DAYS != 504 else ""
    print(f"model=rg_resmoe  global_lr={GLOBAL_LR}  residual_lr={RESIDUAL_LR}  alphas={ALPHAS}  "
          f"lb_weight={LB_WEIGHT}  hidden={HIDDEN}  global_hidden={GLOBAL_HIDDEN}  seed={SEED}  train_window_days={TRAIN_WINDOW_DAYS}  "
          f"gate_aux={GATE_AUX_FEATURES}  hard_routing={HARD_ROUTING}  "
          f"n_experts={N_EXPERTS_ABLATION}  zero_init={ZERO_INIT}")

    DATADRIVEN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    results_by_alpha = {}
    for alpha in ALPHAS:
        print(f"\n=== model=rg_resmoe lr={RESIDUAL_LR} alpha={alpha} lb_weight={LB_WEIGHT} ===")
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

    results = results_by_alpha[best_alpha]
    target  = results[TARGET]
    moe_mse = ((target - results['residual_pred']) ** 2).mean()
    print(f"\n=== TEST: model=rg_resmoe alpha={best_alpha} ===")
    print(f"MoE  test_mse={moe_mse:.5f}")
    print(f"dominant-expert usage: {results['dominant_expert'].value_counts(normalize=True).sort_index().to_dict()}")


if __name__ == "__main__":
    main()
