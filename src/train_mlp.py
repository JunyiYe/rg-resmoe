"""MLP baselines: a single two-hidden-layer block on the 16 stock features.

MLP-S (HIDDEN=16, the default) matches one RG-ResMoE building block;
MLP-L (HIDDEN=44) matches RG-ResMoE's overall parameter budget.
INPUT_STATES=1 appends the two regime state variables z to the input --
the "+z" input-pathway variants of the paper's pathway table (their
learning rate was re-selected from scratch; default switches to 0.003).

Usage:
  SEED=0 HIDDEN=16 python src/train_mlp.py                  # MLP-S
  SEED=0 HIDDEN=44 python src/train_mlp.py                  # MLP-L
  SEED=0 HIDDEN=16 INPUT_STATES=1 python src/train_mlp.py   # MLP-S(+z)
Output: results/predictions/mlp_{s,l}[_state]/seed{SEED}.parquet
Columns: Date, ticker, vol_20, target_fwd_vol_5, fwd_ret_1, plain_pred
"""
import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from walk_forward import (
    CHECK_EVERY,
    DEVICE,
    FEATURES,
    LR,
    MAX_EPOCHS,
    PATIENCE_CHECKS,
    PROCESSED_DIR,
    RESULTS_DIR,
    STEP_DAYS,
    TARGET,
    TRAIN_WINDOW_DAYS,
    _split_train_val,
    add_lag_features,
)

SEED   = int(os.environ.get("SEED", "0"))
HIDDEN = int(os.environ.get("HIDDEN", "16"))

# INPUT_STATES=1 appends market_vol_20/idio_vol_20 to the MLP input -- the
# same two regime state variables z that train_rg_resmoe.py feeds its gate.
# This is the "+z" input pathway: identical information, different entry
# point. Its learning rate was re-selected from scratch (paper protocol),
# hence the different default below.
EXTRA_FEATURES = os.environ.get("INPUT_STATES", "0") == "1"
LR_USE = float(os.environ.get("LR", "0.003" if EXTRA_FEATURES else "0.03"))
GATE_AUX_FEATURES = ["market_vol_20", "idio_vol_20"]
ACTIVE_FEATURES = FEATURES + GATE_AUX_FEATURES if EXTRA_FEATURES else FEATURES



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


class PlainMLP(nn.Module):
    def __init__(self, in_dim=len(ACTIVE_FEATURES), hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_es(train_X, train_y, val_X, val_y, val_dates, seed=SEED, lr=LR_USE):
    torch.manual_seed(seed)
    model = PlainMLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(train_X, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(train_y, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(val_X,   dtype=torch.float32, device=DEVICE)
    yv = torch.tensor(val_y,   dtype=torch.float32, device=DEVICE)
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
    model.eval()
    with torch.no_grad():
        final_pred = model(Xv).cpu().numpy()
    best_val_ic = compute_val_ic(final_pred, val_y, val_dates)
    es_tag = f"ES@{epoch}" if epoch < max_epochs else f"FULL({max_epochs})"
    print(f"    {es_tag}  best_val_mse={best_val:.5f}  best_val_ic={best_val_ic:.5f}")
    return model, best_val, best_val_ic


def run_walk_forward(panel, seed=SEED, lr=LR_USE):
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
        scaler   = StandardScaler().fit(train_df[ACTIVE_FEATURES])
        train_X  = scaler.transform(train_df[ACTIVE_FEATURES])
        val_X    = scaler.transform(val_df[ACTIVE_FEATURES])
        test_X   = scaler.transform(test_df[ACTIVE_FEATURES])
        train_y  = train_df[TARGET].values
        val_y    = val_df[TARGET].values
        val_dates = val_df["Date"].values
        model, best_val, best_val_ic = train_es(train_X, train_y, val_X, val_y, val_dates, seed, lr)
        val_mses.append(best_val)
        val_ics.append(best_val_ic)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(test_X, dtype=torch.float32, device=DEVICE)).cpu().numpy()
        rec = test_df[["Date", "ticker", "vol_20", TARGET, "fwd_ret_1"]].copy()
        rec["plain_pred"] = pred
        records.append(rec)
    return pd.concat(records, ignore_index=True), float(np.mean(val_mses)), float(np.nanmean(val_ics))


def main():
    n_params = len(ACTIVE_FEATURES) * HIDDEN + HIDDEN ** 2 + 3 * HIDDEN + 1
    print(f"PlainMLP  in_dim={len(ACTIVE_FEATURES)}  hidden={HIDDEN}  params={n_params}  seed={SEED}  lr={LR_USE}  "
          f"input_states={EXTRA_FEATURES}")

    panel = pd.read_parquet(PROCESSED_DIR / os.environ.get("PANEL_FILE", "panel_systemic.parquet"))
    panel = add_lag_features(panel)

    results, mean_val_mse, mean_val_ic = run_walk_forward(panel, seed=SEED, lr=LR_USE)
    tw_tag = f"_tw{TRAIN_WINDOW_DAYS}" if TRAIN_WINDOW_DAYS != 504 else ""
    size = {16: "mlp_s", 44: "mlp_l"}.get(HIDDEN, f"mlp_h{HIDDEN}")
    model_dir = RESULTS_DIR / (size + ("_state" if EXTRA_FEATURES else ""))
    model_dir.mkdir(parents=True, exist_ok=True)
    out = model_dir / f"seed{SEED}{tw_tag}.parquet"
    results.to_parquet(out, index=False)

    test_mse = ((results[TARGET] - results["plain_pred"]) ** 2).mean()
    print(f"rows={len(results)}  mean_val_mse={mean_val_mse:.5f}  mean_val_ic={mean_val_ic:.5f}  test_mse={test_mse:.5f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
