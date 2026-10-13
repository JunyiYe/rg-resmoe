#!/usr/bin/env bash
# Full pipeline for the main experiments (headline + pathway tables),
# sequential. On a cluster, parallelize the (model, seed) loop — every
# training run is one process covering all 30 walk-forward windows, and the
# whole pipeline is idempotent (finished outputs are skipped).
set -euo pipefail
cd "$(dirname "$0")"

N_SEEDS="${N_SEEDS:-30}"
P=results/predictions
mkdir -p results/summary

# 1. Panels: raw OHLCV CSVs -> feature panel -> + regime state variables z
[ -f data/processed/panel.parquet ]          || python src/build_panel.py
[ -f data/processed/panel_systemic.parquet ] || python src/add_regime_states.py

# 2. Classical baselines (CPU, deterministic): Persistence, HAR, GARCH, Ridge(+z)
[ -f "$P/classical/garch.parquet" ] || python src/train_classical_baselines.py
[ -f "$P/classical/ridge.parquet" ] || python src/train_ridge.py

# 3. Neural models, 30 seeds each (GPU). run <output> <ENV=val ...> <cmd ...>
run() {
  local out="$1"; shift
  if [ -e "$out" ]; then echo "skip  $out"; return; fi
  env "$@"
}

for s in $(seq 0 $((N_SEEDS - 1))); do
  # RG-ResMoE (main model) and its no-state-variables control (-z)
  run "$P/rg_resmoe/seed${s}.parquet"           SEED=$s python src/train_rg_resmoe.py
  run "$P/rg_resmoe_no_states/seed${s}.parquet" SEED=$s python src/train_rg_resmoe_no_states.py

  # Standard (non-residual) MoE
  run "$P/standard_moe/seed${s}.parquet"        SEED=$s python src/train_standard_moe.py

  # MLP-S / MLP-L, and the (+z) input-pathway variants
  run "$P/mlp_s/seed${s}.parquet"       SEED=$s HIDDEN=16 python src/train_mlp.py
  run "$P/mlp_l/seed${s}.parquet"       SEED=$s HIDDEN=44 python src/train_mlp.py
  run "$P/mlp_s_state/seed${s}.parquet" SEED=$s HIDDEN=16 INPUT_STATES=1 python src/train_mlp.py
  run "$P/mlp_l_state/seed${s}.parquet" SEED=$s HIDDEN=44 INPUT_STATES=1 python src/train_mlp.py

  # Optional (RUN_ABLATIONS=1): routing / design variants behind the ablation tables
  if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
    run "$P/rg_resmoe_hard_routing/seed${s}.parquet" \
      SEED=$s HARD_ROUTING=1 python src/train_rg_resmoe.py
    run "$P/rg_resmoe_k2_matched/seed${s}.parquet" \
      SEED=$s N_EXPERTS=2 EXPERT_HIDDEN=25 python src/train_rg_resmoe.py
    run "$P/rg_resmoe_k6_matched/seed${s}.parquet" \
      SEED=$s N_EXPERTS=6 EXPERT_HIDDEN=12 python src/train_rg_resmoe.py
    run "$P/rg_resmoe_random_init/seed${s}.parquet" \
      SEED=$s ZERO_INIT=0 python src/train_rg_resmoe.py
    run "$P/rule_gate_vol_quantiles/seed${s}.parquet" \
      SEED=$s python src/train_rule_gate_moe.py vol
    run "$P/rule_gate_market_idio/seed${s}.parquet" \
      SEED=$s python src/train_rule_gate_moe.py systemic_idio
    run "$P/rule_gate_sectors/seed${s}.parquet" \
      SEED=$s python src/train_rule_gate_moe.py sector
  fi
done

# 4. Aggregate: headline metrics, significance, collapse counts
N_SEEDS=$N_SEEDS python src/summarize_results.py
N_SEEDS=$N_SEEDS python src/dm_tests.py
N_SEEDS=$N_SEEDS python src/collapse_rates.py
if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
  N_SEEDS=$N_SEEDS python src/dm_tests_variants.py
fi
