#!/usr/bin/env bash
# Japan TSE Prime cross-market replication (paper Section 8, Table 9).
# Trains the four models in that table: MLP-L, MLP-L(+z), Standard MoE,
# RG-ResMoE, all with the same frozen hyperparameters as the U.S. run.
set -euo pipefail
cd "$(dirname "$0")"

export MARKET=jp
export PANEL_FILE=panel_systemic_jp.parquet
N_SEEDS="${N_SEEDS:-30}"
P=results/predictions/jp
mkdir -p results/summary

# 1. Panel: raw OHLCV CSVs -> feature panel -> + regime state variables z
[ -f data/processed/panel_jp.parquet ]          || RAW_DIR_NAME=raw_jp PANEL_OUT=panel_jp.parquet python src/build_panel.py
[ -f data/processed/panel_systemic_jp.parquet ] || PANEL_IN=panel_jp.parquet PANEL_OUT=panel_systemic_jp.parquet python src/add_regime_states.py

run() {
  local out="$1"; shift
  if [ -e "$out" ]; then echo "skip  $out"; return; fi
  env "$@"
}

for s in $(seq 0 $((N_SEEDS - 1))); do
  run "$P/mlp_l/seed${s}.parquet"       SEED=$s HIDDEN=44 python src/train_mlp.py
  run "$P/mlp_l_state/seed${s}.parquet" SEED=$s HIDDEN=44 INPUT_STATES=1 python src/train_mlp.py
  run "$P/standard_moe/seed${s}.parquet" SEED=$s python src/train_standard_moe.py
  run "$P/rg_resmoe/seed${s}.parquet"    SEED=$s python src/train_rg_resmoe.py
done

# 2. Aggregate
N_SEEDS=$N_SEEDS python src/summarize_results_jp.py
N_SEEDS=$N_SEEDS python src/dm_tests_jp.py
