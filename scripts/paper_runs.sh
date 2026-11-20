#!/usr/bin/env bash
# Every run of the paper "Lit Up but Not Needed", in pipeline order. The title of a block names the tables and figures
# it feeds. Run from the repository root, top to bottom, or hand the commands to a job scheduler: each command needs
# one GPU unless it is marked CPU.
#
#   MOEDEP_DATA    datasets (default data/, read by the Python code)
#   MOEDEP_MODELS  model weights (default models/)
#   MOEDEP_RUNS    outputs (default runs/)
#   PYTHON         interpreter of the main environment (requirements.txt)
#   PYTHON_KIMI    interpreter of the Kimi-VL environment (requirements-kimi.txt)
#
# Lesion and generation runs append to results.jsonl and skip the (item, condition) pairs that are already there, so
# several commands share one output folder and an interrupted command can be started again. A change is always taken
# against the "none" condition of the same folder.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1      # as in the paper's runs: every model is read from a local folder
PY="${PYTHON:-python}"
PY_KIMI="${PYTHON_KIMI:-python}"
MODELS="${MOEDEP_MODELS:-models}"
RUNS="${MOEDEP_RUNS:-runs}"
export MOEDEP_RUNS="$RUNS"
QWEN="$MODELS/Qwen3-VL-30B-A3B-Instruct"
KIMI="$MODELS/Kimi-VL-A3B-Instruct"
GEMMA="$MODELS/gemma-4-26B-A4B-it"
Q="$RUNS/qwen3vl"; A="$Q/analysis"
K="$RUNS/kimivl"; KA="$K/analysis"
KNB="$RUNS/kimivl_nobias"
G="$RUNS/gemma4"; GA="$G/analysis"

# ---- 1. Qwen3-VL: routing pass and expert sets (Fig. 2, Table 2, Sec. 4, and the masks of blocks 2 to 6)
"$PY" -m moedep.run_routing --model "$QWEN" --out_dir "$Q" --sets mm,text,ascii,shapes \
  --n_mm 120 --n_alpaca 400 --n_qonly 300 --n_concepts 300 --n_var 6
"$PY" -m moedep.analyze_routing --run_dir "$Q"                      # CPU
"$PY" -m moedep.expert_sets --run_dir "$Q" --affinity_count 462     # CPU

# ---- 2. Qwen3-VL: lesions at all positions (Table 1 upper block, Table S1, Fig. 3A, random draws 0 and 1)
"$PY" -m moedep.run_ablation --model "$QWEN" --out_dir "$Q/ablation2" \
  --mask "V10=$A/substrate_ratio10.json" --mask "V05b=$A/substrate_bshare05.json" --mask "TOPIMG16=$A/substrate_topimg16.json" \
  --mask "TOPIMG32=$A/substrate_topimg32.json" --mask "TOPTXT16=$A/substrate_toptxt16.json" --mask "T10=$A/substrate_text10.json" \
  --cond none --cond V10:reroute --cond V10:zero --cond V05b:reroute --cond TOPIMG16:reroute --cond TOPIMG16:zero \
  --cond TOPIMG32:reroute --cond TOPTXT16:reroute --cond T10:reroute --cond random:V10:0 --cond random:V10:1 \
  --sets mm,ascii,shapes --n_mm 120 --n_concepts 300 --n_var 6
"$PY" -m moedep.run_ablation --model "$QWEN" --out_dir "$Q/ablation2" \
  --mask "V10=$A/substrate_ratio10.json" --mask "TOPIMG16=$A/substrate_topimg16.json" --mask "TOPIMG16_noV10=$A/substrate_topimg16_noV10.json" \
  --mask "TOPIMG16_V10part=$A/substrate_topimg16_V10part.json" --mask "SHARED=$A/substrate_shared.json" --mask "TOPTXT16=$A/substrate_toptxt16.json" \
  --cond none --cond V10:reroute --cond TOPIMG16:reroute --cond TOPIMG16_noV10:reroute --cond TOPIMG16_V10part:reroute \
  --cond SHARED:reroute --cond SHARED:zero --cond TOPTXT16:reroute \
  --sets mm,shapes --n_mm 120 --n_var 6
"$PY" -m moedep.run_ablation --model "$QWEN" --out_dir "$Q/ablation_hard" \
  --mask "V10=$A/substrate_ratio10.json" --mask "TOPIMG16_noV10=$A/substrate_topimg16_noV10.json" --mask "TOPIMG16_V10part=$A/substrate_topimg16_V10part.json" \
  --cond none --cond V10:reroute --cond TOPIMG16_V10part:reroute --cond TOPIMG16_noV10:reroute --sets hard --n_mm 100
# generation metrics: caption NLL and greedy TextVQA (columns TVQA and NLL of Table 1)
"$PY" -m moedep.run_generation --model "$QWEN" --out_dir "$Q/generation" \
  --mask "V10=$A/substrate_ratio10.json" --mask "TOPIMG16=$A/substrate_topimg16.json" --mask "TOPIMG16_noV10=$A/substrate_topimg16_noV10.json" \
  --mask "TOPIMG16_V10part=$A/substrate_topimg16_V10part.json" --mask "SHARED=$A/substrate_shared.json" --mask "TOPTXT16=$A/substrate_toptxt16.json" \
  --cond none --cond V10:reroute --cond TOPIMG16_V10part:reroute --cond TOPIMG16_noV10:reroute --cond TOPIMG16:reroute --cond SHARED:reroute --cond TOPTXT16:reroute

# ---- 3. Qwen3-VL: scoped lesions (Table 1 lower block, Table 2, Table S5, Fig. 3, Fig. S1, random draws 2 to 4)
"$PY" -m moedep.run_ablation --model "$QWEN" --out_dir "$Q/ablation_scope" \
  --mask "V10=$A/substrate_ratio10.json" --mask "TOPIMG16_noV10=$A/substrate_topimg16_noV10.json" --mask "TOPIMG16_V10part=$A/substrate_topimg16_V10part.json" --mask "SHARED=$A/substrate_shared.json" \
  --cond none --cond TOPIMG16_noV10:reroute --cond TOPIMG16_noV10:reroute:image --cond TOPIMG16_noV10:reroute:text \
  --cond TOPIMG16_V10part:reroute:image --cond TOPIMG16_V10part:reroute:text --cond V10:reroute:image --cond V10:reroute:text \
  --cond SHARED:reroute:image --cond SHARED:reroute:text --cond random:V10:2 --cond random:V10:3 --cond random:V10:4 \
  --sets mm,hard,arc --n_mm 120
"$PY" -m moedep.run_ablation --model "$QWEN" --out_dir "$Q/ablation_scope" \
  --mask "TOPIMG16_noV10=$A/substrate_topimg16_noV10.json" --mask "SHARED=$A/substrate_shared.json" --mask "ALL=$A/substrate_all.json" \
  --mask "TOPIMG16=$A/substrate_topimg16.json" --mask "TOPIMG32=$A/substrate_topimg32.json" \
  --cond TOPIMG16_noV10:zero:image --cond SHARED:zero:image --cond ALL:zero:image --cond ALL:zero:text \
  --cond TOPIMG16:reroute:image --cond TOPIMG32:reroute:image --cond TOPIMG16:reroute:text \
  --sets mm,hard,arc --n_mm 120
"$PY" -m moedep.run_generation --model "$QWEN" --out_dir "$Q/generation_scope" \
  --mask "V10=$A/substrate_ratio10.json" --mask "TOPIMG16_noV10=$A/substrate_topimg16_noV10.json" \
  --cond none --cond TOPIMG16_noV10:reroute --cond TOPIMG16_noV10:reroute:image --cond TOPIMG16_noV10:reroute:text --cond V10:reroute:image --cond V10:reroute:text
"$PY" -m moedep.run_generation --model "$QWEN" --out_dir "$Q/generation_scope" \
  --mask "TOPIMG16_noV10=$A/substrate_topimg16_noV10.json" --mask "ALL=$A/substrate_all.json" --mask "TOPIMG32=$A/substrate_topimg32.json" \
  --cond TOPIMG16_noV10:zero:image --cond ALL:zero:image --cond TOPIMG32:reroute:image
"$PY" -m moedep.run_generation --model "$QWEN" --out_dir "$Q/generation_scope2" \
  --mask "SHARED=$A/substrate_shared.json" --mask "TOPIMG16=$A/substrate_topimg16.json" --mask "ALL=$A/substrate_all.json" \
  --cond none --cond SHARED:reroute:image --cond SHARED:reroute:text --cond TOPIMG16:reroute:image --cond TOPIMG16:reroute:text --cond ALL:zero:text

# ---- 4. Qwen3-VL: pruning panel (Table S3). The generation commands extend the folder of block 2.
"$PY" -m moedep.run_ablation --model "$QWEN" --out_dir "$Q/ablation_prune" \
  --mask "V10=$A/substrate_ratio10.json" --mask "LOWLOAD=$A/substrate_lowload.json" \
  --cond none --cond V10:reroute --cond LOWLOAD:reroute --cond random:V10:0 --cond random:V10:1 \
  --sets mm,hard,arc --n_mm 120
"$PY" -m moedep.run_ablation --model "$QWEN" --out_dir "$Q/ablation_prune" \
  --mask "V10=$A/substrate_ratio10.json" --mask "UNION=$A/substrate_union.json" --mask "EL462=$A/substrate_el462.json" \
  --cond UNION:reroute --cond EL462:reroute --cond random:UNION:0 --sets mm,hard,arc --n_mm 120
"$PY" -m moedep.run_generation --model "$QWEN" --out_dir "$Q/generation" \
  --mask "V10=$A/substrate_ratio10.json" --mask "LOWLOAD=$A/substrate_lowload.json" \
  --cond LOWLOAD:reroute --cond random:V10:0
"$PY" -m moedep.run_generation --model "$QWEN" --out_dir "$Q/generation" \
  --mask "UNION=$A/substrate_union.json" --mask "EL462=$A/substrate_el462.json" \
  --cond UNION:reroute --cond EL462:reroute --cond random:UNION:0

# ---- 5. Qwen3-VL: what the vision-exclusive experts receive and produce (Fig. 4)
"$PY" -m moedep.patch_analysis "$Q"                                 # CPU
"$PY" -m moedep.expert_norms --model "$QWEN" --out_dir "$Q/expert_norms" --analysis "$A" --n_items 40

# ---- 6. Qwen3-VL: deprivation and re-exposure training (Fig. 5, Table S2)
"$PY" -m moedep.train_deprive --model "$QWEN" --analysis "$A" --out_dir "$RUNS/deprive_aux_off_c0" --cond aux_off --steps 1200 --aux_coef 0 --bias_gamma 0.001 --eval_every 100
"$PY" -m moedep.train_deprive --model "$QWEN" --analysis "$A" --out_dir "$RUNS/deprive_aux_on_c0.01" --cond aux_on --steps 1200 --aux_coef 0.01 --bias_gamma 0.001 --eval_every 100
"$PY" -m moedep.train_deprive --model "$QWEN" --analysis "$A" --out_dir "$RUNS/deprive_aux_on_c0.1" --cond aux_on --steps 1200 --aux_coef 0.1 --bias_gamma 0.001 --eval_every 100
"$PY" -m moedep.train_deprive --model "$QWEN" --analysis "$A" --out_dir "$RUNS/deprive_freeze_V10_c0.01" --cond freeze_V10 --steps 1200 --aux_coef 0.01 --bias_gamma 0.001 --eval_every 100
"$PY" -m moedep.train_deprive --model "$QWEN" --analysis "$A" --out_dir "$RUNS/deprive_block_V10_c0.01" --cond block_V10 --steps 1200 --aux_coef 0.01 --bias_gamma 0.001 --eval_every 100
"$PY" -m moedep.train_deprive --model "$QWEN" --analysis "$A" --out_dir "$RUNS/deprive_bias_balance_c0" --cond bias_balance --steps 1200 --aux_coef 0 --bias_gamma 0.001 --eval_every 100
"$PY" -m moedep.train_deprive --model "$QWEN" --analysis "$A" --out_dir "$RUNS/deprive_bias_balance_c0_g0.02" --cond bias_balance --steps 1200 --aux_coef 0 --bias_gamma 0.02 --eval_every 100
# the two bias conditions measured again with the bias active (diamonds of Fig. 5A, load column of Table S2)
"$PY" -m moedep.train_restore --model "$QWEN" --analysis "$A" --out_dir "$RUNS/evalstate_bias_balance_c0" --state "$RUNS/deprive_bias_balance_c0/state.pt" --steps 0 --keep_bias 1 --max_pixels $((1280*32*32))
"$PY" -m moedep.train_restore --model "$QWEN" --analysis "$A" --out_dir "$RUNS/evalstate_bias_balance_c0_g0.02" --state "$RUNS/deprive_bias_balance_c0_g0.02/state.pt" --steps 0 --keep_bias 1 --max_pixels $((1280*32*32))
# re-exposure: 400 caption steps from each trained state and from the undeprived model. The bias is removed for update
# rate 0.001 and kept for update rate 0.02.
"$PY" -m moedep.train_restore --model "$QWEN" --analysis "$A" --out_dir "$RUNS/restore_none" --state none --steps 400 --eval_every 50
"$PY" -m moedep.train_restore --model "$QWEN" --analysis "$A" --out_dir "$RUNS/restore_aux_off_c0" --state "$RUNS/deprive_aux_off_c0/state.pt" --steps 400 --eval_every 50
"$PY" -m moedep.train_restore --model "$QWEN" --analysis "$A" --out_dir "$RUNS/restore_aux_on_c0.01" --state "$RUNS/deprive_aux_on_c0.01/state.pt" --steps 400 --eval_every 50
"$PY" -m moedep.train_restore --model "$QWEN" --analysis "$A" --out_dir "$RUNS/restore_aux_on_c0.1" --state "$RUNS/deprive_aux_on_c0.1/state.pt" --steps 400 --eval_every 50
"$PY" -m moedep.train_restore --model "$QWEN" --analysis "$A" --out_dir "$RUNS/restore_freeze_V10_c0.01" --state "$RUNS/deprive_freeze_V10_c0.01/state.pt" --steps 400 --eval_every 50
"$PY" -m moedep.train_restore --model "$QWEN" --analysis "$A" --out_dir "$RUNS/restore_block_V10_c0.01" --state "$RUNS/deprive_block_V10_c0.01/state.pt" --steps 400 --eval_every 50
"$PY" -m moedep.train_restore --model "$QWEN" --analysis "$A" --out_dir "$RUNS/restore_bias_balance_c0" --state "$RUNS/deprive_bias_balance_c0/state.pt" --steps 400 --eval_every 50 --keep_bias 0
"$PY" -m moedep.train_restore --model "$QWEN" --analysis "$A" --out_dir "$RUNS/restore_bias_balance_c0_g0.02" --state "$RUNS/deprive_bias_balance_c0_g0.02/state.pt" --steps 400 --eval_every 50 --keep_bias 1
"$PY" -m moedep.summarize_deprive 'deprive_*'                       # CPU, prints the metrics of every evaluation step
"$PY" -m moedep.summarize_deprive 'evalstate_*'                     # CPU
"$PY" -m moedep.summarize_deprive 'restore_*'                       # CPU

# ---- 7. Kimi-VL: routing pass, expert sets, lesions, scoped lesions (Fig. 2, Fig. 3B, Fig. S1, Table 2, Table S6)
"$PY_KIMI" -m moedep.run_routing --model "$KIMI" --out_dir "$K" --sets mm,text,ascii,shapes --n_mm 120 --n_alpaca 400 --n_qonly 300 --n_concepts 300 --n_var 6
"$PY" -m moedep.analyze_routing --run_dir "$K"                      # CPU
"$PY" -m moedep.expert_sets --run_dir "$K"                          # CPU
"$PY_KIMI" -m moedep.run_ablation --model "$KIMI" --out_dir "$K/ablation" \
  --mask "V10=$KA/substrate_ratio10.json" --mask "TOPIMG16=$KA/substrate_topimg16.json" --mask "TOPIMG16_noV10=$KA/substrate_topimg16_noV10.json" \
  --mask "TOPIMG16_V10part=$KA/substrate_topimg16_V10part.json" --mask "SHARED=$KA/substrate_shared.json" --mask "TOPTXT16=$KA/substrate_toptxt16.json" \
  --cond none --cond V10:reroute --cond V10:zero --cond TOPIMG16:reroute --cond TOPIMG16_noV10:reroute --cond TOPIMG16_V10part:reroute \
  --cond SHARED:reroute --cond TOPTXT16:reroute --cond random:V10:0 --sets mm,shapes --n_mm 120 --n_var 6
"$PY_KIMI" -m moedep.run_ablation --model "$KIMI" --out_dir "$K/ablation_scope" \
  --mask "V10=$KA/substrate_ratio10.json" --mask "TOPIMG16_noV10=$KA/substrate_topimg16_noV10.json" --mask "SHARED=$KA/substrate_shared.json" --mask "ALL=$KA/substrate_all.json" \
  --cond none --cond TOPIMG16_noV10:reroute --cond TOPIMG16_noV10:reroute:image --cond TOPIMG16_noV10:reroute:text \
  --cond V10:reroute:image --cond V10:reroute:text --cond SHARED:reroute:image --cond SHARED:reroute:text --cond ALL:zero:image \
  --sets mm,hard,arc --n_mm 120
"$PY_KIMI" -m moedep.run_ablation --model "$KIMI" --out_dir "$K/ablation_scope2" \
  --mask "ALL=$KA/substrate_all.json" --cond none --cond ALL:zero:text --sets mm,hard,arc --n_mm 120

# ---- 8. Kimi-VL without its balancing bias (Fig. 6, Table 2, middle block of Table S6)
"$PY_KIMI" -m moedep.run_routing --model "$KIMI" --out_dir "$KNB" --sets mm,text,ascii --n_mm 120 --n_alpaca 400 --n_qonly 300 --n_concepts 300 --bias_scale 0
"$PY" -m moedep.analyze_routing --run_dir "$KNB"                    # CPU
"$PY" -m moedep.expert_sets --run_dir "$KNB" --overlap_with "$K"    # CPU
# lesions under both routing regimes: native bias (scale 1) and bias removed (scale 0)
"$PY_KIMI" -m moedep.run_ablation --model "$KIMI" --out_dir "$K/ablation_bias1" --bias_scale 1 \
  --mask "V10=$KA/substrate_ratio10.json" --mask "V10nb=$KNB/analysis/substrate_ratio10.json" \
  --mask "TOPIMG16_noV10=$KA/substrate_topimg16_noV10.json" --mask "TOPIMG16_V10part=$KA/substrate_topimg16_V10part.json" \
  --cond none --cond V10:reroute --cond V10nb:reroute --cond TOPIMG16_noV10:reroute --cond TOPIMG16_V10part:reroute \
  --sets mm,shapes,arc --n_mm 120 --n_var 6
"$PY_KIMI" -m moedep.run_ablation --model "$KIMI" --out_dir "$K/ablation_bias0" --bias_scale 0 \
  --mask "V10=$KA/substrate_ratio10.json" --mask "V10nb=$KNB/analysis/substrate_ratio10.json" \
  --mask "TOPIMG16_noV10=$KA/substrate_topimg16_noV10.json" --mask "TOPIMG16_V10part=$KA/substrate_topimg16_V10part.json" \
  --cond none --cond V10:reroute --cond V10nb:reroute --cond TOPIMG16_noV10:reroute --cond TOPIMG16_V10part:reroute \
  --sets mm,shapes,arc --n_mm 120 --n_var 6
"$PY_KIMI" -m moedep.run_ablation --model "$KIMI" --out_dir "$K/ablation_scope_bias0" --bias_scale 0 \
  --mask "V10nb=$KNB/analysis/substrate_ratio10.json" --mask "TOPIMG16_noV10=$KA/substrate_topimg16_noV10.json" --mask "ALL=$KA/substrate_all.json" \
  --cond none --cond ALL:zero:image --cond TOPIMG16_noV10:reroute:image --cond TOPIMG16_noV10:reroute:text \
  --cond V10nb:reroute:image --cond V10nb:reroute:text \
  --sets mm,hard,arc --n_mm 120

# ---- 9. Gemma-4: the complete pipeline (Fig. 2, Fig. 3B, Fig. S1, Table 2, Table S4, Table S6)
# In the paper's run this short pass came first. Six of its ten items are not among the 600 of the full pass and stay in
# the index, so the routing map of Gemma-4 holds 2,773 items, 606 of them multimodal.
"$PY" -m moedep.run_routing --model "$GEMMA" --out_dir "$G" --sets mm --n_mm 2 --limit 3
"$PY" -m moedep.run_routing --model "$GEMMA" --out_dir "$G" --sets mm,text,ascii,shapes --n_mm 120 --n_alpaca 400 --n_qonly 300 --n_concepts 300 --n_var 6
"$PY" -m moedep.analyze_routing --run_dir "$G"                      # CPU
"$PY" -m moedep.expert_sets --run_dir "$G"                          # CPU
"$PY" -m moedep.run_ablation --model "$GEMMA" --out_dir "$G/ablation" \
  --mask "V10=$GA/substrate_ratio10.json" --mask "TOPIMG16=$GA/substrate_topimg16.json" --mask "TOPIMG16_noV10=$GA/substrate_topimg16_noV10.json" \
  --mask "TOPIMG16_V10part=$GA/substrate_topimg16_V10part.json" --mask "SHARED=$GA/substrate_shared.json" --mask "TOPTXT16=$GA/substrate_toptxt16.json" --mask "LOWLOAD=$GA/substrate_lowload.json" \
  --cond none --cond V10:reroute --cond V10:zero --cond TOPIMG16:reroute --cond TOPIMG16_noV10:reroute --cond TOPIMG16_V10part:reroute \
  --cond SHARED:reroute --cond TOPTXT16:reroute --cond LOWLOAD:reroute --cond random:V10:0 --sets mm,shapes,hard,arc --n_mm 120 --n_var 6
"$PY" -m moedep.run_generation --model "$GEMMA" --out_dir "$G/generation2" \
  --mask "V10=$GA/substrate_ratio10.json" --mask "TOPIMG16_noV10=$GA/substrate_topimg16_noV10.json" --mask "TOPIMG16_V10part=$GA/substrate_topimg16_V10part.json" --mask "LOWLOAD=$GA/substrate_lowload.json" \
  --cond none --cond V10:reroute --cond TOPIMG16_V10part:reroute --cond TOPIMG16_noV10:reroute --cond LOWLOAD:reroute --cond random:V10:0
"$PY" -m moedep.run_ablation --model "$GEMMA" --out_dir "$G/ablation_scope" \
  --mask "V10=$GA/substrate_ratio10.json" --mask "TOPIMG16_noV10=$GA/substrate_topimg16_noV10.json" --mask "SHARED=$GA/substrate_shared.json" --mask "ALL=$GA/substrate_all.json" \
  --cond none --cond TOPIMG16_noV10:reroute --cond TOPIMG16_noV10:reroute:image --cond TOPIMG16_noV10:reroute:text \
  --cond V10:reroute:image --cond V10:reroute:text --cond SHARED:reroute:image --cond SHARED:reroute:text --cond ALL:zero:image \
  --sets mm,hard,arc --n_mm 120
"$PY" -m moedep.run_generation --model "$GEMMA" --out_dir "$G/generation_scope" \
  --mask "TOPIMG16_noV10=$GA/substrate_topimg16_noV10.json" --mask "ALL=$GA/substrate_all.json" \
  --cond none --cond TOPIMG16_noV10:reroute --cond TOPIMG16_noV10:reroute:image --cond TOPIMG16_noV10:reroute:text --cond ALL:zero:image
"$PY" -m moedep.run_ablation --model "$GEMMA" --out_dir "$G/ablation_scope2" \
  --mask "ALL=$GA/substrate_all.json" --cond none --cond ALL:zero:text --sets mm,hard,arc --n_mm 120

# ---- 10. Tables (CPU)
"$PY" -m moedep.tables "$RUNS"

# ---- 11. Control for the text scope (run after the paper, feeds none of its tables or figures)
# The text scope holds the vision delimiters. These runs split it into text without the delimiters and the delimiters alone.
for spec in "$PY|$QWEN|$Q|$A" "$PY_KIMI|$KIMI|$K|$KA" "$PY|$GEMMA|$G|$GA"; do
  IFS="|" read -r py model run an <<< "$spec"
  "$py" -m moedep.run_ablation --model "$model" --out_dir "$run/ablation_delim" \
    --mask "TOPIMG16_noV10=$an/substrate_topimg16_noV10.json" --mask "ALL=$an/substrate_all.json" \
    --cond none --cond TOPIMG16_noV10:reroute:textonly --cond TOPIMG16_noV10:reroute:delim --cond ALL:zero:textonly --cond ALL:zero:delim \
    --sets mm,hard --n_mm 120
  "$PY" -m moedep.intervals "$run/ablation_delim"                   # CPU
done
