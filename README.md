# Lit Up but Not Needed

Code for the paper "Lit Up but Not Needed: Scoped Lesions Show Where Multimodal Mixture-of-Experts Depend on Their Experts".

Repository: https://github.com/gezhixuan/moedep

## What this repository reproduces

The paper studies the routed experts of three mixture-of-experts vision-language models: Qwen3-VL-30B-A3B, Kimi-VL-A3B and Gemma-4-26B-A4B. This repository contains the code that produced its numbers:

- the routing pass, which records the experts that every image token and every text token selects, and the expert sets defined from it (Fig. 2, Table 2)
- lesions of those sets at all positions, by re-routing or by zeroing (Table 1, Tables S1, S3 and S4)
- scoped lesions, which remove a set only at image-token positions or only at text positions (Table 1, Tables S5 and S6, Fig. 3, Fig. S1)
- generation metrics under a lesion: caption negative log-likelihood and greedy TextVQA
- the patch-level and output-level analysis of the vision-exclusive experts (Fig. 4)
- text-only deprivation training and caption re-exposure of Qwen3-VL under seven balancing conditions (Fig. 5, Table S2)
- the removal of the balancing bias of Kimi-VL (Fig. 6)

The repository holds no result files, no model weights, no datasets and no figure code. The runs write per-item result files, and `python -m moedep.tables` prints the tables of the paper from them.

## Contents

| File | Role |
|---|---|
| `moedep/paths.py` | reads the three locations from environment variables |
| `moedep/model.py` | Qwen3-VL wrapper: routing capture, answer scoring, lesions by re-route or zero, scoped lesions |
| `moedep/kimi_model.py` | Kimi-VL wrapper, the scale of its balancing bias, and `load_runner`, which picks the wrapper for a model folder |
| `moedep/gemma_model.py` | Gemma-4 wrapper |
| `moedep/data.py`, `moedep/benchmarks.py` | probe sets and dataset loaders |
| `moedep/run_routing.py` | routing pass over the probe items |
| `moedep/analyze_routing.py` | routing counts per token group, vision-exclusive and text-exclusive sets, loads, divergences (CPU) |
| `moedep/expert_sets.py` | the other expert sets, derived from the routing counts (CPU) |
| `moedep/run_ablation.py` | lesions and scoped lesions, first-token accuracy on every probe |
| `moedep/run_generation.py` | caption NLL and greedy TextVQA under a lesion |
| `moedep/patch_analysis.py`, `moedep/expert_norms.py` | inputs and outputs of the vision-exclusive experts (Fig. 4) |
| `moedep/token_breakdown.py` | text load on the vision-exclusive experts by token type (supporting analysis of the format control) |
| `moedep/lora_experts.py`, `moedep/train_deprive.py`, `moedep/train_restore.py`, `moedep/summarize_deprive.py` | deprivation and re-exposure training |
| `moedep/tables.py` | prints the tables (CPU) |
| `moedep/intervals.py` | prints the paired bootstrap intervals of a lesion run (CPU) |
| `scripts/paper_runs.sh` | every command of the paper, in order |

## Hardware

Each run uses one GPU. The supplement of the paper states that one GPU suffices for every experiment. The runs of the paper used a card with 141 GB of memory, and some lesion runs used a card with 94 GB. Peak GPU memory allocated by PyTorch in those runs:

| Run | Peak memory |
|---|---|
| routing pass and lesions on Qwen3-VL | 58 to 60 GiB |
| routing pass and lesions on Gemma-4 | 49 GiB |
| routing pass and lesions on Kimi-VL | 33 GiB |
| deprivation training | 88 GiB |
| re-exposure training | 71 GiB |

With more than one visible GPU, the Qwen3-VL and Gemma-4 wrappers spread the model over them (`device_map="auto"`).

Run times on the 141 GB card, from the supplement: a routing pass over the 2,767 probe items takes 13 minutes on Qwen3-VL, the full lesion suite 2 hours, one deprivation run 3.2 hours, and the complete Gemma-4 pipeline 65 minutes.

## Installation

```bash
git clone https://github.com/gezhixuan/moedep
cd moedep
pip install -r requirements.txt
```

`requirements.txt` lists the versions of the environment used for Qwen3-VL, Gemma-4 and all CPU analysis (Python 3.11). The Kimi-VL model runs use a second environment with transformers 4.50.3 (Python 3.10):

```bash
pip install -r requirements-kimi.txt
```

All commands are run from the repository root as `python -m moedep.<module>`.

Three environment variables set the locations. The defaults are folders in the repository root.

| Variable | Content | Default |
|---|---|---|
| `MOEDEP_DATA` | datasets | `data/` |
| `MOEDEP_MODELS` | model weights | `models/` |
| `MOEDEP_RUNS` | outputs of every run | `runs/` |

## Models

All three models are used as released. Place each in a folder of this name under `MOEDEP_MODELS`.

| Folder | Model | Source |
|---|---|---|
| `Qwen3-VL-30B-A3B-Instruct` | Qwen3-VL-30B-A3B | Hugging Face model `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| `Kimi-VL-A3B-Instruct` | Kimi-VL-A3B | Hugging Face model `moonshotai/Kimi-VL-A3B-Instruct` |
| `gemma-4-26B-A4B-it` | Gemma-4-26B-A4B | Hugging Face model `google/gemma-4-26B-A4B-it` |

## Datasets

Place each dataset in a folder of this name under `MOEDEP_DATA`. The second column lists the files that the code reads.

| Folder | Files read | Source |
|---|---|---|
| `POPE/` | `data/*.parquet` | Hugging Face dataset `lmms-lab/POPE` |
| `MMVP/` | `Questions.csv`, `MMVP Images/<index>.jpg` | Hugging Face dataset `MMVP/MMVP` |
| `vstar_bench/` | `test_questions.jsonl`, the images and the box file `<image>.json` next to each image | Hugging Face dataset `craigwu/vstar_bench` |
| `ZoomBench/` | `data/test.parquet` | Hugging Face dataset `inclusionAI/ZoomBench` |
| `TextVQA_GT_bbox/` | `data/*.parquet` | Hugging Face dataset `jrzhang/TextVQA_GT_bbox` |
| `HR-Bench/` | `hr_bench_4k.parquet`, `hr_bench_8k.parquet` | Hugging Face dataset `DreamMr/HR-Bench` |
| `MME-RealWorld-Lite/` | `data/*.parquet` | Hugging Face dataset `yifanzhang114/MME-RealWorld-lite-lmms-eval` |
| `alpaca/` | `data/*.parquet` with the columns `instruction`, `input`, `output` | Hugging Face dataset `tatsu-lab/alpaca` |
| `ASCIIEval/` | `test.jsonl` | Hugging Face dataset `ASCIIEval/ASCIIEval` |
| `ai2_arc/` | `ARC-Easy/test-*.parquet` | Hugging Face dataset `allenai/ai2_arc` |
| `ultrachat_200k/` | `data/train_sft-*.parquet`, `data/test_sft-*.parquet` | Hugging Face dataset `HuggingFaceH4/ultrachat_200k` |
| `coco_ann/` | `annotations/captions_val2014.json` (COCO 2014 caption annotations) | cocodataset.org, 2014 train/val annotations |
| `coco_val2014/` | `val2014/COCO_val2014_<12-digit id>.jpg` (COCO val2014 images) | cocodataset.org, 2014 val images |

The COCO captions are needed for the caption NLL. The COCO images are needed only for re-exposure training, and UltraChat only for deprivation training. Check the license of each dataset and model before use.

## Pipeline

`scripts/paper_runs.sh` lists every command of the paper in this order, with the exact arguments. The steps below show the pattern for one model. `$RUN` stands for the run folder of a model, for example `runs/qwen3vl`.

1. Routing pass and expert-set masks. `run_routing` runs one forward per probe item and stores the top-k experts of every token in every MoE layer. `analyze_routing` counts selections per token group and writes the vision-exclusive and text-exclusive sets with a report. `expert_sets` derives the remaining sets. The masks are JSON files `$RUN/analysis/substrate_<name>.json`.

   ```bash
   python -m moedep.run_routing --model models/Qwen3-VL-30B-A3B-Instruct --out_dir $RUN --sets mm,text,ascii,shapes \
     --n_mm 120 --n_alpaca 400 --n_qonly 300 --n_concepts 300 --n_var 6
   python -m moedep.analyze_routing --run_dir $RUN
   python -m moedep.expert_sets --run_dir $RUN --affinity_count 462
   ```

2. Lesions. `run_ablation` scores every probe item under each condition. `--mask NAME=file` names a set, `--cond NAME:reroute` removes it by re-routing, `--cond NAME:zero` by zeroing, `--cond random:NAME:SEED` removes a random set of the same size per layer, and `--cond none` is the intact model.

   ```bash
   python -m moedep.run_ablation --model models/Qwen3-VL-30B-A3B-Instruct --out_dir $RUN/ablation2 \
     --mask V10=$RUN/analysis/substrate_ratio10.json --cond none --cond V10:reroute --cond V10:zero --cond random:V10:0 \
     --sets mm,ascii,shapes --n_mm 120 --n_concepts 300 --n_var 6
   ```

3. Scoped lesions. A third field restricts the lesion to image-token positions or to text positions. The text scope is every position that is not an image token: the chat template, the question, the vision delimiters (two tokens per image in Qwen3-VL and Gemma-4, three in Kimi-VL) and the generated tokens. Two further scopes split it, `textonly` (text without the delimiters) and `delim` (the delimiters alone). In the three models of the paper a lesion at the delimiters alone changes the five-probe mean by less than 2 points, and the text scope without the delimiters reproduces the cost of the text scope (block 11 of `scripts/paper_runs.sh`).

   ```bash
   python -m moedep.run_ablation --model models/Qwen3-VL-30B-A3B-Instruct --out_dir $RUN/ablation_scope \
     --mask TOPIMG16_noV10=$RUN/analysis/substrate_topimg16_noV10.json --mask ALL=$RUN/analysis/substrate_all.json \
     --cond none --cond TOPIMG16_noV10:reroute:image --cond TOPIMG16_noV10:reroute:text --cond ALL:zero:image \
     --sets mm,hard,arc --n_mm 120
   ```

4. Generation metrics. `run_generation` takes the same `--mask` and `--cond` arguments and writes the teacher-forced NLL of the five COCO reference captions of each POPE image and the VQA soft accuracy of a greedy TextVQA answer. The soft accuracy is min(matches / 3, 1) over the ten human answers, after lower-casing and removal of punctuation and articles. The official VQA evaluation also normalizes number words and contractions and averages over leave-one-out subsets of the answers. Rescored that way, the accuracies of the paper are 0.2 to 1.6 points lower and no change against the intact model moves by more than 0.7 points.

5. Deprivation and re-exposure training (Qwen3-VL). `train_deprive` trains on UltraChat text with LoRA on every routed expert and trainable routers, under one of the conditions `aux_off`, `aux_on`, `freeze_V10`, `block_V10`, `bias_balance`. `train_restore` continues from a saved state on COCO captions and leaves out the COCO images of the POPE items, the only tracked probe that is built on COCO. With `--steps 0` it only evaluates a state. `summarize_deprive` prints the metrics of every evaluation step.

6. Kimi-VL bias removal. `--bias_scale 0` in `run_routing` and `run_ablation` scales the native balancing bias of Kimi-VL to zero. The routing pass without the bias gives its own vision-exclusive set, which the lesion runs name `V10nb`.

7. Gemma-4. The same routing pass, expert sets, lesions, scoped lesions and generation metrics, with the Gemma-4 model folder. The routing map of Gemma-4 in the paper covers 606 multimodal items, six more than the other two models (block 9 of `scripts/paper_runs.sh`).

Lesion and generation runs are resumable. They append to `results.jsonl` in the output folder and skip the pairs of item and condition that are already present, so several commands can share one folder. Each run also writes `summary.md` with the accuracy per condition.

## Paper items and commands

Block numbers refer to `scripts/paper_runs.sh`. Run folders are relative to `MOEDEP_RUNS`. The repository does not draw the figures. The last column names the files that hold the numbers of each item.

| Paper item | Blocks | Output |
|---|---|---|
| Fig. 2, numbers of Sec. 4 | 1, 7, 8, 9 | `<run>/analysis/report.md`, `<run>/analysis/counts.npz`, printed output of `expert_sets` |
| Table 1 | 2, 3 | `qwen3vl/ablation2`, `ablation_scope`, `generation`, `generation_scope`, printed by `moedep.tables` |
| Table 2 | 1 to 3, 7 to 9 | printed by `moedep.tables` |
| Fig. 3A | 2, 3 | `qwen3vl/ablation2`, `ablation_hard`, `ablation_scope`, `generation`, `generation_scope` |
| Fig. 3B, Fig. S1 | 3, 7, 9 | `ablation_scope` and `ablation_scope2` of each model |
| Fig. 4 | 5 | `qwen3vl/analysis/patch_report.md`, `qwen3vl/expert_norms/expert_norms.md` |
| Fig. 5, Table S2 | 6 | `metrics.jsonl` in `deprive_*`, `evalstate_*`, `restore_*`, printed by `moedep.summarize_deprive` |
| Fig. 6 | 7, 8 | `kimivl_nobias/analysis`, `kimivl/ablation_bias1`, `kimivl/ablation_bias0` |
| Table S1 | 2 | `qwen3vl/ablation2/summary.md` (rows of the sets `ascii` and `shapes`) |
| Table S3 | 2, 4 | `qwen3vl/ablation_prune`, `generation`, printed by `moedep.tables` |
| Table S4 | 9 | `gemma4/ablation`, `generation2`, printed by `moedep.tables` |
| Table S5 | 3 | `qwen3vl/ablation_scope`, `generation_scope`, `generation_scope2`, printed by `moedep.tables` |
| Table S6 | 7, 8, 9 | `ablation_scope`, `ablation_scope2`, `ablation_scope_bias0` of Kimi-VL and Gemma-4, printed by `moedep.tables` |

## Printing the tables and the intervals

```bash
python -m moedep.tables            # reads MOEDEP_RUNS
python -m moedep.tables /path/to/runs
```

This prints Table 1, Table 2, Tables S3 to S6 and the five-probe means (POPE, MMVP, V*, ZoomBench, TextVQA) from the `results.jsonl` files and, for Table 2, from the routing counts. A cell prints as `n/a` when its run is missing. Each row label ends with the condition name used in the result files.

The paired 95% bootstrap intervals of a lesion run (Fig. 3, Fig. S1 and the intervals quoted in the text) are printed by

```bash
python -m moedep.intervals $MOEDEP_RUNS/qwen3vl/ablation_scope
```

## Internal names

Code, mask files and result files use short internal names for the expert sets. A mask file is `<run>/analysis/substrate_<file>.json`.

| Internal name | Mask file | Term in the paper | Appears in |
|---|---|---|---|
| `V10` | `ratio10` | vision-exclusive experts | Tables 1, 2, S1 to S6, Figs. 2 to 6, Fig. S1 |
| `V10nb` | `ratio10` of the run without the bias | vision-exclusive experts of Kimi-VL defined without its balancing bias | Fig. 6, Table S6 |
| `T10` | `text10` | text-exclusive experts | Fig. 2, Table S1 |
| `TOPIMG16` | `topimg16` | image top-16 set (the 8 highest in Kimi-VL) | Sec. 5, Table S5 |
| `TOPIMG16_V10part` | `topimg16_V10part` | vision-exclusive part of the image top-16 set | Table 1, Table S4, Fig. 3A |
| `TOPIMG16_noV10` | `topimg16_noV10` | dual-use part of the image top-16 set | Table 1, Tables S4 to S6, Figs. 3, 4 and 6, Fig. S1 |
| `TOPIMG32` | `topimg32` | image top-32 set | Table S5 |
| `TOPTXT16` | `toptxt16` | text top-16 set | Table 1 |
| `SHARED` | `shared` | above-uniform set (image load and text load both at least 1/E) | Table 1, Tables S5 and S6, Fig. S1 |
| `ALL` | `all` | every routed expert | Tables 1 and 2, Tables S5 and S6, Fig. 3, Fig. S1 |
| `LOWLOAD` | `lowload` | least-used set | Tables S3 and S4 |
| `UNION` | `union` | union of the vision-exclusive and the least-used set | Table S3 |
| `EL462` | `el462` | affinity-462 set | Table S3 |
| `V05b` | `bshare05` | experts with a balanced text share of at most 5% (`tbshare05`: at least 95%) | Fig. 2A |

Condition names in `results.jsonl` are built from these names.

| Condition | Meaning |
|---|---|
| `none` | intact model |
| `NAME_reroute` | set removed by re-routing at all positions |
| `NAME_zero` | set removed by zeroing at all positions |
| `NAME_reroute_image`, `NAME_zero_image` | scoped lesion at image-token positions |
| `NAME_reroute_text`, `NAME_zero_text` | scoped lesion at text positions |
| `NAME_reroute_textonly`, `NAME_reroute_delim` (also with `zero`) | the text scope without the vision delimiters, and the delimiters alone |
| `random_NAME_SEED` | random set with the size of NAME in every layer, re-routed |

The probes are named `pope`, `mmvp`, `vstar`, `zoombench`, `textvqa`, `hr_bench_4k`, `hr_bench_8k`, `mmerw_lite` and `arc_easy`. The training conditions `aux_off`, `aux_on` (coefficient 0.01 or 0.1), `freeze_V10`, `block_V10` and `bias_balance` (update rate 0.001 or 0.02) are the rows aux off, aux 0.01, aux 0.1, freeze, block, bias 0.001 and bias 0.02 of Table S2.

## Random controls and seeds

`--cond random:NAME:SEED` draws, in every layer, as many experts as the set NAME has in that layer from the experts outside NAME, with `numpy.random.default_rng(100 + SEED)`. The five random draws of Table 1 are seeds 0 and 1 (folders `ablation2` and `ablation_prune`) and seeds 2 to 4 (folder `ablation_scope`). The generation metrics of the random control use seed 0. The random control of the 1,454-expert union in Table S3 is `random:UNION:0`.

The probe items are drawn with `random.Random(seed)` and `--seed 0`, the default of every script. Training uses `--seed 0` for the order of the dialogues and for the choice of caption images.

## Notes on exact reproduction

The numbers of the paper were produced with this code, with three corrections made for the release. None of them changes the design of an experiment.

- The option order and the line shuffling of the ASCIIEval items were seeded with Python's string hash, which differs between processes. They now use a fixed hash. The ASCIIEval rows of Table S1 therefore reproduce up to the order of the answer options. Conditions inside one run were always compared on identical items.
- `update_balance_bias` added the balancing bias to router logits that already include it since a later change of the router patch. The two bias conditions of the paper were trained before that change and counted the bias once, as the code does again.
- The caption drawn for each image in re-exposure training now follows `--seed`.

First-token results can differ by one item of a probe between GPU types. A lesion that leaves the model near chance, such as the dual-use part of the image top-16 set at all positions, differed by up to three items per probe between two of our runs, with the same mean over probes.

## Citation

```bibtex
@misc{ge2026litup,
  title  = {Lit Up but Not Needed: Scoped Lesions Show Where Multimodal Mixture-of-Experts Depend on Their Experts},
  author = {Zhixuan Ge and Anqi Li and Kexin Geng and Wei Qiu},
  year   = {2026},
  note   = {arXiv preprint, number to be added}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The font `moedep/fonts/DejaVuSansMono.ttf`, used to render ASCII art as images, is distributed under its own license (see `moedep/fonts/LICENSE_DEJAVU`).
