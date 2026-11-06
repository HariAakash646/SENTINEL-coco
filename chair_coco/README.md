# SENTINEL on COCO captioning (CHAIR), any HF VLM

## Quick start: the whole pipeline on any machine

Needs: Linux x86_64, NVIDIA GPUs with ~40 GB or more each (developed on L40S 46 GB) and a driver for CUDA 12.8, ~60 GB
of disk, internet on the first run, and [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
Everything else (Python 3.12 venv with the exact package versions, model weights, COCO images) is fetched by the script.

```bash
cd SENTINEL
echo 'OPENAI_API_KEY=sk-...' > .env              # gpt-5.1 judge key; .env is git-ignored (or: export OPENAI_API_KEY=...)
tmux new -s sentinel                             # long run: survives a dropped connection
bash run_pipeline.sh qwen2_5_vl_3b 42            # <llm> <seed>: setup, generate 7000 pairs, train, eval
GPUS=0,1 bash run_pipeline.sh qwen2_5_vl_3b 42   # same on two GPUs (generation runs one process per GPU)
bash run_pipeline.sh qwen2_5_vl_3b 42 judge      # a single stage; every stage resumes where it stopped
```

- **Stages:** `setup` (venv, models, images, COCO annotations) → `profile` (FLOP cost profile, CPU) → `generate`
  (SENTINEL's preference pairs, until 7000) → `pairs` → `train` (C-DPO, LoRA, the paper's recipe) → `captions`
  (greedy captions of the 1,000 v2 RL-test images, base model and SENTINEL) → `judge` (the paper's Table 1 metrics:
  CHAIR, matcher metrics, gpt-5.1 judge, against the base model). `eval` = `captions judge`.
- **GPU or not:** only `generate pairs train captions` use the GPUs. `setup` and `profile` are downloads and CPU
  work, and `judge` is minutes of CPU and then waiting on the Batch API. Where GPU time is allocated (Slurm),
  run them as separate jobs on the same `WORK`, so the GPUs are busy for the whole of their job:
  ```bash
  bash run_pipeline.sh qwen2_5_vl_3b 42 setup profile                   # CPU-only job
  GPUS=0,1 bash run_pipeline.sh qwen2_5_vl_3b 42 generate pairs train captions   # GPU job
  JUDGE_WAIT=0 bash run_pipeline.sh qwen2_5_vl_3b 42 judge              # CPU-only: scores, submits the batches, exits
  bash run_pipeline.sh qwen2_5_vl_3b 42 judge                           # later, CPU-only: collects, writes the table
  ```
  Generation runs in a few large rounds (100 images per GPU to measure the pair yield, then 90% of what the rest
  needs, then small top-ups), not rounds of 100: each round reloads every model on every GPU and waits for the
  slowest GPU. `CHUNK=N` fixes the round size instead.
- **Models:** the `LLM_PATH` table at the top of `run_pipeline.sh` maps a name to its weights, a Hugging Face id or a
  local directory; change the path there to swap a model. A new name also needs `configs/<name>.yaml` (training settings).
  Gemma 3 is gated: accept its license on Hugging Face and export `HF_TOKEN`.
- **Seed:** sets generation sampling and training (data order, LoRA init). The image lists and the eval decoding are
  fixed, so seeds are compared on the same data. The paper reports three seeds (42, 43, 44).
- **Settings** (environment variables): `GPUS`, `PAIRS` (7000), `BATCH` (64) and `EPOCHS` (1) (a non-default value is
  added to the run's name, e.g. `cdpo_2000_bs16`, and reuses the generated pairs), `WORK` (./work), `N_IMAGES` (8000, up to 16000),
  `JUDGE_BUDGET` ($10 per run), `JUDGE_MODE` (`batch` / `live`), `JUDGE_IMAGE_SOURCE` (`url` / `base64`),
  `RUN_JUDGE`, `MATCHED`, `HEDGE_STRICT`, `COCO_TRAIN2014_DIR` / `COCO_VAL2014_DIR` /
  `COCO_ANN_DIR` (use a local COCO), `HF_TOKEN`; see the header of `run_pipeline.sh`.
- **Smoke test first** (~1 h on one GPU, under $0.10 of judging): `PAIRS=64 LIMIT=8 bash run_pipeline.sh qwen2_5_vl_3b 1`
  (add `JUDGE_MODE=live` to skip the batch wait)
  runs every stage on 64 pairs (one training update; fewer than 64 cannot fill the batch) and evaluates 8 test
  images (the reference CHAIR needs all 1,000 and is skipped). It writes to its own run dir; downloads are reused.
- **Results** in `work/runs/<llm>_seed<seed>/`: `eval/cdpo_7000/table.md` (this seed's Table 1 row and paired
  differences), the files behind it (below), `train/cdpo_7000/checkpoint-final` (the LoRA adapter),
  `train/cdpo_7000/flops_pipeline.json` (FLOPs), `logs/`. The Table 1 row over seeds, mean ± sd:
  `work/venv/bin/python chair_coco/eval/paper_table.py --name SENTINEL work/runs/qwen2_5_vl_3b_seed*/eval/cdpo_7000`
- **Time and cost** (Qwen2.5-VL-3B, one L40S, measured on the 2000-pair run): generation ~0.9 images/min per GPU at
  ~2.45 pairs/image, so 7000 pairs take ~2,900 images, ~53 h on one GPU or ~27 h on two; training ~1 h 50 min; eval
  ~2 h (HF captions of 1,000 images, base and model), plus the judge's batch wait (below). The gpt-5.1 judge goes
  through the OpenAI Batch API, as in the paper: ~$1.2 per 1,000 captions, ~$3.5 on a model's first eval (base, model,
  matched base), ~$2.3 after (`JUDGE_MODE=live`: twice that).
- **The batch judge** (`judge_batch.py`): the eval submits every caption it needs judged as one campaign and waits
  for it, usually minutes to a few hours, at most 24 h (OpenAI's completion window). Stopping the script (Ctrl-C, a
  lost session) does not stop the batches; rerun the same command and it picks them up from
  `eval/cdpo_7000/judge_batch/state.json`. Before submitting it books a deliberately high estimate in
  `eval/judge_spend.jsonl` (~$1.8 per 1,000) and stops if that would pass `JUDGE_BUDGET`; on collecting it books the
  real cost. `python chair_coco/judge_batch.py --workdir <eval dir>/judge_batch --status` (or `--cancel`) shows (or
  cancels) the batches.

## Data: the probe project's v2 splits

The paper evaluates every method on the v2 splits of the probe repo (`new_data/coco_v2/`, copied to `data/coco_v2/`):

| split | images | source | used here |
|---|---|---|---|
| `rl_test` | 1,000 | val2014 images that COCO 2017 kept as val2017 | the test set of every eval |
| `probe_test` | 300 | same | not used (excluded from training) |
| `rl_train` | 3,000 | train2014 | SENTINEL's first 3,000 training images |
| `probe_train` | 1,000 | train2014 | not used (excluded from training) |

The four are disjoint by image id and by image bytes. The test images come only from val2017 because COCO 2017 moved
most of val2014 into train2017, and LLaVA-1.5's instruction data uses train2017 images.

SENTINEL's candidate training images (`data/train2014_images.tsv`, made by `prepare_data.py`): the 3,000 `rl_train`
images first, in their manifest order, the images the RL policy and the VES-RFT and PAPO baselines train on; then 13,000
train2014 images with at least two COCO categories, in a fixed shuffled order (seed 42), disjoint from all four v2
splits by id and by bytes (one byte-identical image dropped). At ~2.45 pairs per image the 7,000 pairs need ~2,900
images, so SENTINEL trains almost only on the RL-training images, and never on the probe's training images or on a
test image. There is no validation split: as in the paper, the final checkpoint is evaluated, fixed before training.

Changes from the first version of this port: the test set was the older 600-image `coco_rl_test` (24 images in common
with v2's 1,000), which the probe repo has retired as not comparable; and the training list was train2014 minus the
old (val2014) splits, whose first 4,000 images happened to be exactly v2's `probe_train` and `rl_train`, in the same
seeded order, because both lists came from the same shuffle of train2014.

## Eval: the paper's Table 1

`eval_probe.sh` follows the probe repo's v2 eval step for step (`src/run.py`: `eval:rollout`, `eval:judge`,
`eval:compare`; `store/post_judge_run.sh`: truncation null, reference CHAIR, matched-length judging), with
byte-identical copies of its scripts in `eval/` (`eval/PROVENANCE.md`):

1. **Captions:** `rollout_acts.py`, HF transformers `generate()`, greedy, up to 512 new tokens, the 1,000 `rl_test`
   images, "Describe this image in detail." as the only user turn of the chat template (no system prompt is set;
   Qwen's template adds "You are a helpful assistant."), processor `max_pixels` 1003520 for Qwen (`LLM_MAX_PIXELS`).
2. **Truncation null:** `truncation_null.py` scores both caption sets with the COCO synonym matcher against the
   segmentation labels, and cuts each base caption to the model's caption length (`base_trunc_token/`).
3. **Reference CHAIR:** `ref_chair.py` runs OPERA's copy of the CHAIR code (`reference_chair_opera.py`) on the base,
   the model and the matched base.
4. **Judge:** gpt-5.1 with `judge_v3.txt` (image detail high) on the base, the model and the matched base; with
   `HEDGE_STRICT=1` also `judge_v3_nohedge.txt`, which judges a hedged claim as if it were stated plainly. Through
   the OpenAI Batch API with COCO image URLs (`judge_batch.py`), as the paper's judgments were made (their
   `meta.json`: `transport: openai-batch`, `image_source: url`); each request is the one `judge_openai.py` sends live.
5. **Judged comparison:** `compare_greedy_arms.py`, model vs base, paired, 10,000 bootstrap resamples, seed 42:
   `judged_compare_full.json`, `judged_compare_matched.json` (and `_matched_nohedge.json`).
6. **Table:** `paper_table.py` -> `table.md`; given several seeds' eval dirs it prints the Table 1 row as mean ± sd.

Checked against the paper (CPU, no API): run on the stored v2 Qwen2.5-VL-3B files, `paper_table.py` prints the paper's
base row (41.9, 9.2, 0.593, 2.27, 31.5, 70.5, 80.0, 203.5) and method row (33.1±0.8 ... 178.6±14.1) exactly; recomputed
from the stored seed-42 captions and judgments, `truncation_null.json`, `judged_compare_full.json`,
`judged_compare_matched.json` and the matched base captions are identical to the paper's files, and `ref_chair.json`
too except the base's CHAIR_I in the 5th decimal (9.2101% vs 9.2089%, one object mention: the part-of-speech tagger of
NLTK 3.9.2 differs slightly from the one the paper ran).

### The Table 1 columns

| column | definition | file, field |
|---|---|---|
| CHAIR_S ↓ | share of captions that mention at least one hallucinated COCO object | `ref_chair.json`, `ref.arms.<name>.rl.CHAIR_s` |
| CHAIR_I ↓ | hallucinated object mentions / all object mentions, every mention counted (repeats too) | same, `CHAIR_i` |
| Recall ↑ | ground-truth objects the caption mentions / ground-truth objects, pooled over images | same, `recall` |
| Wrong obj./cap. ↓ | COCO categories the caption names that are not in the image's labels, per caption | `truncation_null.json`, `arms.RL (<name>).halluc_per_caption` |
| Precision ↑ | named categories that are in the labels / named categories, pooled over images | same, `precision` |
| Tokens | mean caption length in tokens | same, `mean_tokens` |
| Judged err./cap. ↓ | statements per caption the gpt-5.1 judge flags as clearly not supported by the image: wrong objects, attributes, counts, relations | `judged_compare_full.json`, `post.judge_derived.mean_findings_per_caption` |
| Clean ↑ | share of captions with no judged error | same, `clean_caption_rate` |

The CHAIR columns use the standard CHAIR ground truth: the image's COCO segmentation labels plus the objects named in
its five human COCO captions, words mapped to objects by CHAIR's own synonym list and lemmatizer. Wrong objects and
precision use the probe repo's synonym matcher and the segmentation labels only. The judge does not flag a hedged claim
("appears to be ...") unless the object is clearly absent; the hedge-strict prompt removes that exception.

Also in `table.md` per seed: the paired difference to the base for every column with its 95% bootstrap interval
(CHAIR: 2,000 resamples, seed 0, as in the paper's script; judged: 10,000, seed 42; matcher metrics: 10,000, seed 42),
an exact two-sided McNemar test of the clean-caption rate, and the matched-length results.

On the sathwik box or turing, the paper's own v2 base arm can replace the regenerated one, so the base row is the
paper's exactly: `BASE_ROLLOUTS=<v2 run>/rollouts/rl_test_base BASE_JUDGED=<v2 run>/judged/rl_test_base/gpt-5.1-high-f44096cee394`
(on turing: `~/probe-rl-hallucination/store/runs/v2_qwen25vl3b/`).

The sections below record how the port was built and checked on the original machine (paths there are that machine's).

A port of the release in this repo (ICCV 2025; data generation in `run/`, `model/`; C-DPO training through
LLaMA-Factory for Qwen, HA-DPO code for LLaVA-1.5) to COCO train2014 images and to models loaded through vLLM and
`AutoModelForImageTextToText`: Qwen2.5-VL-3B now, with configs for `llava-hf/llava-1.5-7b-hf` and
`google/gemma-3-4b-it`. It was first evaluated on the probe repo's older 600-image split; it now uses the paper's v2 eval (above).

The release's data-generation loop runs on its own helpers (sampling, coreference, object extraction, detector
cross-check, pairing); the training is a re-implementation of its LLaMA-Factory C-DPO recipe in the probe
environment.

## Pipeline

| step | script | GPU | output under `run_dir` |
|---|---|---|---|
| – | `flop_profile.py`: the backbone's FLOP cost profile, once per model (CPU, ~10 min) | – | `/scratch/sathwik/flop_profiles/<model>.json` |
| 0 | `prepare_data.py`: 8000 COCO train2014 images (≥ 2 GT categories, no probe image), seeded order | – | `data/images.jsonl` (done) |
| 1 | `generate.py`: SENTINEL sampling + detection + pairing, resumable | 1 | `results/<model>.jsonl`, `results/<model>_data_pair.jsonl`, `flops/generate.json` |
| 2 | `build_pairs.py`: first 7000 pairs | – | `data/train_pairs.jsonl` |
| 3 | `train_cdpo.py`: C-DPO, LoRA r128, 1 epoch | 1–2 | `train/cdpo/checkpoint-final`, `train/cdpo/flops.json`, `train/cdpo/flops_pipeline.json` |
| 4 | `eval_probe.sh`: the paper's Table 1 eval on the v2 `rl_test` split (see above) | 1 | `eval/<name>/` |

```bash
cd ~/SENTINEL
source /scratch/sathwik/sentinel_runs/env.sh      # HF cache, NLTK, Ultralytics dirs, $SENTINEL_PY
C=chair_coco/configs/qwen2_5_vl_3b.yaml             # or llava_1_5_7b.yaml / gemma3_4b.yaml
R=/scratch/sathwik/sentinel_runs/qwen2_5_vl_3b

CUDA_VISIBLE_DEVICES= ~/.venv/bin/python chair_coco/flop_profile.py --model Qwen/Qwen2.5-VL-3B-Instruct   # once per backbone
CUDA_VISIBLE_DEVICES=1 $SENTINEL_PY chair_coco/generate.py --config $C --num-images 1000   # resumable; repeat until >= 7000 pairs
~/.venv/bin/python chair_coco/build_pairs.py --config $C
CUDA_VISIBLE_DEVICES=0,1 ~/.venv/bin/accelerate launch --num_processes 2 chair_coco/train_cdpo.py --config $C
RUN_DIR=$R GPU=0 bash chair_coco/eval_probe.sh cdpo_final --adapter $R/train/cdpo/checkpoint-final
```

`generate.py` runs in an overlay venv, `/scratch/sathwik/sentinel_runs/env`, which sees every package of `~/.venv`
through a `.pth` file and adds only what the release's tools need: spaCy 3.8.18 + `en_core_web_md` 3.8.0,
fastcoref 2.1.6, ultralytics 8.4.195, supervision, pyDeprecate. `~/.venv` is not modified. Training and eval
run in `~/.venv`. Tool weights: GroundingDINO-base, `lizhuang144/flan-t5-base-VG-factual-sg` and
`biu-nlp/lingmess-coref` in `HF_HUB_CACHE=/scratch/sathwik/probe_language_models`; `yolo11x.pt` (sha256-checked,
from `Ultralytics/YOLO11` on HF) in `/scratch/sathwik/sentinel_runs/tools/yolo/`.

For another model, also pass `BASE_MODEL=<hf id>` to `eval_probe.sh`; it then generates and judges a BASE arm for that
model, since stored BASE / GRPO arms exist only for Qwen2.5-VL-3B. `eval_probe.sh` runs the judge by default (Batch
API, ~$1.2 per 1,000 captions; live, ~$2.4: GRPO ck250's 600 cost $1.42; spend is capped at `JUDGE_BUDGET` dollars,
default 10, per run dir by `judge_batch.py` / `judge_capped.py`); `RUN_JUDGE=0` skips it, `SAMPLED=0` skips the T=0.7 n=3 replicates, `LIMIT=N` evaluates the
first N test images only. No LLM judge is needed anywhere in SENTINEL (two detectors decide hallucination), so the
Gemma-3-12B vLLM judge is not used.

## Data compared with the release

| role | release | here |
|---|---|---|
| images | 7814 Visual Genome images (`image_data.jsonl`) | COCO **train2014**: the 3,000 v2 `rl_train` images first, then train2014 images outside every v2 split (see *Data: the probe project's v2 splits*); disjoint from the val2014 test set by construction (asserted) |
| prompt | four detail-caption prompts | `Describe this image in detail.`, the eval prompt (one of the four) |
| pairs | first 7000 of the generation order (Qwen2.5-VL-7B: 7000 pairs from 2177 images, 3.2 / image) | first 7000 in image order; 2.45 / image on the 2000-pair run (817 images) |

Caveat: YOLO11x is trained on COCO train2017, which contains these train2014 images. On our training images its labels
are therefore close to the COCO annotations that CHAIR scores against, so the preference signal is more COCO-aligned
than the release's on Visual Genome. The detectors only label training data, and the v2 test images (val2017) are not
in train2017, so neither detector has seen a test image. OPA-DPO (detail_23k GT) and CHAIR-DPO / PAPO (CHAIR-family rewards) carry related overlaps.

## Fidelity to the release

| item | release | this port | why |
|---|---|---|---|
| sampling | n=10, T=0.7, stop at ".", "!", "?", ".\n", ".\n\n", EOS; max 512 tokens | identical (hard-coded in the loop); top_p 1, top_k off, repetition_penalty 1 set explicitly | vLLM ≥ 0.8 would apply Qwen's generation_config (top_k=1, i.e. greedy, rep. penalty 1.05); the release's vllm 0.7.3 did not |
| prompt continuation | chat template + assistant turn minus `<\|im_end\|>\n` | chat template + generation prompt + context (the same string for Qwen) | works for any chat template |
| stop / finish rule, coreference, object extraction, DINO (base, 0.35 / 0.25), YOLO11x, uncertain-object handling, pairing (`maybe_build_pair`), context choice, bootstrapping | release code | release code, called as is | |
| text stored in pairs and context | public code: coreference-**resolved** sentences | the model's **raw** sentences (`--pair-text raw`, default); resolved copies still drive object extraction | the released data is raw text: in the Qwen2.5-VL-7B set 470 / 14,000 responses start with a pronoun and 3 start lowercase (quote continuations); the public code gave 6 / 38 lowercase starts ("a bicycle has a basket.") and "The focal point of The image". `--pair-text resolved` reproduces the public code |
| fastcoref | transformers 4.51 | eager attention + `post_init()` patched in | transformers 5 defaults to SDPA (Longformer has none) and needs `post_init()`; all checkpoint keys load |
| GroundingDINO post-processing | `box_threshold=` | `threshold=` | renamed in transformers ≥ 4.51 |
| spaCy model | `en_core_web_md` for spaCy 3.8 | 3.8.0 | |
| pair conversion | `get_llama_factory_data_pair.py` reads `win` / `lose` | reads `y_win` / `y_lose`, which is what generation writes | the release script would emit empty responses |
| C-DPO masking | LLaMA-Factory `pairwise.py`: truncate each response after its last "." token, then mask the context by token search | labels = the target sentence's tokens, from character offsets, no EOS | see below |
| loss | DPO sigmoid, β 0.1, reference = base model (LoRA disabled), log-probs summed | identical | |
| LoRA | r128, `lora_target all` (alpha 2r = 256, dropout 0), vision tower + projector frozen | identical; targets = every LM linear layer | |
| optimisation | lr 3e-6 cosine, warmup 0, 1 epoch, global batch 64, AdamW, wd 0, clip 1.0, bf16, cutoff 4096 | identical | |
| lr for Qwen2.5-VL-3B | no 3B recipe (7B: 3e-6; Qwen2-VL-2B: 2e-6) | 3e-6 | same family as the 7B recipe |
| image size | native (vLLM) / LLaMA-Factory default 768² | `max_pixels` 1003520 everywhere, the probe eval resize | COCO images (≤ 640 px) are unchanged by either |

### The release's C-DPO masking fails on 39% of its own Qwen pairs

`pairwise.py` appends `context + " " + y` to the prompt, cuts each response after the last `.` token, then finds the
context's token ids in the sequence and masks everything up to them. Qwen's tokenizer fuses a sentence-final period
with a following newline (`.\n` and `.\n\n` are single tokens). When the target sentence ends that way it has no
standalone `.` token, so the cut falls back inside the context; the context search then fails and nothing is masked,
and the pair trains on context text instead of the sentence. On the released `Qwen2_5_VL_7B_Instruct_SENTINEL_7k.json`
this happens to the chosen response in 1899 pairs, the rejected one in 1926, and at least one of them in 2763 / 7000
(39%; `/scratch/sathwik/sentinel_runs/release_data/`). The paper's Eq. 2 excludes only the context from the gradient,
which is what `train_cdpo.py` does for every pair.

## Smoke test (2026-10-31, one L40S)

| stage | result |
|---|---|
| generate, 6 images, batch 5 | 16 pairs from 5 images (3.2 / image, as in the release), 5.2 min, 0 lowercase-initial responses |
| build_pairs | 16 kept, no empty contexts |
| train_cdpo, 2 steps × 8 pairs | step-1 loss = ln 2 with zero rewards (policy = reference), ~2.3 s / pair, 13.5 GB peak |
| eval, 4 test images | rollouts with the LoRA in vLLM, judge-free tables vs BASE, BASE / GRPO and vLLM BASE, gpt-5.1 judge ($0.01) and paired reports vs BASE and GRPO |

Also checked: `train_cdpo.py` takes a step on Gemma-3-4B (238 LoRA targets, no vision modules); on LLaVA-1.5-7B the
same check ran out of memory because another user's 30 GB job was on the GPU, so it is untested there.

Throughput (one L40S): batch 5 (release) 1.2 images/min; batch 16 with GroundingDINO in chunks of ≤ 5 and vLLM at
0.4 of the GPU 2.4 images/min (`/scratch/sathwik/sentinel_runs/timing_b16`, now the config default). Batch 16 with
unchunked DINO runs out of memory. Chunks of 5 and 10 give identical detections; single-image calls differ on 1 borderline
box in 10 images (the processor pads images per batch), a dependence the release's own batch of 5 already has.

Pair yield is the open number: 3.2 pairs / image on the 6 smoke images, 0.6 on the 16 timing images (fewer
hallucinated objects there: 0.5 vs 1.2 per image), 1.2 overall. At 1.2, 7000 pairs need ~6000 images (~40 h at
2.4 images/min); at the release's 3.2, ~2200 images (~15 h). Run `generate.py --num-images 1000`, check the pair count,
and continue (it resumes); the manifest has 8000 images and `prepare_data.py --n-images` can extend it (65,268 in the
pool). Training ~4.5 h on one L40S (7000 × 2.3 s), about half on two; eval ~15 min + judge.

## FLOPs

Counted with ~/VESRFT's convention ("model FLOPs": matmuls on real tokens, multiply-add = 2, dense attention,
padding excluded; the full list is the docstring of `flops.py`), so the totals compare with VES-RFT's. The
VLM's cost comes from a profile measured once per backbone with torch's FlopCounterMode
(`/scratch/sathwik/flop_profiles/Qwen2.5-VL-3B-Instruct.json`: per token a = 5.55e9, attention b = 294,912 per
token², logits h = 6.22e8, vision c = 5.16e9 per image token; held-out passes fit to 7e-5).

| stage | file | what is counted |
|---|---|---|
| generate | `flops/generate.json` (a resumed run continues it) | VLM prefill (one per request, shared by its 10 samples) + decode per sample, from token counts; YOLO11x, fastcoref, FACTUAL T5 and GroundingDINO measured with FlopCounterMode around each release call |
| train | `train/<name>/flops.json` (and a copy in each checkpoint) | reference forward, LoRA policy forward, backward, checkpoint recompute, AdamW, per update and summed over GPUs |
| total | `train/<name>/flops_pipeline.json` | generate + train |

Eval FLOPs are not counted. The generation counts are what was spent, including images that gave no pair, pairs
beyond the 7000 used, and work redone after an interruption. Three choices to keep in mind when comparing:
each step's prefill re-counts the image and the whole context although vLLM may reuse cached KV blocks (an upper
bound on that part); `extra/vlm_prefill_if_unshared` gives the prefill counted once per sample, the way VES-RFT
counts its generation; spaCy runs on thinc / cupy, which torch does not see, and is not counted (small CNN).

`test_flops.py` checks the accounting against FlopCounterMode on the real model (CPU, fp32, eager):

| check | result |
|---|---|
| prompt token count of the generator vs the processor, with and without context | equal |
| HF generate of 1 and 20 tokens after the prompt (+ context) vs prefill + decode | within 5e-7 |
| C-DPO micro-step: reference forward, LoRA policy forward | within 5e-7 |
| C-DPO micro-step: backward + checkpoint recompute | within 1.4e-3 |
| YOLO11x / fastcoref / T5 parser / GroundingDINO under FlopCounterMode | 1.5e11 / 6.9e11 (2 texts) / 1.3e10 (2 sentences) / 1.1e12 (1 image); counting adds 0.1–0.9 s per call on CPU (`--no-count-tool-flops` turns it off) |

The backward check found that transformers 5's `gradient_checkpointing_enable()` calls `enable_input_require_grads()`,
which also hooks the vision tower's patch embedding: every update then backpropagates through and recomputes the
frozen ViT. `train_cdpo.py` now keeps only the language model's hook (`text_input_require_grads`); the LoRA gradients
are bit-identical and a policy forward + backward costs 1.12e13 instead of 1.49e13 FLOPs per sequence (T = 450).
TRL makes the same call for PEFT + gradient checkpointing, so other TRL-trained arms may pay the same cost.

## Eval: the stored BASE was decoded by a different engine

(History: this was the vLLM eval path. `eval_probe.sh` now runs the main pipeline's HF-engine eval above, so arms
and base are decoded the same way as the stored BASE and GRPO ck250.)

The stored Qwen BASE (`probe_new_runs/comparision/pre_rl_greedy`) and GRPO ck250 arms are HF `generate()` output;
every arm evaluated by `eval_probe.sh` (and by the CHAIR-DPO / PAPO / OPA-DPO launchers) is decoded by vLLM. The base
model alone, on the 600 test images: HF 203 mean tokens, 2.3% clipped, CHAIR_S 41.7, CHAIR_I 20.5; vLLM 287 tokens,
5.5% clipped, CHAIR_S 42.2, CHAIR_I 19.7 (`/scratch/sathwik/sentinel_runs/base_hf_vs_vllm_600.md`). Setting vLLM's
repetition_penalty to Qwen's 1.05 does not close the gap (309 tokens). So length deltas against the stored BASE carry
~+84 tokens of engine effect. `eval_probe.sh` also compares each arm with the base model decoded by vLLM
(`/scratch/sathwik/sentinel_runs/base_vllm_greedy_600`); judge it once with `JUDGE_VLLM_BASE=1` (~$1.6) to get the
engine-matched gpt-5.1 comparison. In the smoke test the 2-step adapter was 369 vs 367 tokens against the vLLM base,
but 369 vs 215 against the stored BASE.

## Files

| file | role |
|---|---|
| `config.py` | YAML-as-argparse-defaults, one section per stage |
| `vlm_common.py` | image resize, chat prompt, model-agnostic batching and log-probs (shared with `~/Nullu/chair_coco`) |
| `prepare_data.py`, `generate.py`, `build_pairs.py`, `train_cdpo.py` | steps 0–3 |
| `flops.py`, `flop_profile.py` | FLOP accounting and the backbone cost profile (shared with `~/Nullu/chair_coco`) |
| `test_flops.py` | checks the accounting against torch's FlopCounterMode on the real model (CPU) |
| `eval_probe.sh` | step 4: CHAIR + gpt-5.1 eval of an adapter or a model against its base model |
| `configs/*.yaml` | Qwen2.5-VL-3B, LLaVA-1.5-7B (hf), Gemma-3-4B |
| `../run_pipeline.sh` | the whole pipeline for one LLM and seed (Quick start) |
| `requirements.txt`, `overrides.txt`, `requirements.lock.txt` | top-level pins, the two version overrides the runs used, and the exact locked environment the script installs |
| `data/train2014_images.tsv`, `data/val2014_test_images.tsv` | image id, file, md5 of the 16,000 candidate training images (seeded order, `prepare_data.py`) and of the 1,000 v2 `rl_test` images |
| `eval/` | the probe repo's v2 eval scripts (vendored, `eval/PROVENANCE.md`), `ref_chair.py`, `paper_table.py` |
| `data/coco_v2/` | the probe repo's v2 split manifests (`rl_test` is the test set) |
| `judge_batch.py` | the gpt-5.1 judge through the OpenAI Batch API (the paper's), under a spend cap; the default |
| `judge_capped.py` | the gpt-5.1 judge with live calls under a spend cap (`JUDGE_MODE=live`) |
| `flop_profiles/` | FLOP cost profiles per backbone (`flop_profile.py`; the pipeline makes missing ones) |
