#!/usr/bin/env bash
# SENTINEL (ICCV 2025) on COCO captioning, end to end on one Linux machine with NVIDIA GPUs:
#   setup -> FLOP profile -> generate preference pairs -> build the training set -> C-DPO training -> eval captions
#   -> judge (CHAIR, gpt-5.1 judge, Table 1 row)
#
#   bash run_pipeline.sh <llm> <seed> [stage ...]
#   bash run_pipeline.sh qwen2_5_vl_3b 42                       # every stage
#   GPUS=0,1 bash run_pipeline.sh qwen2_5_vl_3b 42 generate     # some stages, on two GPUs
#
# Stages, in order (default: all): setup profile generate pairs train captions judge. Each one is resumable: re-running
# the same command skips finished work and continues. Everything is logged to $WORK/runs/<llm>_seed<seed>/logs/pipeline.log.
#   needs a GPU:  generate pairs train captions  (pairs is a minute of CPU between generate and train)
#   needs none:   setup profile                  (downloads, ~1-3 h on a fresh machine; FLOP profile, minutes of CPU)
#                 judge                          (CHAIR and the matcher metrics, minutes of CPU; then the gpt-5.1 Batch
#                                                 API, which mostly waits, up to 24 h)
#   eval = captions judge (the two in one go)
# On a machine where GPU time is allocated and paid for (Slurm), run the groups as separate jobs on the same WORK:
#   bash run_pipeline.sh <llm> <seed> setup profile                 # CPU-only job
#   bash run_pipeline.sh <llm> <seed> generate pairs train captions # GPU job: the GPUs are busy from start to end
#   JUDGE_WAIT=0 bash run_pipeline.sh <llm> <seed> judge            # CPU-only: scores, submits the batches, exits
#   bash run_pipeline.sh <llm> <seed> judge                         # CPU-only, later: collects, writes the table
#
# Settings (environment variables, default in brackets):
#   GPUS [the job's GPUs, else 0]  GPU indices, comma-separated: 0,1 = GPUs 0 and 1. Generation runs one process per
#                       GPU; training uses all of them (1, 2, 4 or 8). Under Slurm leave it unset: the job's GPUs.
#   PAIRS [7000]        preference pairs to train on (the paper's 7000)
#   BATCH [64]          pairs per training update (the paper's 64); EPOCHS [1] passes over the pairs (may be fractional).
#                       Updates = PAIRS / BATCH x EPOCHS (7000 pairs: 109). A non-default value is part of the run's
#                       name (cdpo_2000_bs16, cdpo_2000_ep3), so variants sit beside the paper-recipe run and share its
#                       pairs: changing them costs only training and eval, not generation
#   WORK [./work]       venv, model weights, COCO images and runs (~45 GB, plus ~0.2 GB per 1000 images)
#   N_IMAGES [8000]     COCO train2014 images generation may use (at most 16000; raise it if 8000 give too few pairs)
#   IMAGES_FROM [0]     generation skips the first N images of the list (e.g. ones another run already processed);
#                       it then uses positions IMAGES_FROM .. N_IMAGES-1
#   EXCLUDE_IMAGES      file of image ids, one per line, that generation must not use
#   JUDGE_BUDGET [10]   most dollars the gpt-5.1 judge may spend per run (see chair_coco/judge_batch.py); a first eval
#                       on the 1,000 test images judges the base, the model and the matched base, ~$3.5 (Batch API)
#   JUDGE_MODE [batch]  batch = OpenAI Batch API, as in the paper: half price, but the eval waits for the batches
#                       (usually minutes to a few hours, at most 24 h; rerun the command to resume after a stop);
#                       live = concurrent live calls (~10 min per 1,000 captions, twice the price)
#   JUDGE_IMAGE_SOURCE [url]  batch only: url = COCO image URLs (the paper's); base64 = embed the image files
#   JUDGE_WAIT [1]      0 = the judge stage submits the batches and returns at once; run it again later to collect
#   CHUNK               images per GPU per generation round (default: 100 for the first round, which measures the
#                       pairs per image, then 90% of what the remaining pairs need, so generation reloads its models
#                       ~4 times instead of every 100 images and no GPU waits on the others for long)
#   RUN_JUDGE [1]       0 = skip the gpt-5.1 judge (CHAIR, wrong objects, precision still run)
#   MATCHED [1]         also judge the matched-length base (the paper's matched-length results)
#   HEDGE_STRICT [0]    also judge with the hedge-strict prompt (the paper's supplement; ~$2.5 more with the Batch API)
#   LIMIT [0]           evaluate only the first N test images (smoke tests: PAIRS=64 LIMIT=8)
#   BASE_ROLLOUTS, BASE_JUDGED   an existing base arm for the eval, e.g. the paper's (eval_probe.sh)
#   COCO_TRAIN2014_DIR, COCO_VAL2014_DIR, COCO_ANN_DIR   existing COCO image / annotation folders instead of downloading
#   HF_TOKEN            Hugging Face token, for gated models (Gemma 3)
#   TRAINER [native]    native = chair_coco/train_cdpo.py (the authors' recipe reimplemented, with exact context masking);
#                       llamafactory = the authors' own framework: LLaMA-Factory a8caf09 with their three patched files
#                       (llamafactory/src/...) and their qwen_2_5_vl_7B_dpo.sh arguments, in its own venv
#                       (chair_coco/lf/). Run setup with TRAINER=llamafactory once to build it. Adds _lf to the run's
#                       name (cdpo_2000_bs16_lf), so pass the same TRAINER to train, captions and judge.
# The gpt-5.1 judge reads OPENAI_API_KEY from the environment, or OPENAI_API_KEY=... from ./.env (git-ignored).
set -euo pipefail

# ---------------------------------------------------------------------------------------------------------------------
# LLMs: <name> -> weights, a Hugging Face id or a local directory. Edit the right-hand side to swap a model. A new name
# also needs chair_coco/configs/<name>.yaml (copy the closest one: it holds the generation and training settings).
declare -A LLM_PATH=(
  [qwen2_5_vl_3b]="Qwen/Qwen2.5-VL-3B-Instruct"
  [qwen2_5_vl_7b]="Qwen/Qwen2.5-VL-7B-Instruct"   # the authors' Qwen2.5-VL model; their 7k pairs: chair_coco/import_release_pairs.py
  [llava_1_5_7b]="llava-hf/llava-1.5-7b-hf"
  [gemma3_4b]="google/gemma-3-4b-it"
  [gemma4_12b]="google/gemma-4-12B-it"     # needs a GPU with >= 80 GB; runs in its own venvs (LLM_ENV below)
)
# Eval image cap passed to the processor (the probe pipeline's rollout.max_pixels): 1003520 for Qwen, 0 = the
# processor's default, which fixed-resolution models (LLaVA) need.
declare -A LLM_MAX_PIXELS=(
  [qwen2_5_vl_3b]=1003520
  [qwen2_5_vl_7b]=1003520
)
# Hugging Face revisions to pin (the ones the original runs used); a name not listed downloads the current version.
declare -A LLM_REVISION=(
  [qwen2_5_vl_3b]="66285546d2b821cf421d4f5eb2576359d3770cd3"
  [qwen2_5_vl_7b]="cc594898137f460bfe9f0759e9844b3ce807cfb5"   # the snapshot of the probe paper's Qwen2.5-VL-7B runs
  [gemma4_12b]="43c9d014b231474fa7212dbb48a762566c2b255d"   # the snapshot of the probe paper's Gemma-4-12B runs
)
# LLaMA-Factory chat template per model, for TRAINER=llamafactory (the authors' scripts: qwen2_vl for Qwen2-VL and
# Qwen2.5-VL). A model without an entry can only use the default trainer.
# Models whose architecture the main venv (transformers 5.5, vLLM 0.17.1) cannot load get their own pair of venvs,
# chair_coco/envs/<name>/{gen,policy}.lock.txt: "gen" runs generation (vLLM + SENTINEL's tools), "policy" everything
# else (training, test captions, eval). Gemma 4 12B (gemma4_unified) needs transformers >= 5.10: its policy venv is a
# copy of the probe paper's Gemma-4-12B environment (transformers 5.20.2, torch 2.10 / CUDA 12.8), its gen venv vLLM 0.30.0
# with the same transformers (torch 2.13 / CUDA 12.9: NVIDIA driver >= 575). The CUDA build is in <part>.torch-backend.
declare -A LLM_ENV=(
  [gemma4_12b]="gemma4"
)
declare -A LLM_LF_TEMPLATE=(
  [qwen2_5_vl_3b]="qwen2_vl"
  [qwen2_5_vl_7b]="qwen2_vl"
)
# ---------------------------------------------------------------------------------------------------------------------

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CC=$REPO/chair_coco
usage="usage: bash run_pipeline.sh <llm: ${!LLM_PATH[*]}> <seed> [setup profile generate pairs train captions judge | eval]"
LLM=${1:?$usage}; SEED=${2:?$usage}; shift 2
if (( $# )); then STAGES=("$@"); else STAGES=(setup profile generate pairs train captions judge); fi
MODEL=${LLM_PATH[$LLM]:-}
[[ -n $MODEL ]] || { echo "unknown llm '$LLM'. $usage" >&2; exit 2; }
[[ $SEED =~ ^[0-9]+$ ]] || { echo "seed must be an integer. $usage" >&2; exit 2; }
CONFIG=$CC/configs/$LLM.yaml
[[ -f $CONFIG ]] || { echo "missing $CONFIG (training settings for '$LLM')" >&2; exit 2; }
MODEL_NAME=$(basename "${MODEL%/}")

# GPUs: a comma-separated list of GPU indices. Unset: the GPUs this job was given (CUDA_VISIBLE_DEVICES, which Slurm
# sets for a job with --gres=gpu:N), else GPU 0.
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
[[ $GPUS =~ ^[A-Za-z0-9-]+(,[A-Za-z0-9-]+)*$ ]] \
  || { echo "GPUS must be GPU indices separated by commas, e.g. GPUS=0 or GPUS=0,1 (got '$GPUS')" >&2; exit 2; }
IFS=, read -ra GPU_LIST <<< "$GPUS"; NGPU=${#GPU_LIST[@]}
PAIRS=${PAIRS:-7000}; N_IMAGES=${N_IMAGES:-8000}
WORK=$(realpath -m "${WORK:-$REPO/work}")
RUN=$WORK/runs/${LLM}_seed$SEED
PY=$WORK/venv/bin/python
ENV_NAME=${LLM_ENV[$LLM]:-}
if [[ -n $ENV_NAME ]]; then
  PY=$WORK/envs/$ENV_NAME/policy/bin/python       # training, test captions, eval
  GEN_PY=$WORK/envs/$ENV_NAME/gen/bin/python      # generation
else
  GEN_PY=$PY
fi
# TRAINER=llamafactory: LLaMA-Factory at the commit the authors' patched files come from, in its own venv
LF_COMMIT=a8caf09c7fb61d227004450a9c9c2500f2b17a88
LF_SRC=$WORK/llamafactory/LLaMA-Factory-$LF_COMMIT
LF_VENV=$WORK/llamafactory/venv
PROFILE=$CC/flop_profiles/$MODEL_NAME.json
BATCH=${BATCH:-64}; EPOCHS=${EPOCHS:-1}
[[ $BATCH =~ ^[0-9]+$ ]] && (( BATCH >= 1 )) || { echo "BATCH must be a positive integer" >&2; exit 2; }
[[ $EPOCHS =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "EPOCHS must be a positive number" >&2; exit 2; }
OUT=cdpo_$PAIRS
(( BATCH != 64 )) && OUT+=_bs$BATCH
[[ $EPOCHS != 1 && $EPOCHS != 1.0 ]] && OUT+=_ep$EPOCHS
TRAINER=${TRAINER:-native}
case $TRAINER in
  native) ;;
  llamafactory) OUT+=_lf ;;
  *) echo "TRAINER must be native or llamafactory (got '$TRAINER')" >&2; exit 2 ;;
esac
(( N_IMAGES <= 16000 )) || { echo "N_IMAGES is at most 16000 (chair_coco/data/train2014_images.tsv)" >&2; exit 2; }
(( PAIRS >= BATCH )) || { echo "PAIRS must be at least BATCH ($BATCH), one training batch" >&2; exit 2; }

export HF_HOME=$WORK/hf HF_HUB_CACHE=$WORK/hf/hub HF_HUB_OFFLINE=1
export NLTK_DATA=$WORK/nltk_data YOLO_CONFIG_DIR=$WORK/tools/ultralytics_cfg YOLO_OFFLINE=1
ANN=${COCO_ANN_DIR:-$WORK/coco/annotations}
export XDG_CACHE_HOME=$WORK/cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# uv's package cache and its Python live in WORK too (not ~/.cache, ~/.local), so a small home directory is enough
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WORK/uv/cache} UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$WORK/uv/python}
# large wheels (torch, CUDA libraries) over a slow or stalling cluster network: uv's default 30 s read timeout is too short
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
# generation initializes CUDA before vLLM starts its engine process, and a forked process cannot reuse it
# ("Cannot re-initialize CUDA in forked subprocess", seen on two clusters): start vLLM's workers with spawn
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
mkdir -p "$RUN/logs" "$RUN/data" "$WORK/data"
exec > >(tee -a "$RUN/logs/pipeline.log") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
die() { log "ERROR: $*"; exit 1; }
log "SENTINEL pipeline: llm=$LLM ($MODEL) seed=$SEED pairs=$PAIRS gpus=$GPUS stages=${STAGES[*]}"
log "work dir $WORK, run dir $RUN"

# ---------------------------------------------------------------------------------------------------------------------
fetch_images() {   # fetch_images <tsv: image_id file md5> <n> <dest> <url prefix> <local dir or "">
  local tsv=$1 n=$2 dest=$3 url=$4 local_dir=$5 todo
  mkdir -p "$dest"
  # missing or not matching its md5 (a broken download is removed and fetched again)
  todo=$(head -n "$n" "$tsv" | awk -F'\t' '{print $3 "  " $2}' | (cd "$dest" && md5sum -c - 2> /dev/null || true) \
         | grep -v ': OK$' | sed 's/: [A-Z].*$//' || true)
  local n_todo; n_todo=$(printf '%s' "$todo" | grep -c . || true)
  log "  $dest: $((n - n_todo)) / $n present and verified, $n_todo to get"
  if (( n_todo > 0 )); then
    printf '%s\n' "$todo" | (cd "$dest" && xargs -r rm -f)
    if [[ -n $local_dir ]]; then
      printf '%s\n' "$todo" | while read -r f; do ln -sf "$local_dir/$f" "$dest/$f"; done
    else
      printf '%s\n' "$todo" | (cd "$dest" && xargs -P 16 -I{} curl -sf --retry 5 -o {} "$url/{}") &
      local dl=$!
      while kill -0 "$dl" 2> /dev/null; do sleep 30; log "    $(find "$dest" -name '*.jpg' -size +0 | wc -l) / $n"; done
      wait "$dl" || log "  some downloads failed after 5 retries; the md5 check below names them (re-run to retry)"
    fi
  fi
  head -n "$n" "$tsv" | awk -F'\t' '{print $3 "  " $2}' | (cd "$dest" && md5sum --quiet -c -) \
    || die "md5 check failed in $dest (re-run setup to fetch missing or broken files)"
  log "  md5 ok"
}

stage_setup() {
  [[ -n $ENV_NAME ]] || log "[setup] python environment -> $WORK/venv (exact versions: chair_coco/requirements.lock.txt)"
  # uv's installer puts it in ~/.local/bin (older versions: ~/.cargo/bin), which non-interactive shells and batch
  # jobs often do not have on PATH
  command -v uv > /dev/null || export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
  command -v uv > /dev/null || die "uv is needed: curl -LsSf https://astral.sh/uv/install.sh | sh  (then re-run)"
  log "  uv $(uv --version | cut -d' ' -f2) at $(command -v uv)"
  command -v nvidia-smi > /dev/null || log "WARNING: nvidia-smi not found; generation, training and eval need NVIDIA GPUs"
  if [[ -n $ENV_NAME ]]; then
    local part
    for part in gen policy; do
      log "[setup] $LLM venv '$part' -> $WORK/envs/$ENV_NAME/$part (chair_coco/envs/$ENV_NAME/$part.lock.txt)"
      [[ -x $WORK/envs/$ENV_NAME/$part/bin/python ]] || uv venv --python 3.12 "$WORK/envs/$ENV_NAME/$part"
      uv pip sync --python "$WORK/envs/$ENV_NAME/$part/bin/python" \
        --torch-backend "$(cat "$CC/envs/$ENV_NAME/$part.torch-backend" 2> /dev/null || echo cu128)" "$CC/envs/$ENV_NAME/$part.lock.txt"
    done
  else
    [[ -x $PY ]] || uv venv --python 3.12 "$WORK/venv"
    uv pip sync --python "$PY" --torch-backend cu128 "$CC/requirements.lock.txt"
  fi
  HF_HUB_OFFLINE=0 "$PY" -c "import nltk; [nltk.download(p, download_dir='$NLTK_DATA', quiet=True) for p in ('punkt_tab', 'wordnet', 'averaged_perceptron_tagger_eng')]"

  log "[setup] model weights -> $HF_HUB_CACHE"
  HF_HUB_OFFLINE=0 "$PY" - "$MODEL" "${LLM_REVISION[$LLM]:-}" <<'PYEOF'
import os, sys, time
from huggingface_hub import snapshot_download
model, model_rev = sys.argv[1], sys.argv[2] or None
repos = [("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"),   # SENTINEL's detector
         ("lizhuang144/flan-t5-base-VG-factual-sg", "b7923dae0e6f924fe37957a1cee920dac1d50a7f"),  # scene-graph parser
         ("biu-nlp/lingmess-coref", "fa5d8a827a09388d03adbe9e800c7d8c509c3935")]  # coreference (fastcoref)
if not os.path.isdir(model):
    repos.insert(0, (model, model_rev))
for repo, rev in repos:
    t = time.time()
    snapshot_download(repo, revision=rev)
    if rev:   # a download by commit hash writes no refs/main, which offline loading by repo id needs
        ref = os.path.join(os.environ["HF_HUB_CACHE"], "models--" + repo.replace("/", "--"), "refs", "main")
        os.makedirs(os.path.dirname(ref), exist_ok=True)
        open(ref, "w").write(rev)
    print(f"  {repo} ok ({time.time() - t:.0f} s)", flush=True)
PYEOF
  local yolo=$WORK/tools/yolo/yolo11x.pt
  mkdir -p "$(dirname "$yolo")" "$YOLO_CONFIG_DIR"
  [[ -s $yolo ]] || curl -sfL --retry 5 -o "$yolo" https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11x.pt
  echo "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24  $yolo" | sha256sum --quiet -c - \
    || die "yolo11x.pt checksum mismatch"
  log "  yolo11x ok"

  log "[setup] COCO images (train2014: the first $N_IMAGES of chair_coco/data/train2014_images.tsv; val2014: the 1,000 v2 test images)"
  fetch_images "$CC/data/train2014_images.tsv" "$N_IMAGES" "$WORK/coco/train2014" http://images.cocodataset.org/train2014 "${COCO_TRAIN2014_DIR:-}"
  fetch_images "$CC/data/val2014_test_images.tsv" 1000 "$WORK/coco/val2014" http://images.cocodataset.org/val2014 "${COCO_VAL2014_DIR:-}"
  log "[setup] COCO val2014 annotations (instances, captions) for the reference CHAIR -> $ANN"
  if [[ -n ${COCO_ANN_DIR:-} ]]; then
    [[ -f $COCO_ANN_DIR/instances_val2014.json && -f $COCO_ANN_DIR/captions_val2014.json ]] || die "no val2014 annotations in $COCO_ANN_DIR"
  elif [[ ! -f $ANN/instances_val2014.json || ! -f $ANN/captions_val2014.json ]]; then
    mkdir -p "$ANN"
    curl -sfL --retry 5 -o "$ANN/annotations_trainval2014.zip" http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    "$PY" -c "import zipfile, os; z = zipfile.ZipFile('$ANN/annotations_trainval2014.zip'); \
[open(os.path.join('$ANN', os.path.basename(n)), 'wb').write(z.read(n)) for n in ('annotations/instances_val2014.json', 'annotations/captions_val2014.json')]"
    rm -f "$ANN/annotations_trainval2014.zip"
  fi
  "$PY" - "$CC" "$WORK" <<'PYEOF'
import json, os, sys
cc, work = sys.argv[1:]
with open(f"{work}/data/train2014_images.jsonl", "w") as f:
    for line in open(f"{cc}/data/train2014_images.tsv"):
        image_id, name, _ = line.rstrip("\n").split("\t")
        f.write(json.dumps({"image_id": image_id, "image_path": f"{work}/coco/train2014/{name}",
                            "question": "Describe this image in detail."}) + "\n")
rows = json.load(open(f"{cc}/data/coco_v2/rl_test.json"))   # the paper's RL-test split, 1,000 val2014 images
for r in rows:
    r["path"] = f"{work}/coco/val2014/{r['file_name']}"
json.dump(rows, open(f"{work}/data/rl_test.json", "w"))
PYEOF

  [[ $TRAINER == llamafactory ]] && setup_llamafactory
  log "[setup] checks"
  "$GEN_PY" - "$MODEL" <<'PYEOF'
import os, sys
import fastcoref, nltk, spacy, supervision, torch, transformers, ultralytics, vllm
from huggingface_hub import try_to_load_from_cache
spacy.load("en_core_web_md"); nltk.data.find("tokenizers/punkt_tab"); nltk.data.find("corpora/wordnet.zip")
nltk.data.find("taggers/averaged_perceptron_tagger_eng")
model = sys.argv[1]
for repo in ([] if os.path.isdir(model) else [model]) + ["IDEA-Research/grounding-dino-base",
                                                          "lizhuang144/flan-t5-base-VG-factual-sg", "biu-nlp/lingmess-coref"]:
    assert isinstance(try_to_load_from_cache(repo, "config.json"), str), f"{repo} does not resolve offline"
print(f"  ok: torch {torch.__version__} (CUDA available: {torch.cuda.is_available()}), transformers {transformers.__version__}, "
      f"vllm {vllm.__version__}; models resolve offline")
PYEOF
  if [[ -n $ENV_NAME ]]; then   # the policy venv: training and eval packages, and the model's config and processor
    "$PY" - "$MODEL" <<'PYEOF'
import sys
import accelerate, nltk, openai, peft, torch, transformers
from transformers import AutoConfig, AutoProcessor
AutoConfig.from_pretrained(sys.argv[1]); AutoProcessor.from_pretrained(sys.argv[1])
print(f"  ok (policy venv): torch {torch.__version__}, transformers {transformers.__version__}, peft {peft.__version__}, "
      f"accelerate {accelerate.__version__}; {sys.argv[1]} config and processor load offline")
PYEOF
  fi
}

setup_llamafactory() {
  log "[setup] TRAINER=llamafactory: LLaMA-Factory $LF_COMMIT + the authors' patched files -> $LF_SRC"
  if [[ ! -f $LF_SRC/src/train.py ]]; then
    mkdir -p "$WORK/llamafactory"
    curl -sfL --retry 5 -o "$WORK/llamafactory/src.tar.gz" "https://github.com/hiyouga/LLaMA-Factory/archive/$LF_COMMIT.tar.gz"
    mkdir -p "$LF_SRC"   # the archive's top folder follows the repo's current name (LlamaFactory-<sha>); strip it
    tar xzf "$WORK/llamafactory/src.tar.gz" -C "$LF_SRC" --strip-components=1
    rm -f "$WORK/llamafactory/src.tar.gz"
  fi
  local f
  for f in converter.py parser.py processor/pairwise.py; do   # the three files the authors changed (README: llamafactory/)
    cp "$REPO/llamafactory/src/llamafactory/data/$f" "$LF_SRC/src/llamafactory/data/$f"
  done
  log "[setup] LLaMA-Factory venv -> $LF_VENV (chair_coco/lf/requirements.lock.txt, DeepSpeed 0.15.4)"
  [[ -x $LF_VENV/bin/python ]] || uv venv --python 3.11 "$LF_VENV"
  uv pip sync --python "$LF_VENV/bin/python" --torch-backend cu128 "$CC/lf/requirements.lock.txt"
  uv pip install --python "$LF_VENV/bin/python" --no-deps -e "$LF_SRC"
  # DeepSpeed's build imports deepspeed, which imports Triton when present, and Triton >= 3 raises without a GPU driver
  # (a CPU-only setup job). Hide Triton from the build only: it compiles no ops (DS_BUILD_OPS=0), so the package is the
  # same; at training time, on a GPU, Triton loads as usual.
  local no_triton=$WORK/llamafactory/build_without_triton
  mkdir -p "$no_triton/triton"
  echo 'raise ImportError("Triton hidden while building DeepSpeed (run_pipeline.sh setup)")' > "$no_triton/triton/__init__.py"
  DS_BUILD_OPS=0 PYTHONPATH=$no_triton uv pip install --python "$LF_VENV/bin/python" --no-deps --no-build-isolation deepspeed==0.15.4
  # On a CPU-only node, importing LLaMA-Factory imports DeepSpeed (via transformers), which imports Triton, which needs a
  # GPU driver: check with Triton hidden there (training runs on a GPU, where it loads normally)
  local check_path=
  "$LF_VENV/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2> /dev/null \
    || check_path=$no_triton
  PYTHONPATH=$check_path "$LF_VENV/bin/python" - <<'PYEOF'
import importlib.metadata, inspect
import peft, torch, transformers, trl
from llamafactory.data.processor import pairwise
from llamafactory.extras.misc import check_dependencies
import llamafactory.train.tuner  # the whole training stack (trainers, datasets, multimodal plugins)
check_dependencies()
assert "mark_context" in inspect.getsource(pairwise), "the authors' pairwise.py is not the one installed"
import deepspeed
ds = f"deepspeed {deepspeed.__version__}" + ("" if torch.cuda.is_available() else " (checked without Triton: no GPU here)")
print(f"  ok: LLaMA-Factory with the authors' C-DPO patch; torch {torch.__version__}, transformers {transformers.__version__}, "
      f"peft {peft.__version__}, trl {trl.__version__}, {ds}")
PYEOF
}

stage_profile() {
  if [[ -f $PROFILE ]]; then log "[profile] $PROFILE exists"; return; fi
  log "[profile] measuring $MODEL's FLOP cost profile on CPU (fp32; minutes, ~4x the model size in RAM) -> $PROFILE"
  CUDA_VISIBLE_DEVICES= "$PY" "$CC/flop_profile.py" --model "$MODEL" --image-dir "$WORK/coco/train2014" --out "$PROFILE"
}

COMMON=()
common_args() { COMMON=(--config "$CONFIG" --model "$MODEL" --run-dir "$RUN"); }

pair_status() {   # "<pairs in the finished prefix> <finished prefix> <manifest size>"
  "$PY" "$CC/build_pairs.py" "${COMMON[@]}" --max-pairs 0 --out-file status_pairs.jsonl > /dev/null
  "$PY" -c "import json; m = json.load(open('$RUN/data/status_pairs_meta.json')); print(m['kept'], m['finished_prefix'], sum(1 for _ in open('$RUN/data/images.jsonl')))"
}

stage_generate() {
  [[ -f $PROFILE ]] || die "no FLOP profile; run the profile stage first"
  # the images this run may use: list positions IMAGES_FROM .. N_IMAGES-1, minus the ids in EXCLUDE_IMAGES
  "$PY" - "$WORK/data/train2014_images.jsonl" "$RUN/data/images.jsonl" "$N_IMAGES" "${IMAGES_FROM:-0}" "${EXCLUDE_IMAGES:-}" <<'PYEOF'
import json, sys
src, dst, n, start, exclude = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
skip = {line.strip() for line in open(exclude) if line.strip()} if exclude else set()
rows = [json.loads(line) for line in open(src)][start:n]
keep = [r for r in rows if str(r["image_id"]) not in skip]
with open(dst, "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in keep)
if start or skip:
    print(f"  images: list positions {start}..{n - 1}, {len(rows) - len(keep)} excluded by id -> {len(keep)} usable")
PYEOF
  [[ -s $RUN/data/images.jsonl ]] || die "no usable images (N_IMAGES=$N_IMAGES, IMAGES_FROM=${IMAGES_FROM:-0})"
  common_args
  local chunk kept prefix total last=-1
  while :; do
    read -r kept prefix total < <(pair_status)
    log "[generate] $kept pairs in the first $prefix / $total images (target $PAIRS)"
    (( kept >= PAIRS )) && return
    (( prefix < total )) || die "all $total images used and only $kept pairs: re-run with a larger N_IMAGES (up to 16000), setup first"
    (( prefix > last )) || die "the last round finished no image; see $RUN/logs/generate_shard*.log"
    last=$prefix
    # Each round starts every model afresh on every GPU and ends when the slowest GPU finishes, so few large rounds
    # keep the GPUs busy. The first (100 images per GPU) measures the pairs per image; the next ones take 90% of the
    # images the remaining pairs need (at least 25 per GPU), so the overshoot past PAIRS stays small.
    if [[ -n ${CHUNK:-} ]]; then chunk=$CHUNK
    elif (( kept == 0 || prefix < 50 )); then chunk=100
    else chunk=$(awk -v p="$PAIRS" -v k="$kept" -v f="$prefix" -v n="$NGPU" \
                   'BEGIN { c = int(0.9 * (p - k) / (k / f) / n + 0.999); print (c < 25 ? 25 : c) }')
    fi
    log "[generate] round: up to $chunk images on each of $NGPU GPU(s)"
    local pids=() i
    for i in "${!GPU_LIST[@]}"; do
      CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} "$GEN_PY" "$CC/generate.py" "${COMMON[@]}" --flop-profile "$PROFILE" --seed "$SEED" \
        --hf-cache "$HF_HUB_CACHE" --tools-dir "$WORK/tools" --shard "$i" --num-shards "$NGPU" --num-images "$chunk" \
        >> "$RUN/logs/generate_shard$i.log" 2>&1 &
      pids+=($!)
    done
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || die "generation shard $i failed: see $RUN/logs/generate_shard$i.log"
    done
  done
}

stage_pairs() {
  common_args
  "$PY" "$CC/build_pairs.py" "${COMMON[@]}" --max-pairs "$PAIRS" --out-file "train_pairs_$PAIRS.jsonl"
  local kept; kept=$("$PY" -c "import json; print(json.load(open('$RUN/data/train_pairs_${PAIRS}_meta.json'))['kept'])")
  (( kept == PAIRS )) || die "only $kept of $PAIRS pairs: run the generate stage (with a larger N_IMAGES if needed)"
}

stage_train() {
  if [[ -f $RUN/train/$OUT/checkpoint-final/adapter_model.safetensors ]]; then log "[train] $RUN/train/$OUT exists"; return; fi
  (( BATCH % NGPU == 0 )) || die "the batch of $BATCH pairs must split evenly over $NGPU GPUs (GPUS=$GPUS)"
  [[ -f $RUN/data/train_pairs_$PAIRS.jsonl ]] || die "no $RUN/data/train_pairs_$PAIRS.jsonl: run generate and pairs first"
  if [[ $TRAINER == llamafactory ]]; then train_llamafactory; return; fi
  common_args
  local launch=("$(dirname "$PY")/accelerate" launch --num_processes "$NGPU" --mixed_precision no)
  (( NGPU > 1 )) && launch+=(--multi_gpu)
  log "[train] C-DPO on $NGPU GPU(s), $PAIRS pairs, batch $BATCH, $EPOCHS epoch(s) -> $RUN/train/$OUT"
  CUDA_VISIBLE_DEVICES=$GPUS "${launch[@]}" "$CC/train_cdpo.py" "${COMMON[@]}" --flop-profile "$PROFILE" --seed "$SEED" \
    --train-file "train_pairs_$PAIRS.jsonl" --out-name "$OUT" \
    --total-batch-size "$BATCH" --epochs "$EPOCHS" 2>&1 | tee -a "$RUN/logs/train_$OUT.log"
  [[ -f $RUN/train/$OUT/checkpoint-final/adapter_model.safetensors ]] || die "training failed: see $RUN/logs/train_$OUT.log"
}

# The authors' training: llamafactory/examples/SENTINEL/Qwen_2_5_VL/qwen_2_5_vl_7B_dpo.sh, argument for argument.
# Differences: the model and the dataset (ours); gradient accumulation = BATCH / GPUs (theirs: 16 x 4 GPUs = 64);
# lr and epochs from the config / EPOCHS; --seed SEED (their script leaves LLaMA-Factory's default, 42); processes are
# started by torchrun instead of the deepspeed launcher (same DeepSpeed ZeRO-2 config; torchrun honours
# CUDA_VISIBLE_DEVICES as Slurm sets it); --report_to none. --flash_attn fa2 as in their script: LLaMA-Factory falls back
# to PyTorch SDPA, with a warning, when flash-attn is not installed (the same attention, computed by another kernel).
train_llamafactory() {
  local tmpl=${LLM_LF_TEMPLATE[$LLM]:-}
  [[ -n $tmpl ]] || die "TRAINER=llamafactory has no LLaMA-Factory template for $LLM (LLM_LF_TEMPLATE)"
  [[ -x $LF_VENV/bin/python && -f $LF_SRC/src/train.py ]] || die "no LLaMA-Factory venv: run setup with TRAINER=llamafactory first"
  local T=$RUN/train/$OUT lr port
  lr=$("$PY" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['train']['lr'])")
  port=$("$PY" -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
  if [[ -f $T/lf_output/adapter_model.safetensors && -f $T/lf_output/all_results.json ]]; then
    log "[train] LLaMA-Factory training already finished ($T/lf_output); converting its adapter only"
  else
  "$PY" "$CC/lf/convert_pairs.py" --pairs "$RUN/data/train_pairs_$PAIRS.jsonl" --out-dir "$T/lf_data" --name sentinel
  log "[train] LLaMA-Factory C-DPO (the authors' framework) on $NGPU GPU(s), $PAIRS pairs, batch $BATCH ="
  log "        $NGPU x 1 x $((BATCH / NGPU)), $EPOCHS epoch(s), lr $lr -> $T/lf_output"
  ( cd "$LF_SRC" && CUDA_VISIBLE_DEVICES=$GPUS "$LF_VENV/bin/torchrun" --nproc_per_node "$NGPU" --master_port "$port" \
      src/train.py \
      --deepspeed "$CC/lf/ds_z2_config.json" \
      --model_name_or_path "$MODEL" \
      --stage dpo \
      --flash_attn fa2 \
      --do_train \
      --finetuning_type lora \
      --lora_rank 128 \
      --lora_target all \
      --pref_beta 0.1 \
      --pref_loss sigmoid \
      --dataset sentinel \
      --dataset_dir "$T/lf_data" \
      --template "$tmpl" \
      --output_dir "$T/lf_output" \
      --overwrite_cache \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps $((BATCH / NGPU)) \
      --lr_scheduler_type cosine \
      --logging_steps 1 \
      --save_steps 1000 \
      --learning_rate "$lr" \
      --num_train_epochs "$EPOCHS" \
      --plot_loss \
      --bf16 \
      --warmup_ratio 0 \
      --ddp_timeout 180000000 \
      --cutoff_len 4096 \
      --max_samples 1000000 \
      --overwrite_output_dir \
      --preprocessing_num_workers 16 \
      --seed "$SEED" \
      --report_to none ) 2>&1 | tee -a "$RUN/logs/train_$OUT.log"
  [[ -f $T/lf_output/adapter_model.safetensors ]] || die "LLaMA-Factory training failed: see $RUN/logs/train_$OUT.log"
  fi
  log "[train] adapter -> the eval's transformers naming, every tensor checked -> $T/checkpoint-final"
  CUDA_VISIBLE_DEVICES= "$PY" "$CC/lf/convert_adapter.py" --base-model "$MODEL" --adapter "$T/lf_output" \
    --out "$T/checkpoint-final" 2>&1 | tee -a "$RUN/logs/train_$OUT.log"
  [[ -f $T/checkpoint-final/adapter_model.safetensors ]] || die "adapter conversion failed: see $RUN/logs/train_$OUT.log"
  log "[train] note: LLaMA-Factory training is not FLOP-counted (generation's count is in $RUN/flops/)"
}

eval_probe() {   # eval_probe <steps> [env ...]: the paper's eval (chair_coco/eval_probe.sh) on this run's adapter
  local steps=$1; shift
  [[ -f $RUN/train/$OUT/checkpoint-final/adapter_model.safetensors ]] || die "no trained adapter in $RUN/train/$OUT: run train first"
  env RUN_DIR="$RUN" WORK="$WORK" BASE_MODEL="$MODEL" GPU="$GPUS" PY="$PY" MAX_PIXELS="${LLM_MAX_PIXELS[$LLM]:-0}" \
    JUDGE_BUDGET="${JUDGE_BUDGET:-10}" JUDGE_MODE="${JUDGE_MODE:-batch}" JUDGE_IMAGE_SOURCE="${JUDGE_IMAGE_SOURCE:-url}" \
    JUDGE_WAIT="${JUDGE_WAIT:-1}" RUN_JUDGE="${RUN_JUDGE:-1}" MATCHED="${MATCHED:-1}" HEDGE_STRICT="${HEDGE_STRICT:-0}" \
    COCO_ANN_DIR="$ANN" STEPS="$steps" "$@" bash "$CC/eval_probe.sh" "$OUT" --adapter "$RUN/train/$OUT/checkpoint-final"
}

stage_captions() {
  log "[captions] greedy eval captions of the base model and $OUT on GPU(s) $GPUS"
  eval_probe captions
}

JUDGE_PENDING=0
stage_judge() {
  if [[ ${RUN_JUDGE:-1} == 1 && -z ${OPENAI_API_KEY:-} ]] && ! grep -qs "OPEN_\?AI_API_KEY" "$REPO/.env"; then
    die "the gpt-5.1 judge needs OPENAI_API_KEY (environment or $REPO/.env); or set RUN_JUDGE=0"
  fi
  local mode=${JUDGE_MODE:-batch}
  if [[ $mode == batch && ${JUDGE_WAIT:-1} == 0 ]]; then mode="batch, submit only"; fi
  log "[judge] CHAIR, matcher metrics, gpt-5.1 judge ($mode) and the Table 1 row of $OUT; no GPU used"
  local rc=0
  eval_probe score,judge CUDA_VISIBLE_DEVICES= || rc=$?
  if (( rc == 5 )); then
    JUDGE_PENDING=1
    log "[judge] the batches are running at OpenAI; nothing here needs to stay up. Collect them later (no GPU):"
    log "  bash run_pipeline.sh $LLM $SEED judge"
  elif (( rc != 0 )); then
    die "eval failed (exit $rc); see the output above"
  fi
}

stage_eval() { stage_captions; stage_judge; }

for stage in "${STAGES[@]}"; do
  case $stage in
    setup|profile|generate|pairs|train|captions|judge|eval) "stage_$stage" ;;
    *) die "unknown stage '$stage'. $usage" ;;
  esac
done
log "done: ${STAGES[*]}"
if [[ " ${STAGES[*]} " == *" captions "* && " ${STAGES[*]} " != *" judge "* ]]; then
  log "GPU work done. The rest needs no GPU: bash run_pipeline.sh $LLM $SEED judge"
fi
if [[ -f $RUN/eval/$OUT/table.md && ( " ${STAGES[*]} " == *" judge "* || " ${STAGES[*]} " == *" eval "* ) ]] && (( ! JUDGE_PENDING )); then
  log "results: $RUN/eval/$OUT/table.md (this seed's Table 1 row); pool seeds with:"
  log "  $PY $CC/eval/paper_table.py --name SENTINEL $WORK/runs/${LLM}_seed*/eval/$OUT"
  log "FLOPs:   $RUN/train/$OUT/flops_pipeline.json"
fi
