#!/usr/bin/env bash
# The paper's eval (Table 1, COCO RL-test v2 split: 1,000 val2014 images), for a LoRA adapter (SENTINEL) or a full model
# directory. It follows the probe repo's v2 pipeline step for step -- run.py's eval:rollout / eval:judge / eval:compare
# and store/post_judge_run.sh's analysis -- with byte-identical copies of its scripts in eval/ (eval/PROVENANCE.md):
#   1. greedy captions, HF generate() (eval/rollout_acts.py --greedy --num_rollouts 1 --no-acts, 512 new tokens,
#      "Describe this image in detail.", max_pixels per model) for the base model and the arm
#   2. eval/truncation_null.py: wrong COCO objects per caption, pooled precision and tokens (synonym matcher on the
#      segmentation labels), and the base captions cut to the arm's token counts (matched length)
#   3. eval/ref_chair.py: CHAIR_S, CHAIR_I and recall with the reference CHAIR code (OPERA's chair.py; ground truth =
#      segmentation labels + objects named in the human COCO captions), full length and matched length
#   4. gpt-5.1 judge (eval/judge_v3.txt, detail high) of the base, the arm and the matched base; HEDGE_STRICT=1 also
#      judges the arm and the matched base with the hedge-strict prompt (eval/judge_v3_nohedge.txt). Through the OpenAI
#      Batch API, as the paper did (judge_batch.py: one campaign for all of them, images by COCO URL)
#   5. eval/compare_greedy_arms.py (10000 bootstrap resamples, seed 42): judged_compare_full.json and _matched.json
#   6. eval/paper_table.py: the Table 1 row of this run -> table.md (pool seeds with paper_table.py <dir> <dir> ...)
#
# Step 1 needs the GPU(s); steps 2-6 run on CPU and, with the Batch API, mostly wait. STEPS selects them, so a GPU job
# can stop after the captions: STEPS=captions on the GPU, then STEPS=score,judge anywhere with the same files (no GPU).
#
# usage: eval_probe.sh NAME --adapter DIR | --model DIR       -> $RUN_DIR/eval/NAME/
# env:   RUN_DIR (required)  WORK (setup's work dir: data/rl_test.json, coco/annotations, base_eval/)
#        BASE_MODEL (Qwen/Qwen2.5-VL-3B-Instruct)  MAX_PIXELS (1003520; 0 = processor default, LLaVA)  GPU (0, or 0,1)
#        MANIFEST, COCO_ANN_DIR (default from WORK)  BASE_ROLLOUTS, BASE_JUDGED (reuse a base arm, e.g. the paper's)
#        MATCHED (1)  HEDGE_STRICT (0)  RUN_JUDGE (1)  LIMIT (0; N = first N images, smoke tests only)
#        JUDGE_BUDGET (10 dollars)  PY (python)
#        JUDGE_MODE (batch: the Batch API, half price, results in minutes to hours, at most 24 h; live: concurrent live
#                    calls through judge_capped.py, full price, ~10 min per 1,000 captions)
#        JUDGE_IMAGE_SOURCE (url, the paper's: http://images.cocodataset.org; base64: embed the files)
#        STEPS (captions,score,judge: 1 = captions [GPU]; 2-3 = score [CPU]; 4-6 = judge [CPU, network])
#        JUDGE_WAIT (1; 0 = submit the batches and exit with code 5 instead of waiting; run again to collect them)
# Cost with the Batch API: ~$1.2 per 1,000 judged captions (live: ~$2.4). A first eval judges the base, the arm and the
# matched base (~$3.5; live ~$7); later arms of the same base model ~$2.3 (live ~$4.5).
# Both judges check the spend in $RUN_DIR/eval/judge_spend.jsonl against JUDGE_BUDGET before paying (the batch judge
# books a high estimate before it submits, and the real cost when it collects); on stopping, the script exits before
# the comparisons, so no table comes from a partly judged arm. Stopped or interrupted: rerun the same command; the
# batch judge picks up its running batches from $OUT/judge_batch/state.json.
# The API key: OPENAI_API_KEY in the environment, or OPENAI_API_KEY=... in a .env file at the repository root.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
E=$HERE/eval
PY="${PY:-python}"
NAME="${1:?usage: $0 NAME (--adapter DIR | --model DIR)}"
: "${RUN_DIR:?set RUN_DIR}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
case "${2:-}" in
  --adapter) ARM=(--model "$BASE_MODEL" --adapter "$(realpath "${3:?adapter dir}")") ;;
  --model)   ARM=(--model "$(realpath "${3:?model dir}")") ;;
  *) echo "usage: $0 NAME (--adapter DIR | --model DIR)" >&2; exit 2 ;;
esac
GPU="${GPU:-0}"; LIMIT="${LIMIT:-0}"; MAX_PIXELS="${MAX_PIXELS:-1003520}"
MANIFEST=${MANIFEST:-${WORK:?set WORK or MANIFEST}/data/rl_test.json}
ANN=${COCO_ANN_DIR:-${WORK:-}/coco/annotations}
for f in "$MANIFEST" "$ANN/instances_val2014.json" "$ANN/captions_val2014.json"; do
  [[ -f $f ]] || { echo "missing $f: run setup first" >&2; exit 1; }
done
"$PY" -c "import json, os, sys; r = json.load(open('$MANIFEST')); \
sys.exit(0 if all(os.path.isabs(x['path']) and os.path.exists(x['path']) for x in r) else 'manifest image paths must be absolute and exist')"
N_IMAGES=$("$PY" -c "import json; print(len(json.load(open('$MANIFEST'))))")
(( LIMIT > 0 )) && (( LIMIT < N_IMAGES )) && N_IMAGES=$LIMIT
OUT=$RUN_DIR/eval/$NAME
B=${WORK:-}/base_eval/$(basename "${BASE_MODEL%/}")
BASE_ROLLOUTS=${BASE_ROLLOUTS:-$B/rl_test_base}
BASE_JUDGED=${BASE_JUDGED:-$B/judged_v3}
mkdir -p "$OUT"
log() { echo "[eval $(date '+%H:%M:%S')] $*"; }
count() { find "$1" -maxdepth 1 -name '[0-9]*.json' 2> /dev/null | wc -l; }

# run.py rollout_cmd() for a greedy, activation-free dataset
ROLLOUT=(env PYTHON="$PY" bash "$E/launch_rollout_acts.sh" "$GPU" --manifest "$MANIFEST" --num_rollouts 1 --greedy --no-acts
         --max-new-tokens 512 --max-pixels "$MAX_PIXELS" --limit "$LIMIT")
CMP=("$PY" "$E/compare_greedy_arms.py" --manifest "$MANIFEST" --tokenizer "$BASE_MODEL" --bootstrap 10000 --seed 42)

STEPS=",${STEPS:-captions,score,judge},"
step() { [[ $STEPS == *",$1,"* ]]; }
captions_complete() {
  for d in "$BASE_ROLLOUTS" "$OUT/post_greedy"; do
    (( $(count "$d") >= N_IMAGES )) || { echo "$d: $(count "$d") captions, expected $N_IMAGES (run the captions step)" >&2; exit 1; }
  done
}

# 1. greedy captions (resumable: images with a complete .json are skipped)
if step captions; then
  log "captions, base model ($BASE_MODEL) -> $BASE_ROLLOUTS"
  "${ROLLOUT[@]}" --model "$BASE_MODEL" --rundir "$BASE_ROLLOUTS"
  log "captions, $NAME -> $OUT/post_greedy"
  "${ROLLOUT[@]}" "${ARM[@]}" --rundir "$OUT/post_greedy"
  captions_complete
  log "captions done"
fi

# 2-3 (CPU). Done already for these captions (a rerun to collect batches): skipped.
if step score; then
  captions_complete
  if [[ -s $OUT/truncation_null.json && $OUT/truncation_null.json -nt $OUT/post_greedy \
        && ( LIMIT -gt 0 || -s $OUT/ref_chair.json ) ]]; then
    log "truncation null and reference CHAIR: done for these captions, skipped"
  else
    # 2. truncation null: judge-free object metrics, full and matched length, and the matched base captions
    log "truncation null (wrong objects, precision, tokens; matched base) -> $OUT/truncation_null.json"
    "$PY" "$E/truncation_null.py" --base-rollouts "$BASE_ROLLOUTS" --rl-rollouts "$OUT/post_greedy" --manifest "$MANIFEST" \
      --tokenizer "$BASE_MODEL" --label "$NAME" --emit-rollouts "$OUT/base_trunc_token" --json "$OUT/truncation_null.json" \
      > "$OUT/truncation_null.txt"

    # 3. reference CHAIR (needs every test image captioned, so not with LIMIT)
    if (( LIMIT == 0 )); then
      log "reference CHAIR (CHAIR_S, CHAIR_I, recall) -> $OUT/ref_chair.json"
      "$PY" "$E/ref_chair.py" --chair "$E/reference_chair_opera.py" --ann "$ANN" --manifest "$MANIFEST" \
        --base "$BASE_ROLLOUTS" --arm "$NAME=$OUT/post_greedy,$OUT/base_trunc_token" --workdir "$OUT/ref_chair_work" \
        --out "$OUT/ref_chair.json" > "$OUT/ref_chair.txt"
      tail -3 "$OUT/ref_chair.txt"
    else
      log "reference CHAIR skipped: LIMIT=$LIMIT (it scores all test images)"
    fi
  fi
fi

# 4-5. gpt-5.1 judge (run.py's judge settings: gpt-5.1, detail high, judge_v3.txt; store/post_judge_run.sh judged
# through judge_batch.py with --image-source url) and the judged comparisons of store/post_judge_run.sh
if step judge && [[ "${RUN_JUDGE:-1}" == 1 ]]; then
  captions_complete
  [[ -s $OUT/truncation_null.json ]] || { echo "no $OUT/truncation_null.json: run the score step first" >&2; exit 1; }
  NOHEDGE=$E/judge_v3_nohedge.txt
  TARGETS=("$BASE_ROLLOUTS,$BASE_JUDGED" "$OUT/post_greedy,$OUT/judged_v3")   # captions,judgments[,system prompt]
  [[ "${MATCHED:-1}" == 1 ]] && TARGETS+=("$OUT/base_trunc_token,$OUT/base_trunc_token_judged_v3")
  [[ "${HEDGE_STRICT:-0}" == 1 ]] && TARGETS+=("$OUT/post_greedy,$OUT/judged_v3_nohedge,$NOHEDGE"
                                             "$OUT/base_trunc_token,$OUT/base_trunc_token_judged_v3_nohedge,$NOHEDGE")
  BUDGET=(--budget "${JUDGE_BUDGET:-10}" --ledger "$RUN_DIR/eval/judge_spend.jsonl")
  case "${JUDGE_MODE:-batch}" in
    batch)
      log "gpt-5.1 judge, Batch API: base, $NAME$([[ ${MATCHED:-1} == 1 ]] && echo ', matched base')$([[ ${HEDGE_STRICT:-0} == 1 ]] && echo ', hedge-strict') (already judged captions are skipped)"
      TFLAGS=(); for t in "${TARGETS[@]}"; do TFLAGS+=(--target "$t"); done
      rc=0
      "$PY" "$HERE/judge_batch.py" --workdir "$OUT/judge_batch" "${BUDGET[@]}" "${TFLAGS[@]}" \
        --image-source "${JUDGE_IMAGE_SOURCE:-url}" --limit-images "$LIMIT" \
        $([[ ${JUDGE_WAIT:-1} == 0 ]] && echo --no-wait) || rc=$?
      if (( rc == 5 )); then
        log "batches submitted and running; run the judge step again later to collect them (no GPU needed)"
        exit 5
      fi
      (( rc == 0 )) || exit "$rc" ;;
    live)
      for t in "${TARGETS[@]}"; do
        IFS=, read -r src dst sys <<< "$t"
        log "gpt-5.1 judge, live: $src -> $dst"
        "$PY" "$HERE/judge_capped.py" "${BUDGET[@]}" --rundir "$src" --outdir "$dst" ${sys:+--sys-file "$sys"}
      done ;;
    *) echo "JUDGE_MODE must be batch or live" >&2; exit 2 ;;
  esac
  log "judged comparison, full length -> $OUT/judged_compare_full.json"
  "${CMP[@]}" --pre-rollouts "$BASE_ROLLOUTS" --pre-judged "$BASE_JUDGED" --post-rollouts "$OUT/post_greedy" \
    --post-judged "$OUT/judged_v3" --out "$OUT/judged_compare_full.json" > "$OUT/judged_compare_full.txt"
  if [[ "${MATCHED:-1}" == 1 ]]; then
    log "judged comparison, matched length -> $OUT/judged_compare_matched.json"
    "${CMP[@]}" --pre-rollouts "$OUT/base_trunc_token" --pre-judged "$OUT/base_trunc_token_judged_v3" \
      --post-rollouts "$OUT/post_greedy" --post-judged "$OUT/judged_v3" --out "$OUT/judged_compare_matched.json" \
      > "$OUT/judged_compare_matched.txt"
  fi
  if [[ "${HEDGE_STRICT:-0}" == 1 ]]; then
    log "judged comparison, matched length, hedge-strict prompt -> $OUT/judged_compare_matched_nohedge.json"
    "${CMP[@]}" --pre-rollouts "$OUT/base_trunc_token" --pre-judged "$OUT/base_trunc_token_judged_v3_nohedge" \
      --post-rollouts "$OUT/post_greedy" --post-judged "$OUT/judged_v3_nohedge" \
      --out "$OUT/judged_compare_matched_nohedge.json" > "$OUT/judged_compare_matched_nohedge.txt"
  fi
  # 6. this run's Table 1 row
  if (( LIMIT == 0 )); then
    "$PY" "$E/paper_table.py" --name "$NAME" "$OUT" --out "$OUT/table.json" | tee "$OUT/table.md"
  fi
fi
log "done: $OUT"
