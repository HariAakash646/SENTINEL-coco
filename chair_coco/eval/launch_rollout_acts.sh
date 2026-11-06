#!/usr/bin/env bash

set -u

usage() {
    cat <<'EOF'
Usage:
  launch_rollout_acts.sh GPU_IDS [rollout_acts.py arguments...]

GPU_IDS is a comma-separated list, for example: 0,1,2,3

Example:
  src/extract/launch_rollout_acts.sh 0,1,2,3 \
    --manifest data/coco_train.json \
    --rundir runs/rollouts_train \
    --num_rollouts 8

The launcher supplies --device, --num_shards, and --shard for each worker.
Do not pass those options yourself.
EOF
}

if (( $# == 0 )); then
    usage >&2
    exit 2
fi

gpu_list=$1
shift

if [[ -z "$gpu_list" ]]; then
    echo "error: GPU_IDS cannot be empty" >&2
    exit 2
fi

for arg in "$@"; do
    case "$arg" in
        --device|--device=*|--num_shards|--num_shards=*|--shard|--shard=*)
            echo "error: $arg is managed by the launcher" >&2
            exit 2
            ;;
    esac
done

IFS=',' read -r -a gpu_ids <<< "$gpu_list"
declare -A seen_gpus=()
for gpu in "${gpu_ids[@]}"; do
    if [[ -z "$gpu" ]]; then
        echo "error: GPU_IDS contains an empty entry: $gpu_list" >&2
        exit 2
    fi
    # A repeated GPU runs several workers on one card. Opt in with PROBE_SHARE_GPUS=1: worth it
    # for greedy one-image-at-a-time captioning, which leaves the card half idle.
    if [[ -n "${seen_gpus[$gpu]:-}" && "${PROBE_SHARE_GPUS:-0}" != 1 ]]; then
        echo "error: GPU $gpu is listed more than once (set PROBE_SHARE_GPUS=1 to share a card)" >&2
        exit 2
    fi
    seen_gpus[$gpu]=1
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python}
num_shards=${#gpu_ids[@]}
pids=()

stop_workers() {
    trap - INT TERM
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
    exit 130
}
trap stop_workers INT TERM

for shard in "${!gpu_ids[@]}"; do
    gpu=${gpu_ids[$shard]}
    echo "launching shard $shard/$num_shards on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" "$script_dir/rollout_acts.py" \
        --device cuda:0 \
        --num_shards "$num_shards" \
        --shard "$shard" \
        "$@" &
    pids+=("$!")
done

status=0
for shard in "${!pids[@]}"; do
    if wait "${pids[$shard]}"; then
        echo "shard $shard completed"
    else
        worker_status=$?
        echo "shard $shard failed with status $worker_status" >&2
        status=1
    fi
done

exit "$status"
