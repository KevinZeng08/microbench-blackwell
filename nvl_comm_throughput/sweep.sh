#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
make
result_dir="${RESULT_DIR:-results/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$result_dir"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
torchrun --standalone --nproc-per-node=4 benchmark.py \
    --output "$result_dir/sweep4.csv" "$@" \
    2>&1 | tee "$result_dir/sweep4.log"
python plot.py "$result_dir/sweep4.csv" --output assets/bandwidth.png
python plot.py "$result_dir/sweep4.csv" --x-axis ctas --sizes 1073741824 \
    --output assets/ctas.png
