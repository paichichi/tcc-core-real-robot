#!/usr/bin/env bash
set -euo pipefail

repo=/home/paichichi/projects/tcc-core-real-robot-v8
python=/home/paichichi/miniconda3/envs/tcc-core-parity/bin/python
config=configs/experiment_v8_hrp_official_single_view_60.yaml
buffer=runs/hrp_image_buffer_carrot_60_cartesian_velocity.sqlite3
hub_cache=/home/paichichi/.cache/huggingface/hub
tcc_source=/home/paichichi/projects/TCC-core

cd "$repo"
export PYTHONPATH=src

for backbone in ours_rn50 ours_vit r3m_unadapted d4r_imagenet; do
  output="runs/tcc_mlp_bc_v8_hrp_cartesian_velocity/$backbone/60"
  mkdir -p "$output"
  echo "$(date --iso-8601=seconds) START $backbone"
  "$python" scripts/train_hrp_end_to_end.py \
    --config "$config" \
    --image-buffer "$buffer" \
    --backbone "$backbone" \
    --hub-cache-dir "$hub_cache" \
    --offline \
    --tcc-source-root "$tcc_source" \
    --output-dir "$output" \
    --device cuda:0 \
    --num-workers 6 \
    2>&1 | tee "$output/train.log"
  echo "$(date --iso-8601=seconds) COMPLETE $backbone"
done
