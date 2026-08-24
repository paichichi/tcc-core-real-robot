#!/usr/bin/env bash
set -euo pipefail

mode="${1:-shadow}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="$repo_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Missing $python_bin; create/install the project environment first." >&2
  exit 1
fi

tcc_source_root="${TCC_SOURCE_ROOT:-/home/robotarm/TCC-core}"
if [[ ! -f "$tcc_source_root/xirl/models.py" ]]; then
  echo "TCC source missing: $tcc_source_root/xirl/models.py" >&2
  exit 1
fi

args=(
  --config configs/experiment_v11_basic_chunked_mlp_100.yaml
  --robot-config configs/robot.yaml
  --backbone ours_rn50
  --demonstrations 100
  --task carrot
  --camera-backend realsense-sdk
  --cam-main-serial 838212073584
  --cam-wrist-serial 409122274608
  --tcc-source-root "$tcc_source_root"
  --device auto
)

case "$mode" in
  shadow)
    args+=(--online --execute-home --max-steps 359)
    ;;
  10|359)
    args+=(
      --offline
      --execute-policy
      --supervised-bounded-test
      --emergency-stop-ready
      --max-steps "$mode"
    )
    ;;
  *)
    echo "usage: $0 {shadow|10|359}" >&2
    exit 2
    ;;
esac

exec "$python_bin" scripts/run_policy.py "${args[@]}"
