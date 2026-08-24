#!/usr/bin/env bash
set -euo pipefail

mode="${1:-download}"
if [[ "$mode" != download && "$mode" != --execute ]]; then
  echo "usage: $0 [--execute]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
official_root="${LEROBOT_TROSSEN_ROOT:-/home/robotarm/lerobot_trossen}"
official_script="$official_root/scripts/run_uploaded_v11.sh"

if [[ ! -x "$official_script" ]]; then
  echo "Official LeRobot/Trossen runner missing: $official_script" >&2
  exit 1
fi

export TCC_REAL_ROBOT_SOURCE_ROOT="$repo_root"
exec "$official_script" "$mode"
