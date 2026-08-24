#!/usr/bin/env bash
set -euo pipefail

mode="${1:-download}"
if [[ "$mode" != download && "$mode" != --execute ]]; then
  echo "usage: $0 [--execute]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
launcher_log="$repo_root/output/v11_launcher_${run_stamp}.txt"
mkdir -p "$repo_root/output"
exec > >(tee -a "$launcher_log") 2>&1
echo "V11 launcher text log: $launcher_log"

if [[ -n "${LEROBOT_TROSSEN_ROOT:-}" ]]; then
  official_root="$LEROBOT_TROSSEN_ROOT"
else
  official_root=""
  for candidate in \
    "$repo_root/../lerobot_trossen" \
    "$repo_root/../lerobot_trossen-act-lite-review" \
    "${HOME}/projects/lerobot_trossen" \
    "${HOME}/projects/lerobot_trossen-act-lite-review"; do
    if [[ -x "$candidate/scripts/run_uploaded_v11.sh" ]]; then
      official_root="$(cd "$candidate" && pwd)"
      break
    fi
  done
fi

if [[ -z "$official_root" ]]; then
  echo "Could not find a lerobot_trossen checkout containing scripts/run_uploaded_v11.sh." >&2
  echo "Set LEROBOT_TROSSEN_ROOT to its absolute path." >&2
  exit 1
fi
official_script="$official_root/scripts/run_uploaded_v11.sh"

if [[ ! -x "$official_script" ]]; then
  echo "Official LeRobot/Trossen runner missing: $official_script" >&2
  exit 1
fi

export TCC_REAL_ROBOT_SOURCE_ROOT="$repo_root"
if [[ -z "${BACKBONE_SOURCE_ROOT:-}" && -d "$repo_root/../TCC-core" ]]; then
  export BACKBONE_SOURCE_ROOT="$(cd "$repo_root/../TCC-core" && pwd)"
fi
echo "Official LeRobot root: $official_root"
echo "TCC real-robot root: $TCC_REAL_ROBOT_SOURCE_ROOT"
echo "Backbone source root: ${BACKBONE_SOURCE_ROOT:-<auto-detect>}"
exec "$official_script" "$mode"
