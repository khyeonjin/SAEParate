#!/usr/bin/env bash
# Entry point for the full SAEParate pipelines.
#   bash run.sh <object|style|joint> [MODE]
# MODE: setup | train | unlearn | viz | all | resume   (joint has no viz; default: all)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TYPE="${1:-}"
case "${TYPE}" in
  object|style|joint) ;;
  *) echo "Usage: bash run.sh <object|style|joint> [setup|train|unlearn|viz|all|resume]"; exit 1 ;;
esac
shift
exec bash "${ROOT_DIR}/bash/train_unlearning_${TYPE}.sh" "$@"
