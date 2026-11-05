#!/usr/bin/env bash
# Source on the authorized cloud server; keeps all runtime caches in this project.
FIRE_REMOTE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "$(uname -s)" != Linux || ! -x "$FIRE_REMOTE_ROOT/.envs/sci-env/bin/python" ]]; then
  echo "Cloud Linux project with its own sci-env is required." >&2
  return 2 2>/dev/null || exit 2
fi
export PATH="$FIRE_REMOTE_ROOT/.envs/sci-env/bin:$PATH"
export VIRTUAL_ENV="$FIRE_REMOTE_ROOT/.envs/sci-env"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$FIRE_REMOTE_ROOT/tmp"
export XDG_CACHE_HOME="$FIRE_REMOTE_ROOT/.cache"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export YOLO_CONFIG_DIR="$XDG_CACHE_HOME/ultralytics"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_OFFLINE=true
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" "$TORCH_HOME" "$HF_HOME"
