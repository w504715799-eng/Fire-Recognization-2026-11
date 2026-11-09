#!/usr/bin/env bash
# Source only on the authorized cloud host; does not load models or start GPU work.
if [[ "$(uname -s)" != Linux || ! -f /FIRE_R2-runtime/owner.json ]]; then
  echo 'Cloud FIRE_R2 runtime is required.' >&2
  return 1 2>/dev/null || exit 1
fi
source /root/rivermind-data/FIRE_R2/.envs/sci-env/bin/activate
export PYTHONPATH=/FIRE_R2-runtime/teacher-site-packages:/root/rivermind-data/FIRE_R2/src
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/FIRE_R2-runtime/cache/huggingface
export XDG_CACHE_HOME=/FIRE_R2-runtime/cache
export TMPDIR=/FIRE_R2-runtime/tmp
export YOLO_CONFIG_DIR=/root/rivermind-data/FIRE_R2/.cache/ultralytics
export PIP_NO_INDEX=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR=/FIRE_R2-runtime/cache/pip
