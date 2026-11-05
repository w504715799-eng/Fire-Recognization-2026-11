#!/usr/bin/env bash
# Deliberate environment creation only on the configured, approved cloud instance.
set -euo pipefail
FIRE_REMOTE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ "${1:-}" != "--create-env" || -z "${2:-}" ]]; then
  echo "Usage (approved cloud only): bash cloud/bootstrap.sh --create-env configs/cloud.local.json"
  exit 2
fi
cd "$FIRE_REMOTE_ROOT"
python3 -B code/fire_r2_cli.py cloud-init --cloud-config "$2" --execute
FIRE_ENV_PREFIX="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_prefix"])' "$2")"
export CONDA_PKGS_DIRS="$FIRE_REMOTE_ROOT/.cache/conda"
export PIP_CACHE_DIR="$FIRE_REMOTE_ROOT/.cache/pip"
export TMPDIR="$FIRE_REMOTE_ROOT/tmp"
export PYTHONPYCACHEPREFIX="$FIRE_REMOTE_ROOT/.cache/pycache"
if [[ -e "$FIRE_ENV_PREFIX" ]]; then
  echo "Environment already exists; reusing it: $FIRE_ENV_PREFIX"
else
  FIRE_ENV_MODE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("environment_mode","conda"))' "$2")"
  if [[ "$FIRE_ENV_MODE" == preinstalled_overlay ]]; then
    FIRE_BASE_PYTHON="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_python"])' "$2")"
    "$FIRE_BASE_PYTHON" -m venv --system-site-packages "$FIRE_ENV_PREFIX"
  elif [[ "$FIRE_ENV_MODE" == conda ]]; then
    conda create --prefix "$FIRE_ENV_PREFIX" python=3.11 pip -y
  else
    echo "Unknown environment mode: $FIRE_ENV_MODE" >&2
    exit 2
  fi
fi
echo "Environment ready. No model packages or weights were installed by this script."
echo "Activate this prefix, select a compatible CUDA torch build, then explicitly install cloud/requirements.txt."
