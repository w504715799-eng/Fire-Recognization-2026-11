# Source from bash or zsh. Reuses the migrated environment without reinstalling it.
if [ -n "${ZSH_VERSION:-}" ]; then
  SCI_ACTIVATE_SOURCE="${(%):-%N}"
elif [ -n "${BASH_VERSION:-}" ]; then
  SCI_ACTIVATE_SOURCE="${BASH_SOURCE[0]}"
else
  printf '%s\n' 'Source this script from bash or zsh.' >&2
  return 1
fi
SCI_PROJECT_ROOT="$(cd "$(dirname "$SCI_ACTIVATE_SOURCE")/.." && pwd -P)"
export CONDA_PKGS_DIRS="$SCI_PROJECT_ROOT/.cache/conda"
export CONDA_ENVS_PATH="$SCI_PROJECT_ROOT/.local/miniconda3/envs"
export CONDARC="$SCI_PROJECT_ROOT/.local/condarc"
export PIP_CACHE_DIR="$SCI_PROJECT_ROOT/.cache/pip"
export TMPDIR="$SCI_PROJECT_ROOT/tmp/runtime"
export MPLCONFIGDIR="$SCI_PROJECT_ROOT/.cache/matplotlib"
export PYTHONPYCACHEPREFIX="$SCI_PROJECT_ROOT/.cache/pycache"
export HF_HOME="$SCI_PROJECT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CONDA_OFFLINE=true
export CONDA_NO_PLUGINS=true
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
source "$SCI_PROJECT_ROOT/.local/miniconda3/etc/profile.d/conda.sh" || return 1
conda activate "$SCI_PROJECT_ROOT/.local/miniconda3/envs/sci-env" || return 1
if [ "$(cd "$CONDA_PREFIX" && pwd -P)" != "$SCI_PROJECT_ROOT/.local/miniconda3/envs/sci-env" ]; then
  printf '%s\n' 'Unexpected environment prefix; activation failed.' >&2
  return 1
fi
# Package/network changes must be explicit and recorded; defaults keep checks offline.
