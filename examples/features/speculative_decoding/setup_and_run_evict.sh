#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# One-shot bootstrap + run for the EVICT evaluation on a GPU stand.
#
#   1. (optional) fast-forward the current branch
#   2. ensure uv + a .venv, install this build precompiled (Python-only change:
#      VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto)
#   3. run run_evict_comparison.sh (cost table -> EVICT-vs-baseline comparison)
#
# Typical stand workflow:
#   git clone https://github.com/AsakoKabe/vllm.git && cd vllm
#   git checkout feat/evict-adaptive-verification
#   examples/features/speculative_decoding/setup_and_run_evict.sh
#
# Re-run after a code pull without reinstalling (Python-only edits are picked up
# by the editable install):
#   SKIP_INSTALL=1 examples/features/speculative_decoding/setup_and_run_evict.sh
#
# All run_evict_comparison.sh settings pass through, e.g.:
#   K=4 NUM_PROMPTS=32 examples/features/speculative_decoding/setup_and_run_evict.sh
#
# Gated models (e.g. Llama): export HF_TOKEN=hf_... before running.

set -euo pipefail

SKIP_INSTALL="${SKIP_INSTALL:-0}"   # 1 = skip venv/install, go straight to run
DO_PULL="${DO_PULL:-1}"             # 1 = git pull --ff-only the current branch first
PYVER="${PYVER:-3.12}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

echo "=============================================================="
echo " EVICT stand bootstrap"
echo "   repo   : ${REPO_ROOT}"
echo "   branch : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "=============================================================="

# ----- sync the branch (fast-forward only; tolerate a dirty tree) -----------
if [[ "${DO_PULL}" == "1" ]]; then
  echo ">>> git pull --ff-only ..."
  git pull --ff-only || echo ">>> WARNING: could not fast-forward (local changes " \
    "or diverged?); continuing with the CURRENT checkout. Resolve manually to " \
    "pick up the latest code (e.g. git stash && git pull --ff-only)."
fi

# ----- pass a HF token through under the name the hub client expects --------
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# ----- environment + install ------------------------------------------------
if [[ "${SKIP_INSTALL}" == "1" ]]; then
  echo ">>> SKIP_INSTALL=1: skipping venv/install."
else
  if ! command -v uv >/dev/null 2>&1; then
    echo ">>> uv not found; installing ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
  fi
  if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo ">>> creating .venv (python ${PYVER}) ..."
    uv venv --python "${PYVER}"
  fi
  echo ">>> installing vLLM (precompiled, editable) ..."
  VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto \
    --python "${REPO_ROOT}/.venv/bin/python"
fi

# ----- quick sanity: is this build importable and instrumented? -------------
"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import vllm  # noqa: F401
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.spec_decode.timing import SpecDecodeTimer  # noqa: F401
assert hasattr(SpeculativeConfig, "evict_enabled"), "EVICT not in this build"
print("sanity OK: vLLM importable, EVICT + spec-decode-timing present")
PY

# ----- run the two-phase comparison (all env overrides pass through) --------
echo ">>> launching run_evict_comparison.sh ..."
exec "${SCRIPT_DIR}/run_evict_comparison.sh"
