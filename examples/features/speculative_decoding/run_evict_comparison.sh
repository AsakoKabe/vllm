#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Two-phase EVICT evaluation on the paper's MoE setup (Qwen3-30B-A3B + EAGLE-3):
#   1. Profile the EVICT cost table C(m) (vllm.v1.spec_decode.evict.build_cost_table).
#   2. Run the EVICT-vs-baseline comparison (evict_vs_baseline.py) using that table.
#
# Requires a GPU and this build installed in .venv (Python-only changes:
#   VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto).
#
# EAGLE-3 head: RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3 is the
# vLLM-documented head for target Qwen/Qwen3-30B-A3B-Instruct-2507. The
# lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-* heads are SpecForge/SGLang-format and may
# not load in vLLM without conversion — prefer the RedHatAI head here.
#
# All settings are env-overridable, e.g.:
#   K=8 NUM_PROMPTS=32 ./run_evict_comparison.sh
#   MODEL=meta-llama/Llama-3.1-8B-Instruct EAGLE_DIR=yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
#     ENABLE_ROUTED_EXPERTS=0 ./run_evict_comparison.sh   # dense smoke test

set -euo pipefail

# ----- configuration (override via environment) -----------------------------
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
EAGLE_DIR="${EAGLE_DIR:-RedHatAI/Qwen3-30B-A3B-Instruct-2507-speculator.eagle3}"
METHOD="${METHOD:-eagle3}"
K="${K:-8}"                              # num_speculative_tokens (== cost-table max m)
NUM_PROMPTS="${NUM_PROMPTS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
TP="${TP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
EVICT_MIN_K="${EVICT_MIN_K:-1}"
EVICT_BATCH_REDUCE="${EVICT_BATCH_REDUCE:-max}"
# 1 = capture routed experts so U_r is populated (MoE targets); 0 = skip (dense).
ENABLE_ROUTED_EXPERTS="${ENABLE_ROUTED_EXPERTS:-1}"
# 1 = also run the vanilla-AR baseline for an absolute speedup denominator.
WITH_AR_BASELINE="${WITH_AR_BASELINE:-1}"
# 1 = re-profile the cost table even if COST_TABLE already exists.
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# ----- paths ----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
COST_TABLE="${COST_TABLE:-${REPO_ROOT}/evict_cost_table.json}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python interpreter not found at ${PYTHON}." >&2
  echo "Set PYTHON=... or create the venv (uv venv --python 3.12)." >&2
  exit 1
fi

echo "=============================================================="
echo " EVICT comparison"
echo "   target model : ${MODEL}"
echo "   eagle head   : ${EAGLE_DIR} (${METHOD})"
echo "   K            : ${K}"
echo "   cost table   : ${COST_TABLE}"
echo "   python       : ${PYTHON}"
echo "=============================================================="

cd "${REPO_ROOT}"

# ----- phase 1: build the EVICT cost table C(m) -----------------------------
if [[ "${FORCE_REBUILD}" != "1" && -s "${COST_TABLE}" ]]; then
  echo ">>> phase 1: cost table already exists at ${COST_TABLE} (FORCE_REBUILD=1 to rebuild); skipping."
else
  echo ">>> phase 1: profiling EVICT cost table C(m) for m=1..${K} ..."
  "${PYTHON}" -m vllm.v1.spec_decode.evict.build_cost_table \
    --model "${MODEL}" \
    --method "${METHOD}" \
    --eagle-dir "${EAGLE_DIR}" \
    --max-spec-tokens "${K}" \
    --num-prompts "${NUM_PROMPTS}" \
    --output-len "${OUTPUT_LEN}" \
    --tp "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --out "${COST_TABLE}"
fi

echo ">>> cost table contents:"
cat "${COST_TABLE}"
echo

# ----- phase 2: EVICT-vs-baseline comparison --------------------------------
COMPARE_ARGS=(
  --model "${MODEL}"
  --method "${METHOD}"
  --eagle-dir "${EAGLE_DIR}"
  --num-spec-tokens "${K}"
  --num-prompts "${NUM_PROMPTS}"
  --output-len "${OUTPUT_LEN}"
  --tp "${TP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --evict-cost-table "${COST_TABLE}"
  --evict-min-k "${EVICT_MIN_K}"
  --evict-batch-reduce "${EVICT_BATCH_REDUCE}"
)
[[ "${ENABLE_ROUTED_EXPERTS}" == "1" ]] && COMPARE_ARGS+=(--enable-return-routed-experts)
[[ "${WITH_AR_BASELINE}" != "1" ]] && COMPARE_ARGS+=(--skip-ar)

echo ">>> phase 2: running EVICT-vs-baseline comparison ..."
"${PYTHON}" examples/features/speculative_decoding/evict_vs_baseline.py "${COMPARE_ARGS[@]}"

echo ">>> done."
