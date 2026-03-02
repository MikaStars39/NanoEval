#!/usr/bin/env bash
set -euo pipefail

# Real offline evaluation for math tasks with a local model.
REPO_ROOT=/mnt/llm-train/users/explore-train/qingyu/NanoEval
SYSTEM_PROMPT="You are a careful math solver. Show reasoning clearly and end with the final answer in \\boxed{}."
MODEL_PATH=/mnt/llm-train/users/explore-train/qingyu/.cache/DeepSeek-R1-Distill-Qwen-1.5B
WORKDIR="${REPO_ROOT}/outputs/test"

LOG_FILE="${WORKDIR}/run.log"
mkdir -p "${WORKDIR}"

TASK_ARGS=(
  --stage all
  --task-dir ${REPO_ROOT}/outputs/nano_eval
  --tasks "aime2024@4,aime2025@8"
  --system-prompt "${SYSTEM_PROMPT}"
  --n-proc 8
)

ROLLOUT_ARGS=(
  --backend offline
  --model-path "${MODEL_PATH}"
  --work-dir "${WORKDIR}"
  --temperature 1
  --max-tokens 32768
)

python "${REPO_ROOT}/run.py" \
  "${TASK_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
