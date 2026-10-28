#!/usr/bin/env bash
# Submission target: one RTX PRO 6000 (96 GB), TP=1, DP=1.
# Activate an environment containing a compatible SGLang before running.
set -euo pipefail
MODEL_PATH=${1:?Usage: bash scripts/serve_model.sh /path/to/model}
[[ -f "$MODEL_PATH/config.json" ]] || { echo "Missing model config" >&2; exit 1; }
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
[[ "$CUDA_VISIBLE_DEVICES" != *,* && -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" != -1 ]] || {
  echo "Select exactly one GPU with CUDA_VISIBLE_DEVICES (default: 0)" >&2
  exit 2
}
export NCCL_IB_DISABLE=1 TOKENIZERS_PARALLELISM=false
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name Senova-4B-v2 \
  --host 127.0.0.1 --port "${SGLANG_PORT:-8000}" \
  --tp-size 1 --dp-size 1 \
  --context-length 262144 \
  --mem-fraction-static 0.8 \
  --max-running-requests 1
