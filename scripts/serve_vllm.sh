#!/bin/bash
# vLLM 서빙 스크립트
# 사용법: bash scripts/serve_vllm.sh [모델경로] [GPU수]
#
# 예시:
#   bash scripts/serve_vllm.sh                                          # 기본
#   bash scripts/serve_vllm.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-14B 2
#   bash scripts/serve_vllm.sh /path/to/finetuned_model 1

MODEL=${1:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1B"}
TP_SIZE=${2:-1}
PORT=${3:-8000}

# transformers 호환성 fix (vLLM 0.10.2 + transformers 최신 버전 충돌 방지)
pip install 'transformers>=4.48,<4.52' -q

echo "============================================"
echo " vLLM Serving"
echo " Model : $MODEL"
echo " TP    : $TP_SIZE GPU(s)"
echo " Port  : $PORT"
echo "============================================"

vllm serve "$MODEL" \
    --port "$PORT" \
    --dtype auto \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --max-num-seqs 256


