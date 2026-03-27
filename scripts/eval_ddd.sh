#!/bin/bash
set -e

# ── 모델 목록 ──
declare -A MODELS
MODELS=(
    ["EXAONE"]="LGAI-EXAONE/EXAONE-Deep-7.8B"
    ["Qwen-7B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ["PHI"]="microsoft/phi-4"
    ["Qwen-14B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    ["Qwen-32B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ["TACReward7B"]="Thrillcrazyer/TACReward7B"
    ["TACReward1.5B"]="Thrillcrazyer/TACReward1.5B"
)

# ── 설정 ──
PORT=8000
TP_SIZE=2
API_BASE="http://localhost:${PORT}/v1"
DATASET_NAME="nvidia/Nemotron-Math-v2"
QUESTION_COL="problem"
ANSWER_COL="expected_answer"
OUTPUT_DIR="./results_test"
MAX_TOKENS=16384
TEMPERATURE=0.6
TOP_P=0.9
SPLIT="medium"

# ── 모델별 순회: vLLM 서빙 → eval → 서버 종료 ──
for MODEL_KEY in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_KEY]}"
    echo "============================================"
    echo " [${MODEL_KEY}] ${MODEL_PATH}"
    echo "============================================"

    # 1) vLLM 서버 시작 (백그라운드)
    echo "🚀 Starting vLLM server for ${MODEL_KEY}..."
    vllm serve "$MODEL_PATH" \
        --port "$PORT" \
        --dtype auto \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len "$MAX_TOKENS" \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        --max-num-seqs 256 \
        --trust-remote-code &
    VLLM_PID=$!

    # 서버가 준비될 때까지 대기
    echo "⏳ Waiting for vLLM server to be ready..."
    for i in $(seq 1 120); do
        if curl -s "${API_BASE}/models" > /dev/null 2>&1; then
            echo "✅ vLLM server is ready (took ${i}s)"
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "❌ vLLM server process died unexpectedly"
            exit 1
        fi
        sleep 1
    done

    # 타임아웃 체크
    if ! curl -s "${API_BASE}/models" > /dev/null 2>&1; then
        echo "❌ vLLM server failed to start within 120s"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        continue
    fi

    # 2) 평가 실행
    RESULT_PATH="${OUTPUT_DIR}/${MODEL_KEY}_results.csv"
    mkdir -p "$OUTPUT_DIR"

    echo "📊 Running eval for ${MODEL_KEY}..."
    python eval_deepmath.py \
        --api_base "$API_BASE" \
        --model "$MODEL_PATH" \
        --dataset_name "$DATASET_NAME" \
        --question_col "$QUESTION_COL" \
        --answer_col "$ANSWER_COL" \
        --output_path "$RESULT_PATH" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --dataset_split "$SPLIT" \
        --top_p "$TOP_P" \
        --skip_process_reward True

    echo "✅ Finished eval for ${MODEL_KEY} → ${RESULT_PATH}"

    # 3) vLLM 서버 종료
    echo "🛑 Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    sleep 5

    echo "-----------------------------------------"
done

echo "🎉 All models evaluated!"
