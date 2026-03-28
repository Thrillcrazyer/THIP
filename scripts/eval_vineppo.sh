#!/bin/bash
# VinePPO 모델 평가 스크립트
# 사용법: bash scripts/eval_vineppo.sh

# 평가할 데이터셋 목록
TEST_NAMES=(
    "zwhe99/MATH"
    "zwhe99/aime90"
    "math-ai/aime25"
    "zwhe99/simplerl-minerva-math"
    "zwhe99/simplerl-OlympiadBench"
)

# VinePPO 모델 경로 (필요에 따라 변경)
declare -A MODELS
MODELS=(
    ["VinePPO_7B"]="realtreetune/deepseekmath-7b-sft-MATH-v2"
)

# 공통 설정
# VinePPO(DeepSeek 계열)는 r1-distill-qwen 템플릿 사용
COMMON_ARGS="
    --chat_template_name r1-distill-qwen
    --system_prompt_name disabled
    --bf16 True
    --tensor_parallel_size 1
    --max_model_len 16384
    --temperature 0.6
    --top_p 0.95
    --n 1
"

# 순회하면서 평가
for TEST_NAME in "${TEST_NAMES[@]}"; do
    echo "🔹 Evaluating dataset: $TEST_NAME"
    for MODEL_KEY in "${!MODELS[@]}"; do
        MODEL_PATH=${MODELS[$MODEL_KEY]}
        OUTPUT_DIR="./eval_results/${MODEL_KEY}/$(basename $TEST_NAME)"
        LOG_FILE="log_${MODEL_KEY}_$(basename $TEST_NAME).txt"

        echo "🚀 Running ${MODEL_KEY} on ${TEST_NAME}"
        mkdir -p "$OUTPUT_DIR"

        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
        python evaluate.py \
            --base_model "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --data_id "$TEST_NAME" \
            $COMMON_ARGS \
            2>&1 | tee "$LOG_FILE"

        echo "✅ Finished ${MODEL_KEY} on ${TEST_NAME}, log saved to $LOG_FILE"
        echo "-----------------------------------------"
    done
done

echo "All VinePPO evaluations completed"
