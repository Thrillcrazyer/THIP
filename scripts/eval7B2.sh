# 평가할 데이터셋 목록
TEST_NAMES=(
    "zwhe99/MATH"
    "zwhe99/aime90"
    "math-ai/aime25"
    "math-ai/aime24"
    "zwhe99/simplerl-minerva-math"
    "zwhe99/gpqa_diamond_mc"
    "zwhe99/simplerl-OlympiadBench"
)

# 모델 목록과 대응 경로
declare -A MODELS
MODELS=(
    # ["THIP"]="./Result/QWEN7_THIP/checkpoint-1000"
    # ["GSPO"]="./Result/QWEN7_GSPO/checkpoint-1000"
    # ["GRPO"]="./Result/QWEN7_GRPO/checkpoint-1000"
    # ["DRGRPO"]="./Result/QWEN7_DRGRPO/checkpoint-1000"
    #["SFT"]="./Result/Qwen-7B_SFT/checkpoint-500"
    ["ORG"]="Qwen/Qwen2.5-Math-7B"
)

# 공통 설정
COMMON_ARGS="
    --chat_template_name r1-distill-qwen
    --system_prompt_name simplerl
    --bf16 True
    --tensor_parallel_size 1
    --max_model_len 8192
    --temperature 1.0
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

        # 환경변수 포함 실행
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
        python evaluate.py \
            --base_model "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --data_id "$TEST_NAME" \
            $COMMON_ARGS \

        echo "✅ Finished ${MODEL_KEY} on ${TEST_NAME}, log saved to $LOG_FILE"
        echo "-----------------------------------------"
    done
done

echo "🎯 All evaluations completed successfully!"