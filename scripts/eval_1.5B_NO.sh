# 평가할 데이터셋 목록
TEST_NAMES=(
    "zwhe99/MATH"
    "zwhe99/aime90"
    "math-ai/aime25"
    "zwhe99/simplerl-minerva-math"
    "zwhe99/simplerl-OlympiadBench"
)

# 모델 목록과 대응 경로
declare -A MODELS
MODELS=(
    #["BASELINE"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    #["GSPONOTAC"]="./Result/Qwen-7B_NOTAC_GSPO/checkpoint-240"
    #["GSPOTAC"]="./Result/Qwen-7B_TAC_GSPO/checkpoint-240"
    #["RLOONOTAC"]="./Result/Qwen-7B_NOTAC_RLOO/checkpoint-240"
    #["RLOOTAC"]="./Result/Qwen-7B_TAC_RLOO/checkpoint-240"
    ["GRPONOTAC"]="./Result/Qwen-7B_NOTAC_GRPO/checkpoint-240"
)


# 공통 설정
COMMON_ARGS="
    --chat_template_name r1-distill-qwen
    --system_prompt_name simplerl
    --bf16 True
    --tensor_parallel_size 4
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

echo "evaluations completed"