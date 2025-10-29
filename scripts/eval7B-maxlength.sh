# 평가할 데이터셋 목록
TEST_NAMES=(
    "zwhe99/aime90"
    "math-ai/aime25"
    "zwhe99/simplerl-OlympiadBench"
)

# 모델 목록과 대응 경로
declare -A MODELS
MODELS=(
    ["PM7"]="./Result/Qwen-7B_THIP/checkpoint-3500"
    ["BASELINE"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ['EXGRPO']='rzzhan/ExGRPO-Qwen2.5-Math-7B-Zero'
    ["MATH7"]="zwhe99/DeepMath-Zero-7B"
    ["PRIME"]="PRIME-RL/Eurus-2-7B-PRIME"
)

# max_model_len 값들
MAX_MODEL_LENS=(2048 4096 8192 16384)

# 공통 설정 (max_model_len 제외)
COMMON_ARGS="
    --chat_template_name r1-distill-qwen
    --system_prompt_name simplerl
    --bf16 True
    --tensor_parallel_size 4
    --temperature 0.6
    --top_p 0.95
    --n 1
"

# 순회하면서 평가
for MAX_LEN in "${MAX_MODEL_LENS[@]}"; do
    echo "🔸 Testing with max_model_len=$MAX_LEN"
    
    for TEST_NAME in "${TEST_NAMES[@]}"; do
        echo "🔹 Evaluating dataset: $TEST_NAME"
        
        for MODEL_KEY in "${!MODELS[@]}"; do
            MODEL_PATH=${MODELS[$MODEL_KEY]}
            # output_dir에 max_model_len 포함
            OUTPUT_DIR="./eval_results/${MODEL_KEY}/len_${MAX_LEN}/$(basename $TEST_NAME)"
            LOG_FILE="log_${MODEL_KEY}_len${MAX_LEN}_$(basename $TEST_NAME).txt"

            echo "🚀 Running ${MODEL_KEY} on ${TEST_NAME} with max_len=${MAX_LEN}"
            mkdir -p "$OUTPUT_DIR"

            # 환경변수 포함 실행
            VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
            python evaluate.py \
                --base_model "$MODEL_PATH" \
                --output_dir "$OUTPUT_DIR" \
                --data_id "$TEST_NAME" \
                --max_model_len $MAX_LEN \
                $COMMON_ARGS \
                2>&1 | tee "$LOG_FILE"

            echo "✅ Finished ${MODEL_KEY} on ${TEST_NAME} with max_len=${MAX_LEN}"
            echo "-----------------------------------------"
        done
    done
    
    echo "🎉 Completed all evaluations for max_model_len=$MAX_LEN"
    echo "========================================="
done

echo "🎯 All evaluations completed successfully!"