# 평가할 데이터셋 목록
TEST_NAMES=(
    "zwhe99/MATH"
    "zwhe99/aime90"
    "math-ai/aime24"
    "zwhe99/simplerl-minerva-math"
    "zwhe99/gpqa_diamond_mc"
    "zwhe99/simplerl-OlympiadBench"
)

# 모델 목록과 대응 경로
declare -A MODELS
MODELS=(
    ["PM7"]="./Result/Qwen-7B_THIP/checkpoint-3500"
    ["BASELINE"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ["PRIME"]="PRIME-RL/Eurus-2-7B-PRIME"
    ["MATH7"]="zwhe99/DeepMath-7B"
    ["Skywork"]='Skywork/Skywork-OR1-7B'
    # ["PM"]="./Result/Qwen-1.5B_THIP/checkpoint-500"
    # ["SFT"]="./Result/Qwen-1.5B_SFT/checkpoint-110"
    # ["GSPO"]="./Result/Qwen-1.5B_GSPO/checkpoint-500"
    #["GRPO"]="./Result/Qwen-1.5B_GRPO/checkpoint-500"
    #["DRGRPO"]="./Result/Qwen-1.5B_DRGRPO/checkpoint-425"
    # ["BASELINE"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # ["DRAGRPO"]="SpiceRL/DRA-GRPO"
    # ["PM2"]="./Result/Qwen-1.5B_THIP2/checkpoint-500"
    # ["RS3"]="knoveleng/Open-RS3"
    # ["Graph"]="HKUST-DSAIL/Graph-R1-1.5B"
    # ["STILL"]="RUC-AIBOX/STILL-3-1.5B-preview"
    # ["ExGRPO"]="rzzhan/ExGRPO-Qwen2.5-Math-1.5B-Zero"
    #["DeepMath"]="zwhe99/DeepMath-1.5B"
)
# https://arxiv.org/pdf/2508.05170 (데이터셋기준으)

# 공통 설정
COMMON_ARGS="
    --chat_template_name r1-distill-qwen
    --system_prompt_name simplerl
    --bf16 True
    --tensor_parallel_size 1
    --max_model_len 8192
    --temperature 0.6
    --top_p 0.95
    --n 16
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