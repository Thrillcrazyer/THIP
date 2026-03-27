export WANDB_PROJECT=TAC

## With PM Reward
export WANDB_NAME="Qwen-2.5-1.5B_TAC_GSPO"

# ── vLLM 서버 설정 (Answer2EventAgent용) ──
EVENTLOG_MODEL=${EVENTLOG_MODEL:-"Qwen/Qwen2.5-72B-Instruct"}
EVENTLOG_PORT=${EVENTLOG_PORT:-8000}
EVENTLOG_TP=${EVENTLOG_TP:-1}
EVENTLOG_GPUS=${EVENTLOG_GPUS:-""}

export EVENTLOG_API_KEY="EMPTY"
export EVENTLOG_BASE_URL="http://localhost:${EVENTLOG_PORT}/v1"
export EVENTLOG_MODEL_NAME="$EVENTLOG_MODEL"

echo "============================================"
echo " Starting vLLM for EventLog Agent"
echo " Model : $EVENTLOG_MODEL"
echo " Port  : $EVENTLOG_PORT"
echo " GPUs  : ${EVENTLOG_GPUS:-all}"
echo "============================================"

# vLLM 서버 백그라운드 실행
if [ -n "$EVENTLOG_GPUS" ]; then
    CUDA_VISIBLE_DEVICES="$EVENTLOG_GPUS" vllm serve "$EVENTLOG_MODEL" \
        --port "$EVENTLOG_PORT" \
        --dtype auto \
        --tensor-parallel-size "$EVENTLOG_TP" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 256 &
else
    vllm serve "$EVENTLOG_MODEL" \
        --port "$EVENTLOG_PORT" \
        --dtype auto \
        --tensor-parallel-size "$EVENTLOG_TP" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 256 &
fi
VLLM_PID=$!

# vLLM 서버가 준비될 때까지 대기
echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:${EVENTLOG_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server process died"
        exit 1
    fi
    sleep 5
done

# 서버가 준비되지 않으면 종료
if ! curl -sf http://localhost:${EVENTLOG_PORT}/health > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within timeout"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# 종료 시 vLLM 서버 정리
trap "echo 'Stopping vLLM server...'; kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null" EXIT

# ── 학습 시작 ──
accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    gspo_THIP.py \
    --dataset_name DeepMath-103k\
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct\
    --output_dir Result/Qwen-2.5-1.5B_TAC_GSPO \
    --logging_dir ./logs \
    --report_to wandb \
    --run_name $WANDB_NAME \
    --push_to_hub True \
    --save_safetensors True \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --max_prompt_length 1024 \
    --max_completion_length 16384 \
    --log_completions True \
    --save_strategy steps \
    --use_vllm True \
    --vllm_mode colocate \
    --save_steps 50 \
    --max_steps 240 \
    --per_device_train_batch_size 2 \
    --num_generations 8 \
    --gradient_accumulation_steps 32 \
    --importance_sampling_level sequence \
#    --loss_type dr_grpo \

