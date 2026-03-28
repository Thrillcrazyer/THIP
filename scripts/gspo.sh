export WANDB_PROJECT=TAC

## With PM Reward
export WANDB_NAME="Qwen-2.5-1.5B_TAC_GSPO"

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
    --learning_rate 1e-6 \
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

