export WANDB_PROJECT=THIP_DISTILL

## With PM Reward
export WANDB_NAME="Qwen-1.5B_THIP"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    gspo_THIP.py \
    --dataset_name DeepMath-103k\
    --model_name_or_path Thrillcrazyer/Qwen-1.5B_THIP \
    --output_dir Result/Qwen-1.5B_THIP \
    --logging_dir ./logs \
    --report_to wandb \
    --run_name $WANDB_NAME \
    --push_to_hub True \
    --use_liger_kernel True \
    --save_total_limit 3 \
    --save_safetensors True \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --max_prompt_length 512 \
    --max_completion_length 8192 \
    --log_completions True \
    --save_strategy steps \
    --use_vllm True \
    --vllm_mode colocate \
    --save_steps 50 \
    --max_steps 3500 \
    --per_device_train_batch_size 8 \
    --num_generations 8 \
    --importance_sampling_level sequence \
    --loss_type dr_grpo \

