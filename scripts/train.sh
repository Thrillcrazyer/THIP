export WANDB_PROJECT=THIP_FINAL

## With PM Reward
export WANDB_NAME="Qwen-1.5B_THIP_1125"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    gspo_THIP.py \
    --dataset_name DeepMath-103k\
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\
    --output_dir Result/Qwen-1.5B_THIP_1125 \
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
    --save_steps 100 \
    --max_steps 3000 \
    --per_device_train_batch_size 4 \
    --num_generations 8 \
    --importance_sampling_level sequence \
#    --loss_type dr_grpo \

