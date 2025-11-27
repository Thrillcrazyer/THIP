export WANDB_PROJECT=THIP_DISTILL

## With PM Reward
export WANDB_NAME="Qwen-1.5B_NoThip"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    gspo_THIP.py \
    --dataset_name DeepMath-103k\
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B\
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
    --max_prompt_length 1024 \
    --max_completion_length 8192 \
    --use_vllm True \
    --vllm_mode colocate \
    --log_completions True \
    --save_strategy steps \
    --save_steps 50 \
    --max_steps 5000 \
    --per_device_train_batch_size 4 \
    --num_generations 8 \
    --loss_type dr_grpo \

# ## GRPO
# export WANDB_NAME="Qwen-1.5B_Distill_GRPO"

# accelerate launch \
#     --config_file configs/deepspeed_zero3.yaml \
#     gspo_NOTHIP.py \
#     --dataset_name DeepMath-103k \
#     --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#     --output_dir Result/Qwen-1.5B_GRPO \
#     --logging_dir ./logs \
#     --report_to wandb \
#     --run_name $WANDB_NAME \
#     --push_to_hub True \
#     --use_liger_kernel True \
#     --save_total_limit 3 \
#     --save_safetensors True \
#     --learning_rate 2e-6 \
#     --torch_dtype bfloat16 \
#     --max_prompt_length 512 \
#     --max_completion_length 8192 \
#     --use_vllm True \
#     --vllm_mode colocate \
#     --log_completions True \
#     --save_strategy steps \
#     --save_steps 25 \
#     --max_steps 500 \
#     --per_device_train_batch_size 8 \
#     --num_generations 8 \
#     --loss_type grpo \
    
# ## DRGRPO
# export WANDB_NAME="Qwen-1.5B_Distill_DRGRPO"

# accelerate launch \
#     --config_file configs/deepspeed_zero3.yaml \
#     gspo_NOTHIP.py \
#     --dataset_name DeepMath-103k \
#     --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#     --output_dir Result/Qwen-1.5B_DRGRPO \
#     --logging_dir ./logs \
#     --report_to wandb \
#     --run_name $WANDB_NAME \
#     --push_to_hub True \
#     --use_liger_kernel True \
#     --save_total_limit 3 \
#     --save_safetensors True \
#     --learning_rate 2e-6 \
#     --torch_dtype bfloat16 \
#     --max_prompt_length 512 \
#     --max_completion_length 8192 \
#     --use_vllm True \
#     --vllm_mode colocate \
#     --log_completions True \
#     --save_strategy steps \
#     --save_steps 25 \
#     --max_steps 500 \
#     --per_device_train_batch_size 8 \
#     --num_generations 8 \
#     --loss_type dr_grpo \


