export WANDB_PROJECT=Qwen-1.5B-THIP
export WANDB_NAME="Qwen-1.5B_THIP_GSPO_PMREWARD-$(date +%Y%m%d-%H%M%S)"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    gspo_THIP.py \
    --dataset_name DeepMath-103k \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir Result/Qwen-1.5B_THIP_GSPO_PMREWARD \
    --logging_dir ./logs \
    --report_to wandb \
    --run_name $WANDB_NAME \
    --use_liger_kernel True \
    --save_total_limit 3 \
    --save_safetensors True \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --max_prompt_length 512 \
    --max_completion_length 8192 \
    --use_vllm True \
    --vllm_mode colocate \
    --log_completions True \
    --save_strategy steps \
    --save_steps 128 \
    --per_device_train_batch_size 4 \
    --num_generations 6 \
    --importance_sampling_level sequence \
    --loss_type grpo \
    --attn_implementation= 'flash_attention_2'

export WANDB_PROJECT=Qwen-1.5B-THIP
export WANDB_NAME="Qwen-1.5B_THIP_GSPO_NOPMREWARD-$(date +%Y%m%d-%H%M%S)"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    gspo_THIP.py \
    --dataset_name DeepMath-103k \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --output_dir Result/Qwen-1.5B_THIP_GSPO_NOPMREWARD \
    --logging_dir ./logs \
    --report_to wandb \
    --run_name $WANDB_NAME \
    --use_liger_kernel True \
    --save_total_limit 3 \
    --save_safetensors True \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --max_prompt_length 512 \
    --max_completion_length 8192 \
    --use_vllm True \
    --vllm_mode colocate \
    --log_completions True \
    --save_strategy steps \
    --save_steps 128 \
    --per_device_train_batch_size 4 \
    --num_generations 6 \
    --importance_sampling_level sequence \
    --loss_type grpo \
    --attn_implementation= 'flash_attention_2'