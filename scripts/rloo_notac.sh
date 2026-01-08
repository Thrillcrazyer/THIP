export WANDB_PROJECT=TAC

export WANDB_NAME="Qwen-7B_NOTAC_RLOO"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    RLOO_NoTAC.py \
    --dataset_name DeepMath-103k\
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct\
    --output_dir Result/Qwen-7B_NOTAC_RLOO \
    --logging_dir ./logs \
    --report_to wandb \
    --run_name $WANDB_NAME \
    --push_to_hub True \
    --bf16 True \
    --use_liger_kernel True \
    --save_safetensors True \
    --learning_rate 1e-6 \
    --max_prompt_length 1024 \
    --max_completion_length 16384 \
    --save_total_limit 3 \
    --log_completions True \
    --save_strategy steps \
    --use_vllm True \
    --vllm_mode colocate \
    --save_steps 50 \
    --max_steps 240 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 2 \