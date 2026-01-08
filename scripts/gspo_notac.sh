export WANDB_PROJECT=TAC

## With PM Reward
export WANDB_NAME="Qwen-7B_NOTAC_GSPO"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    gspo_NOTHIP.py \
    --dataset_name DeepMath-103k\
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct\
    --output_dir Result/Qwen-7B_NOTAC_GSPO \
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
    --save_total_limit 3 \
    --use_vllm True \
    --vllm_mode colocate \
    --save_steps 50 \
    --max_steps 240 \
    --per_device_train_batch_size 2 \
    --num_generations 8 \
    --gradient_accumulation_steps 4 \
    --importance_sampling_level sequence \


