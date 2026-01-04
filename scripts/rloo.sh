export WANDB_PROJECT=THIP

export WANDB_NAME="Qwen-1.5B_THIP_RLLO_1125"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    RLOO.py \
    --dataset_name DeepMath-103k\
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\
    --output_dir Result/Qwen-1.5B_THIP_RLLO_0102 \
    --logging_dir ./logs \
    --report_to wandb \
    --run_name $WANDB_NAME \
    --push_to_hub False \
    --bf16 True \
    --use_liger_kernel True \
    --save_safetensors True \
    --learning_rate 1e-6 \
    --max_prompt_length 512 \
    --max_completion_length 4096 \
    --log_completions True \
    --save_strategy steps \
    --save_steps 100 \
    --max_steps 1000 \
    --per_device_train_batch_size 4 \