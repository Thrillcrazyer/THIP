export WANDB_PROJECT=Qwen-1.5B-SFT
export WANDB_NAME="Qwen-1.5B_SFT-$(date +%Y%m%d-%H%M%S)"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    sft.py \
    --config configs/sft_full.yaml \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --packing true packing_strategy wrapped \
    --output_dir Result/Qwen-1.5B_SFT \
    --run_name $WANDB_NAME \
    --use_liger_kernel True \
    --push_to_hub True \
    --save_steps 500 \
    --use_vllm True \
    --vllm_mode colocate \
    --per_device_train_batch_size 16 \