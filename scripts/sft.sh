export WANDB_PROJECT=Qwen-1.5B-SFT
export WANDB_NAME="Qwen-1.5B_SFT-$(date +%Y%m%d-%H%M%S)"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    sft.py \
    --config configs/sft_full.yaml \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --packing true packing_strategy wrapped \
    --run_name $WANDB_NAME \
    --use_liger_kernel True \
    --attn_implementation= 'flash_attention_2'