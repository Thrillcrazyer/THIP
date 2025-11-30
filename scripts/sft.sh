export WANDB_PROJECT=QWEN7_SFT
export WANDB_NAME="QWEN7_SFT-$(date +%Y%m%d-%H%M%S)"

accelerate launch \
  --config_file configs/deepspeed_zero3.yaml \
  sft.py \
  --config configs/sft_full.yaml \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --packing True \
  --packing_strategy wrapped \
  --output_dir Result/Qwen-7B_SFT \
  --run_name "$WANDB_NAME" \
  --use_liger_kernel True \
  --push_to_hub True \
  --save_total_limit 2 \
  --save_steps 500 \
  --per_device_train_batch_size 16