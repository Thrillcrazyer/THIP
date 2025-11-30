export WANDB_PROJECT=THIP_COMPARE_QWEN7

# ## THIP
# export WANDB_NAME="QWEN7_THIP"

# accelerate launch \
#     --config_file configs/deepspeed_zero3.yaml \
#     gspo_THIP.py \
#     --dataset_name DeepMath-103k\
#     --model_name_or_path Qwen/Qwen2.5-7B-Instruct\
#     --output_dir Result/QWEN7_THIP \
#     --logging_dir ./logs \
#     --report_to wandb \
#     --run_name $WANDB_NAME \
#     --save_total_limit 3 \
#     --push_to_hub True \
#     --save_safetensors True \
#     --learning_rate 1e-6 \
#     --torch_dtype bfloat16 \
#     --max_prompt_length 1024 \
#     --max_completion_length 8192 \
#     --use_vllm True \
#     --vllm_mode colocate \
#     --log_completions True \
#     --save_strategy steps \
#     --save_steps 100 \
#     --max_steps 1000 \
#     --per_device_train_batch_size 4 \
#     --num_generations 8 \
#     --importance_sampling_level sequence \
#     --attn_implementation=flash_attention_2\
#     #--loss_type dr_grpo \

# ## GSPO
# export WANDB_NAME="QWEN7_GSPO"

# accelerate launch \
#     --config_file configs/deepspeed_zero3.yaml \
#     gspo_NOTHIP.py \
#     --dataset_name DeepMath-103k\
#     --model_name_or_path Qwen/Qwen2.5-7B-Instruct\
#     --output_dir Result/QWEN7_GSPO \
#     --logging_dir ./logs \
#     --report_to wandb \
#     --save_total_limit 3 \
#     --run_name $WANDB_NAME \
#     --push_to_hub True \
#     --save_safetensors True \
#     --learning_rate 1e-6 \
#     --torch_dtype bfloat16 \
#     --max_prompt_length 1024 \
#     --max_completion_length 8192 \
#     --use_vllm True \
#     --vllm_mode colocate \
#     --log_completions True \
#     --save_strategy steps \
#     --save_steps 100 \
#     --max_steps 1000 \
#     --per_device_train_batch_size 4 \
#     --num_generations 8 \
#     --importance_sampling_level sequence \
#     --attn_implementation=flash_attention_2\

# ## GRPO
# export WANDB_NAME="QWEN7_GRPO"

# accelerate launch \
#     --config_file configs/deepspeed_zero3.yaml \
#     gspo_NOTHIP.py \
#     --dataset_name DeepMath-103k \
#     --model_name_or_path Qwen/Qwen2.5-7B-Instruct\
#     --output_dir Result/QWEN7_GRPO \
#     --logging_dir ./logs \
#     --report_to wandb \
#     --run_name $WANDB_NAME \
#     --push_to_hub True \
#     --save_total_limit 2 \
#     --save_safetensors True \
#     --learning_rate 1e-6 \
#     --torch_dtype bfloat16 \
#     --max_prompt_length 1024 \
#     --max_completion_length 8192 \
#     --use_vllm True \
#     --vllm_mode colocate \
#     --log_completions True \
#     --save_strategy steps \
#     --save_steps 100 \
#     --max_steps 1000 \
#     --per_device_train_batch_size 4 \
#     --num_generations 8 \
#     --attn_implementation=flash_attention_2\
    
## DRGRPO
export WANDB_NAME="QWEN7_DRGRPO"

accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    gspo_NOTHIP.py \
    --dataset_name DeepMath-103k \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct\
    --output_dir Result/QWEN7_DRGRPO \
    --logging_dir ./logs \
    --report_to wandb \
    --run_name $WANDB_NAME \
    --push_to_hub True \
    --save_safetensors True \
    --save_total_limit 2 \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --max_prompt_length 1024 \
    --max_completion_length 8192 \
    --use_vllm True \
    --vllm_mode colocate \
    --log_completions True \
    --save_strategy steps \
    --save_steps 100 \
    --max_steps 1000 \
    --per_device_train_batch_size 4 \
    --num_generations 8 \
    --loss_type dr_grpo \
    --attn_implementation=flash_attention_2 \


