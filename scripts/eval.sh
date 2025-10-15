TEST_NAME='zwhe99/MATH'

# PM Reward
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model ./Result/Qwen-1.5B_THIP/checkpoint-500 \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/PM \
    --bf16 True \
    --tensor_parallel_size 4 \
    --data_id $TEST_NAME \
    --max_model_len 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 32 > log_PM.txt

# SFT
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model ./Result/Qwen-1.5B_SFT/checkpoint-500 \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/SFT \
    --bf16 True \
    --tensor_parallel_size 4 \
    --data_id $TEST_NAME \
    --max_model_len 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 32 > log_SFT.txt

#GSPO
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model ./Result/Qwen-1.5B_GSPO/checkpoint-500 \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/GSPO \
    --bf16 True \
    --tensor_parallel_size 4 \
    --data_id $TEST_NAME \
    --max_model_len 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 32 > log_GSPO.txt

#GRPO
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model ./Result/Qwen-1.5B_GRPO/checkpoint-500 \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/GRPO \
    --bf16 True \
    --tensor_parallel_size 4 \
    --data_id $TEST_NAME \
    --max_model_len 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 32 > log_GRPO.txt

#DRGRPO
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model ./Result/Qwen-1.5B_THIP_DRGRPO/checkpoint-500 \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/DRGRPO \
    --bf16 True \
    --tensor_parallel_size 4 \
    --data_id $TEST_NAME \
    --max_model_len 16384 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 32 > log_DRGRPO.txt



