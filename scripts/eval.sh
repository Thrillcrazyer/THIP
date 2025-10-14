# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
#     --base_model ./DeepSeek-R1-Distill-Qwen-1.5B/checkpoint-256 \
#     --chat_template_name r1-distill-qwen \
#     --system_prompt_name simplerl \
#     --output_dir ./eval_results_WO_THIP/256 \
#     --bf16 True \
#     --tensor_parallel_size 2 \
#     --data_id math-ai/aime25 \
#     --max_model_len 32768 \
#     --temperature 0.6 \
#     --top_p 0.95 \
#     --n 16 > eval_results/eval_log_WO_THIP.txt

# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
#     --base_model ./DeepSeek-R1-Distill-Qwen-1.5B_THIP/checkpoint-384 \
#     --chat_template_name r1-distill-qwen \
#     --system_prompt_name simplerl \
#     --output_dir ./eval_results_THIP/384 \
#     --bf16 True \
#     --tensor_parallel_size 2 \
#     --data_id math-ai/aime25 \
#     --max_model_len 32768 \
#     --temperature 0.6 \
#     --top_p 0.95 \
#     --n 16 > eval_results/eval_log_THIP.txt

# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
#     --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#     --chat_template_name r1-distill-qwen \
#     --system_prompt_name simplerl \
#     --output_dir ./eval_results/Distill \
#     --bf16 True \
#     --tensor_parallel_size 2 \
#     --data_id math-ai/aime25 \
#     --max_model_len 32768 \
#     --temperature 0.6 \
#     --top_p 0.95 \
#     --n 16 > eval_results/eval_log_Distill.txt

# VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
#     --base_model Qwen/Qwen2.5-1.5B-Instruct \
#     --chat_template_name r1-distill-qwen \
#     --system_prompt_name simplerl \
#     --output_dir ./eval_results/Org \
#     --bf16 True \
#     --tensor_parallel_size 2 \
#     --data_id math-ai/aime25 \
#     --max_model_len 32768 \
#     --temperature 0.6 \
#     --top_p 0.95 \
#     --n 16 > eval_results/eval_log_Org.txt

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model ./Qwen-1.5B_SFT/checkpoint-6500 \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/SFT \
    --bf16 True \
    --tensor_parallel_size 2 \
    --data_id math-ai/aime25 \
    --max_model_len 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 32 > log_SFT.txt

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model ./Result/Qwen-1.5B_THIP_GSPO_NOPMREWARD/checkpoint-3328 \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/GSPO \
    --bf16 True \
    --tensor_parallel_size 2 \
    --data_id math-ai/aime25 \
    --max_model_len 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 32 > log_GRPO.txt

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model ./Result/Qwen-1.5B_THIP_GSPO/checkpoint-640 \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/PM \
    --bf16 True \
    --tensor_parallel_size 2 \
    --data_id math-ai/aime25 \
    --max_model_len 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 32 > log_PM.txt



