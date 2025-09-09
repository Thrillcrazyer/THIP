VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/0 \
    --bf16 True \
    --tensor_parallel_size 2 \
    --data_id math-ai/aime25 \
    --max_model_len 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 16

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model ./DeepSeek-R1-Distill-Qwen-1.5B/checkpoint-256 \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/256 \
    --bf16 True \
    --tensor_parallel_size 2 \
    --data_id math-ai/aime25 \
    --max_model_len 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 16

VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python evaluate.py \
    --base_model ./DeepSeek-R1-Distill-Qwen-1.5B/checkpoint-512 \
    --chat_template_name r1-distill-qwen \
    --system_prompt_name simplerl \
    --output_dir ./eval_results/512 \
    --bf16 True \
    --tensor_parallel_size 2 \
    --data_id math-ai/aime25 \
    --max_model_len 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --n 16