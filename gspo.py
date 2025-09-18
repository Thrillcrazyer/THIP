import os

import torch
import re
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import torch
import pandas as pd
from datasets import Dataset
from reward import process_reward_func , answer_reward_func
from utils.chat_template import CHAT_TEMPLATE, SYSTEM_PROMPT, PREFIX_PROMPT, SUFFIX_PROMPT
import ray
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()


    ################
    # Model & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    # 1. Preprocess Dataset
    print("LOAD DEEPMATH DATA SET")
    
    deepmath=pd.read_csv('DeepMath-103k_id.csv')
    dataset = Dataset.from_pandas(deepmath)
    
    split_dataset = dataset.train_test_split(test_size=0.05)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print("LOAD Complete")

    sp = SYSTEM_PROMPT["simplerl"]

    def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": sp},
                    {"role": "user", "content": example['question']},
                ],
            }
    
    def make_sol_in_idx(example):
        return {
            "solution": {f"{example['r1_solution_1']} <index>{example['id']}</index>"}
        }
           
    print("MAPPING DATASET")
    train_dataset = train_dataset.map(make_conversation)
    train_dataset = train_dataset.map(make_sol_in_idx)

    eval_dataset = eval_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_sol_in_idx)
    print("DONE!!")
    
    ################
    # Reward Function for Training
    ################
    def split_solution_and_index(text):
        # <index>...</index> 부분 추출
        match = re.search(r'<index>(.*?)</index>', text)
        if match:
            index = match.group(1)
            # <index>...</index> 부분 제거한 나머지 텍스트
            solution = re.sub(r'<index>.*?</index>', '', text).strip()
        else:
            index = None
            solution = text.strip()
        return solution, index
    
    def split_think_and_answer(text):
        """
        ...</think>정답 형태의 문자열에서
        '생각'과 '정답'을 분리하여 반환합니다.
        """
        match = re.search(r'(.*?)</think>(.*)', text, re.DOTALL)
        if match:
            think = match.group(1).strip()
            answer = match.group(2).strip()
            return think, answer
        else:
            # 태그가 없을 경우 전체를 answer로 반환
            return text, "NO ANSWER"

    def accuracy_reward(completions, solution: list[str], **kwargs):
        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.
        """
        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        for content, sol in zip(contents, solution):
            sol, _ = split_solution_and_index(sol[0])
            try:
                gold_parsed = parse(sol, extraction_mode="first_match")
            except Exception:
                gold_parsed = []

            if len(gold_parsed) != 0:
                # Try parsing predicted answer too
                try:
                    answer_parsed = parse(
                        content,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed="all",
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception as e:
                    print(f"verify failed: {e}, answer: {content}, gold: {sol}")
                    reward = None
            else:
                # fallback to text match
                reward = float(content.strip().lower() == sol.strip().lower())

            rewards.append(reward)

        return rewards
    
    def accuracy_reward2(completions, solution: list[str], **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        
        @ray.remote
        def _compute_accuracy_score(content: str, sol_text: str) -> float:
            try:
                # extract clean solution and index
                sol, _ = split_solution_and_index(sol_text[0])
                _, ans = split_think_and_answer(content)
                accuracy_reward=float(answer_reward_func(sol, ans))
                print("ACCURACY REWARD: ",accuracy_reward)
                return accuracy_reward
            except Exception as e:
                print(f"Error processing content: {e}")
                return 0.0
        # Initialize Ray lazily if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Launch tasks in parallel and gather results
        futures = [_compute_accuracy_score.remote(content, sol) for content, sol in zip(contents, solution)]
        rewards = ray.get(futures)

        return rewards
                
        
    
    def format_reward(completions, solution: list[str], **kwargs):
        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        for content, sol in zip(contents, solution):
            if "</think>" in content:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def process_reward(completions, solution: list[str], **kwargs):
        # Ray-parallel version of the per-sample computation
        contents = [completion[0]["content"] for completion in completions]

        # Define a small remote task to compute confidence score for one sample
        @ray.remote
        def _compute_conf_score(content: str, sol_text: str) -> float:
            try:
                # extract clean solution and index
                _, index = split_solution_and_index(sol_text[0])
                think, _ = split_think_and_answer(content)
                process_reward=float(process_reward_func(think, int(index)))
                print("PROCESS REWARD: ",process_reward)
                return process_reward
            except Exception as e:
                print(f"Error processing content: {e}")
                return 0.0

        # Initialize Ray lazily if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Launch tasks in parallel and gather results
        futures = [_compute_conf_score.remote(content, sol) for content, sol in zip(contents, solution)]
        rewards = ray.get(futures)
        

        return rewards
    

    ################
    # Training
    ################
    training_args.report_to="wandb"
    training_args.use_liger_kernel=True
    training_args.attn_implementation="flash_attention_2"
    

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[format_reward, accuracy_reward, process_reward], #process_reward, accuracy_reward2
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()


    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)