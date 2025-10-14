import os
import re

import pandas as pd
import ray
import torch
from datasets import Dataset
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)

from reward import accuracy_reward
from utils.chat_template import SYSTEM_PROMPT, DEFAULT_PROMPT

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")
from trl.rewards import think_format_reward
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_conversation(example, sp=SYSTEM_PROMPT["simplerl"]):
    return {
        "prompt": [
            {"role": "system", "content": DEFAULT_PROMPT},
            {"role": "user", "content": example['question']},
        ],
    }

def make_sol_in_idx(example):
    return {
        "solution": [f"{example['final_answer']} <index>{example['id']}</index>"]
    }

def prepare_split(dataset: Dataset, system_prompt: str) -> Dataset:
    """Apply conversation and solution mapping for a dataset split."""
    mapped = dataset.map(lambda x: make_conversation(x, sp=system_prompt))
    return mapped.map(make_sol_in_idx)

def split_solution_and_index(text):
    match = re.search(r'<index>(.*?)</index>', text)
    if match:
        index = match.group(1)
        solution = re.sub(r'<index>.*?</index>', '', text).strip()
    else:
        index = None
        solution = text.strip()
    return solution, index

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # 1. Preprocess Dataset
    print("LOAD DEEPMATH DATA SET")
    
    deepmath=pd.read_csv('DeepMath-103k_id.csv')
    dataset = Dataset.from_pandas(deepmath)
    
    split_dataset = dataset.train_test_split(test_size=0.05)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print("LOAD Complete")

    sp = SYSTEM_PROMPT["simplerl"]
           
    print("MAPPING DATASET")
    train_dataset = prepare_split(train_dataset, sp)
    eval_dataset = prepare_split(eval_dataset, sp)
    print("MAPPING COMPLETE")
    
    # ################
    # # Training
    # ################

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[think_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)