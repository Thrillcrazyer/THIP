import os
import re
from datasets import Dataset

try:
    from .chat_template import SYSTEM_PROMPT, DEFAULT_PROMPT
except ImportError:  # Fallback when utils is on sys.path root
    from utils.chat_template import SYSTEM_PROMPT, DEFAULT_PROMPT

def make_conversation(example, sp=SYSTEM_PROMPT["simplerl"]):
    return {
        "prompt": [
            {"role": "system", "content": sp},
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
