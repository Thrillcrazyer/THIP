from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from datasets import load_dataset

from utils.chat_template import CHAT_TEMPLATE, SYSTEM_PROMPT, PREFIX_PROMPT, SUFFIX_PROMPT

load_dotenv()

DATASET_INFO = {
    "zwhe99/MATH": {
        "default_split": "math500",
        "problem_key": "problem",
        "answer_key": "expected_answer",
        "category_keys": ["level", "type"]
    },
    "zwhe99/aime90": {
        "default_split": "2024",
        "problem_key": "problem",
        "answer_key": "expected_answer",
    },
    "zwhe99/amc23": {
        "default_split": "test",
        "problem_key": "question",
        "answer_key": "answer",
    },
    "zwhe99/simplerl-minerva-math": {
        "default_split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
    "math-ai/aime25": {
        "default_split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
    "math-ai/aime24": {
        "default_split": "test",
        "problem_key": "problem",
        "answer_key": "solution",
    },
    "zwhe99/simplerl-OlympiadBench": {
        "default_split": "test",
        "problem_key": "question",
        "answer_key": "final_answer",
    },
    "zwhe99/gpqa_diamond_mc": {
        "default_split": "test",
        "problem_key": "problem",
        "answer_key": "solution",
        "category_keys": ["domain"]
    },
    "zwhe99/pm-en": {
        "default_split": "test",
        "problem_key": "question",
        "answer_key": "answer",
        "category_keys": ["level"]
    }
}



def make_prompt(problem:str)->list:
    messages = []
    messages.append({"role": "user", "content": problem})
    return messages


def main(datasets:list):
    # Prepare LLM
    api_key = os.getenv("DEEPSEEK_KEY")
    client=OpenAI(api_key=api_key,base_url="https://api.deepseek.com")
    
    
    for data_id in datasets:
        problem_key = DATASET_INFO[data_id]["problem_key"]
        answer_key = DATASET_INFO[data_id]["answer_key"]
        split = DATASET_INFO[data_id].get("default_split", "test")
        
        for sample in load_dataset(data_id, split=split):
            problem = sample[problem_key]
            print(f"Problem: {problem}")
            messages = make_prompt(problem)
            
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages
            )
            
            generated_answer = response.choices[0].message.content
            
            print(f"Problem: {problem}")
            print(f"Generated Answer: {generated_answer}")
            print("-" * 50)
            breakpoint()
        

if __name__ == "__main__":
    main(["zwhe99/aime90","math-ai/aime25","zwhe99/simplerl-minerva-math"])    
