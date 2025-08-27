import pandas as pd
import string
import json
import re
import math
from typing import Optional, Tuple,List
from openai import OpenAI
import yaml
import os
from dotenv import load_dotenv
##################################
from pm.miner import Miner
from pm.checker import Checker
import reward
import pm



truelogs=pd.read_csv('eventlogs/DeepMath_eventlog.csv')

def load_template(yaml_file='./reward/prompt.yaml'):
    with open(yaml_file, 'r', encoding='utf-8') as file:
        template_data = yaml.safe_load(file)
    return template_data['answer_prompt']

def process_reward_func(think:str, index:int)->float:
    think_log = reward.Answer2EventAgent().make_event_log(think)
    think_log.log['Case ID'] = str(index)
    reason_net = Miner(think_log).discover()
    
    true_log_df = truelogs[truelogs['Case ID'] == index].copy()
    true_log_df['Case ID'] = str(index)
    true_eventlog = pm.EventLog(true_log_df)

    conf_df = Checker(true_eventlog, reason_net).check()

    return conf_df['F1 Score'].values[0]

def answer_reward_func(ans:str, true:str,model_name="deepseek-chat")->float:
    load_dotenv()
    api_key=os.getenv("DEEPSEEK_KEY")
    client=OpenAI(api_key=api_key,base_url="https://api.deepseek.com")
    template=load_template()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": template.format(prediction=ans, gold=true)}
        ]
    )
    
    return response.choices[0].message.content

_PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    s = s.translate(_PUNCT_TABLE)
    s = " ".join(s.split())
    return s


def _token_f1(a: str, b: str) -> float:
    a_tok = _normalize_text(a).split()
    b_tok = _normalize_text(b).split()
    if not a_tok and not b_tok:
        return 1.0
    if not a_tok or not b_tok:
        return 0.0
    from collections import Counter

    ca, cb = Counter(a_tok), Counter(b_tok)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(ca.values()))
    recall = overlap / max(1, sum(cb.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _extract_numbers(s: str) -> List[float]:
    import re

    s = s or ""
    nums = []
    for m in re.finditer(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s):
        try:
            nums.append(float(m.group(0)))
        except Exception:
            continue
    return nums


def answer_similarity_score(pred: str, gold: str) -> float:
    # number-aware quick match
    pnums, gnums = _extract_numbers(pred), _extract_numbers(gold)
    if pnums and gnums:
        for pn in pnums:
            for gn in gnums:
                if math.isfinite(pn) and math.isfinite(gn):
                    if abs(pn - gn) <= 1e-6 * max(1.0, abs(gn)):
                        return 1.0
    # fallback to token F1
    return _token_f1(pred, gold)