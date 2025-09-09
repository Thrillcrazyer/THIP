import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import answer_reward_func
from utils import get_response
import re

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
            return text.strip(), " "
        
def main(model_name="Qwen/Qwen3-4B-Thinking-2507"):
    #Initialization Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="cuda:1",
        attn_implementation="flash_attention_2",
    )
    
    deepmath=pd.read_csv('DeepMath-103k.csv')
    print(len(deepmath))
    
    for index, row in deepmath.iterrows():
        think, ans = get_response(model, tokenizer, row['question'])

        ans_score = answer_reward_func(ans, row['final_answer'])
        print(f"Golden Answer: {row['final_answer']},\n Answer: {ans},\n Answer_Score: {ans_score}")

        # conf_score = process_reward_func(think, index)
        # print(f"Conf Score: {conf_score}")
        if index==5:
            break

if __name__ == "__main__":
    #main()
    think,answer=split_think_and_answer("안녕하세요 올리버쌤입니다..")
    
    print(f"think: {think}, answer: {answer}")