import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import process_reward_func,answer_reward_func,answer_similarity_score,accuracy_reward
from utils import get_response


def main(model_name="Qwen/Qwen3-4B-Thinking-2507"):
    #Initialization Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="bfloat16",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    
    deepmath=pd.read_csv('DeepMath-103k.csv')
    print(len(deepmath))
    
    for index, row in deepmath.iterrows():
        think, ans = get_response(model, tokenizer, row['question'])

        ans_score = accuracy_reward(ans, row['final_answer'])
        print(f"Golden Answer: {row['final_answer']},\n Answer: {ans},\n Answer_Score: {ans_score}")

        ans_score2=answer_similarity_score(ans,row['final_answer'])
        print(f"ANS SCORE using Similarity: {ans_score2}")
        print(f"ANS SCORE using Library: {ans_score}")
        # conf_score = process_reward_func(think, index)
        # print(f"Conf Score: {conf_score}")
        if index==5:
            break

if __name__ == "__main__":
    main()