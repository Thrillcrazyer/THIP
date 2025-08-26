import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import reward_4_process_func
from utils import get_response


def main(model_name="Qwen/Qwen3-4B-Thinking-2507"):
    #Initialization Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    deepmath=pd.read_csv('DeepMath-103k.csv')
    
    for index, row in deepmath.iterrows():
        think, ans = get_response(model, tokenizer, row['question'])
        
        conf_score = reward_4_process_func(think, index)

        print("ANS:", ans)
        print("Conf Score:", conf_score)
        breakpoint()


    
    

if __name__ == "__main__":
    main()