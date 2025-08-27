
import transformers
from typing import Tuple, List
import torch

@torch.no_grad()
def get_response(
    model:transformers.AutoModelForCausalLM,
    tokenizer:transformers.AutoTokenizer,
    prompt:str)-> Tuple[str, str]:
    messages = [
    {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    return thinking_content, content
    
@torch.no_grad()
def sample_responses(
    model: transformers.AutoModelForCausalLM,
    tokenizer: transformers.AutoTokenizer,
    prompt: str,
    num_samples: int = 2,
    max_new_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Tuple[str, str]]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(model.parameters()).device
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated = model.generate(
        **model_inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_samples,
        max_new_tokens=max_new_tokens,
    )

    results: List[Tuple[str, str]] = []
    for i in range(num_samples):
        output_ids = generated[i][len(model_inputs.input_ids[0]) :].tolist()
        # try to split at </think> like utils.get_response
        try:
            idx = len(output_ids) - output_ids[::-1].index(151668)  # </think>
        except ValueError:
            idx = -1
        thinking = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip("\n")
        answer = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")
        results.append((thinking, answer))
    return results