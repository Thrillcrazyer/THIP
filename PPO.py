import os
import argparse
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from reward import process_reward, accuracy_reward_old, think_format_reward
from utils.chat_template import DEFAULT_PROMPT
from utils.utils import prepare_split


def compute_rewards(
    responses: list[str],
    solutions: list[str],
    w_accuracy: float = 1.0,
    w_process: float = 1.0,
    w_think_format: float = 1.0,
) -> list[torch.Tensor]:
    """Compute rewards for generated responses."""
    completions = [[{"content": r}] for r in responses]
    
    rewards = []
    for i in range(len(responses)):
        total = 0.0
        if w_accuracy != 0.0:
            acc = accuracy_reward_old(completions=[completions[i]], solution=[solutions[i]])
            total += w_accuracy * acc[0]
        if w_process != 0.0:
            proc = process_reward(completions=[completions[i]], solution=[solutions[i]])
            total += w_process * proc[0]
        if w_think_format != 0.0:
            fmt = think_format_reward(completions=[completions[i]], solution=[solutions[i]])
            total += w_think_format * fmt[0]
        rewards.append(torch.tensor(total))
    
    return rewards


def build_dataset(dataset: Dataset, tokenizer, max_prompt_length: int) -> Dataset:
    """Build dataset for PPO training."""
    def tokenize(sample):
        prompt_messages = sample["prompt"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        sample["query"] = prompt_text
        sample["input_ids"] = tokenizer.encode(
            prompt_text, 
            truncation=True, 
            max_length=max_prompt_length,
            padding="max_length",
        )
        return sample
    
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")
    return dataset


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def parse_args():
    parser = argparse.ArgumentParser(description="PPO Training")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./Result/Qwen-1.5B_PPO")
    parser.add_argument("--dataset_name", type=str, default="DeepMath-103k")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_response_length", type=int, default=512)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--bf16", type=str, default="True")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--run_name", type=str, default="ppo_run")
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--push_to_hub", type=str, default="False")
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_safetensors", type=str, default="True")
    parser.add_argument("--w_accuracy", type=float, default=1.0)
    parser.add_argument("--w_process", type=float, default=1.0)
    parser.add_argument("--w_think_format", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    
    args = parse_args()

    # ============================================
    # PPO Config
    # ============================================
    ppo_config = PPOConfig(
        model_name=args.model_name_or_path,
        learning_rate=args.learning_rate,
        batch_size=args.per_device_train_batch_size * args.gradient_accumulation_steps,
        mini_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        max_grad_norm=1.0,
        optimize_cuda_cache=True,
        log_with=args.report_to if args.report_to != "none" else None,
    )
    
    # Generation kwargs
    generation_kwargs = {
        "max_new_tokens": args.max_response_length,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    # ============================================
    # Load Dataset
    # ============================================
    print("LOAD DEEPMATH DATA SET")
    deepmath = pd.read_csv("DeepMath-103k_id.csv")
    dataset = Dataset.from_pandas(deepmath)
    split_dataset = dataset.train_test_split(test_size=0.05)
    train_dataset = split_dataset["train"]
    print("LOAD Complete")

    print("MAPPING DATASET")
    train_dataset = prepare_split(train_dataset, DEFAULT_PROMPT)
    print("MAPPING COMPLETE")

    # ============================================
    # Load Model & Tokenizer
    # ============================================
    print("LOAD MODEL & TOKENIZER")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with value head for PPO
    torch_dtype = torch.bfloat16 if args.bf16.lower() == "true" else torch.float32
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    # Optional: Load reference model (set to None to save memory)
    ref_model = None

    # ============================================
    # Build Dataset
    # ============================================
    print("TOKENIZING DATASET")
    train_dataset = build_dataset(train_dataset, tokenizer, args.max_prompt_length)

    # ============================================
    # Initialize PPO Trainer
    # ============================================
    print("INITIALIZE PPO TRAINER")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        data_collator=collator,
    )

    # Set pad token id for generation
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    # ============================================
    # Training Loop
    # ============================================
    print(f"\n{'='*60}")
    print(f"=== Starting PPO Training ===")
    print(f"Model: {args.model_name_or_path}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {ppo_config.batch_size}")
    print(f"Mini batch size: {ppo_config.mini_batch_size}")
    print(f"PPO epochs: {ppo_config.ppo_epochs}")
    print(f"Max steps: {args.max_steps}")
    print(f"{'='*60}\n")

    for step, batch in tqdm(enumerate(ppo_trainer.dataloader), desc="PPO Training", total=args.max_steps):
        if step >= args.max_steps:
            break
            
        query_tensors = batch["input_ids"]
        solutions = batch["solution"]
        
        # ============================================
        # Step 1: Generate responses
        # ============================================
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **generation_kwargs
        )
        
        # Decode responses
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        
        # ============================================
        # Step 2: Compute rewards
        # ============================================
        rewards = compute_rewards(
            responses=batch["response"],
            solutions=solutions,
            w_accuracy=args.w_accuracy,
            w_process=args.w_process,
            w_think_format=args.w_think_format,
        )
        
        # ============================================
        # Step 3: Run PPO step
        # ============================================
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        
        # Log progress
        if step % 10 == 0:
            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            print(f"\nStep {step}: Mean Reward = {mean_reward:.4f}")
        
        # Save checkpoint periodically
        if step > 0 and step % args.save_steps == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            ppo_trainer.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved at {checkpoint_dir}")

    # ============================================
    # Save Final Model
    # ============================================
    print(f"\n{'='*60}")
    print("=== Saving Final Model ===")
    print(f"{'='*60}\n")
    os.makedirs(args.output_dir, exist_ok=True)
    ppo_trainer.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[DONE] Final model saved at {args.output_dir}")
