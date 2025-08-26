import os
import math
import random
import string
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward import process_reward_func

# Optional Ray support for parallel reward evaluation
try:
    import ray  # type: ignore
except Exception:
    ray = None  # fallback to sequential

# Optional Weights & Biases support for live metrics
try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # disable logging if not available


# -----------------------------
# Utility: answer normalization and similarity (0..1)
# -----------------------------
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


# -----------------------------
# Generation helper: sample multiple responses (think, answer)
# -----------------------------
@torch.no_grad()
def sample_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_samples: int = 2,
    max_new_tokens: int = 512,
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
            idx = 0
        thinking = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip("\n")
        answer = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip("\n")
        results.append((thinking, answer))
    return results


# -----------------------------
# Preference dataset construction
# -----------------------------
def build_preference_pairs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    alpha: float = 0.5,
    num_candidates: int = 2,
    max_rows: int | None = 128,
    seed: int = 42,
    ray_batch_size: int | None = None,
) -> List[Dict[str, str]]:
    random.seed(seed)
    pairs: List[Dict[str, str]] = []

    rows = list(df.iterrows())
    if max_rows is not None:
        rows = rows[: max_rows]

    # helper: batched conf_score evaluation (ray or sequential)
    def eval_conf_scores(think_list: List[str], case_id_list: List[str]) -> List[float]:
        assert len(think_list) == len(case_id_list)
        n = len(think_list)
        if n == 0:
            return []
        bsz = ray_batch_size or n
        # sequential fallback
        if ray is None:
            out: List[float] = []
            for t, cid in zip(think_list, case_id_list):
                try:
                    out.append(float(process_reward_func(t, cid)))
                except Exception:
                    out.append(0.0)
            return out

        # ray path
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)

        @ray.remote
        def worker(think: str, case_id: str) -> float:
            try:
                import reward  # ensure module inside worker
                return float(reward.process_reward_func(think, case_id))
            except Exception:
                return 0.0

        out_scores: List[float] = []
        # process in chunks to limit outstanding tasks to bsz
        for i in range(0, n, bsz):
            chunk_thinks = think_list[i : i + bsz]
            chunk_cases = case_id_list[i : i + bsz]
            futures = [worker.remote(t, c) for t, c in zip(chunk_thinks, chunk_cases)]
            out_scores.extend(ray.get(futures))
        return out_scores

    for idx, row in rows:
        prompt = row.get("question", None)
        gold = row.get("final_answer", None)
        if not isinstance(prompt, str) or not isinstance(gold, str):
            continue

        candidates = sample_responses(model, tokenizer, prompt, num_samples=num_candidates)

        # compute conf_scores in parallel for this row's candidates
        if candidates:
            thinks = [t for (t, _a) in candidates]
            case_ids = [str(idx)] * len(thinks)
            conf_list = eval_conf_scores(thinks, case_ids)
        else:
            conf_list = []

        # keep detailed components for logging
        scored: List[Tuple[float, str, str, float, float]] = []  # (score, think, ans, conf, sim)
        for (think, ans), conf in zip(candidates, conf_list):
            sim = answer_similarity_score(ans, gold)
            score = alpha * float(conf) + (1.0 - alpha) * sim
            scored.append((score, think, ans, float(conf), float(sim)))

        if len(scored) < 2:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen_score, chosen_think, chosen_ans, chosen_conf, chosen_sim = scored[0]
        rejected_score, rejected_think, rejected_ans, rejected_conf, rejected_sim = scored[-1]

        pairs.append({
            "prompt": prompt,
            "chosen": chosen_ans,
            "rejected": rejected_ans,
            # optional logging fields
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "chosen_conf": chosen_conf,
            "rejected_conf": rejected_conf,
            "chosen_sim": chosen_sim,
            "rejected_sim": rejected_sim,
        })

    return pairs


# -----------------------------
# Dataset class
# -----------------------------
class PreferencePairsDataset(Dataset):
    def __init__(self, items: List[Dict[str, str]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.items[idx]


# -----------------------------
# Log-prob computation for completions only
# -----------------------------
def sequence_logprob(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
) -> float:
    device = next(model.parameters()).device
    with torch.no_grad():
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
        comp_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).to(device)

        input_ids = torch.cat([prompt_ids.input_ids, comp_ids.input_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, T, V]

        # shift
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        prompt_len = prompt_ids.input_ids.size(1)
        comp_len = comp_ids.input_ids.size(1)

        # mask positions that belong to completion
        mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        start = prompt_len
        end = prompt_len + comp_len
        mask[:, start:end] = True

        log_probs = F.log_softmax(shift_logits, dim=-1)
        gather = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        selected = gather.masked_select(mask)
        return selected.sum().item()


def sequence_logprob_policy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
) -> torch.Tensor:
    """Like sequence_logprob, but returns a differentiable scalar tensor for the policy."""
    device = next(model.parameters()).device
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )

    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    comp_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).to(device)

    input_ids = torch.cat([prompt_ids.input_ids, comp_ids.input_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [1, T, V]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    prompt_len = prompt_ids.input_ids.size(1)
    comp_len = comp_ids.input_ids.size(1)

    mask = torch.zeros_like(shift_labels, dtype=torch.bool)
    start = prompt_len
    end = prompt_len + comp_len
    mask[:, start:end] = True

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gather = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    selected = gather.masked_select(mask)
    return selected.sum()  # differentiable scalar


# -----------------------------
# DPO training loop
# -----------------------------
@dataclass
class DPOConfig:
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    lr: float = 5e-6
    epochs: int = 1
    batch_size: int = 128
    beta: float = 0.1
    alpha: float = 0.5
    num_candidates: int = 2
    max_rows: int | None = 64
    seed: int = 42
    out_dir: str = "./dpo_out"
    # batch size for ray parallel reward evaluation (defaults to batch_size if None)
    ray_batch_size: int | None = None
    # wandb logging
    wandb_enabled: bool = True
    wandb_project: str | None = "THIP"
    wandb_run_name: str | None = None
    wandb_mode: str | None = None  # e.g., "offline"


def dpo_loss(
    policy_pos: torch.Tensor,
    policy_neg: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_neg: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    diff = (policy_pos - policy_neg) - (ref_pos - ref_neg)
    return -F.logsigmoid(beta * diff).mean()


def train_dpo(cfg: DPOConfig):
    torch.manual_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    policy = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype="auto")
    policy.train()

    ref = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype="auto")
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # Initialize wandb
    run = None
    if cfg.wandb_enabled and wandb is not None and cfg.wandb_project:
        try:
            init_kwargs = {"project": cfg.wandb_project, "config": {k: getattr(cfg, k) for k in cfg.__annotations__.keys()}}
            if cfg.wandb_run_name:
                init_kwargs["name"] = cfg.wandb_run_name
            if cfg.wandb_mode:
                init_kwargs["mode"] = cfg.wandb_mode
            run = wandb.init(**init_kwargs)
        except Exception:
            run = None

    # Multi-GPU placement: prefer using two GPUs (policy on cuda:0, ref on cuda:1)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus >= 2:
            policy.to("cuda:0")
            ref.to("cuda:1")
        else:
            # single GPU fallback
            policy.to("cuda:0")
            ref.to("cuda:0")

    deepmath = pd.read_csv("DeepMath-103k.csv")
    pairs = build_preference_pairs(
        model=policy,
        tokenizer=tokenizer,
        df=deepmath,
        alpha=cfg.alpha,
        num_candidates=cfg.num_candidates,
        max_rows=cfg.max_rows,
        seed=cfg.seed,
        ray_batch_size=cfg.ray_batch_size or cfg.batch_size,
    )
    if not pairs:
        raise RuntimeError("No preference pairs could be constructed.")

    # optional: shutdown ray to free resources (pairs already materialized)
    try:
        if ray is not None and ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    ds = PreferencePairsDataset(pairs)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)

    global_step = 0
    for epoch in range(cfg.epochs):
        for batch in dl:
            prompts: List[str] = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
            chosens: List[str] = batch["chosen"] if isinstance(batch["chosen"], list) else [batch["chosen"]]
            rejects: List[str] = batch["rejected"] if isinstance(batch["rejected"], list) else [batch["rejected"]]

            # policy logprobs with gradients
            pol_pos_t_list: List[torch.Tensor] = []
            pol_neg_t_list: List[torch.Tensor] = []
            for x, c, r in zip(prompts, chosens, rejects):
                pol_pos_t_list.append(sequence_logprob_policy(policy, tokenizer, x, c))
                pol_neg_t_list.append(sequence_logprob_policy(policy, tokenizer, x, r))
            pol_pos_t = torch.stack(pol_pos_t_list).to(next(policy.parameters()).device)
            pol_neg_t = torch.stack(pol_neg_t_list).to(next(policy.parameters()).device)

            # reference logprobs without gradients
            ref_pos, ref_neg = [], []
            with torch.no_grad():
                for x, c, r in zip(prompts, chosens, rejects):
                    ref_pos.append(sequence_logprob(ref, tokenizer, x, c))
                    ref_neg.append(sequence_logprob(ref, tokenizer, x, r))
            policy_device = next(policy.parameters()).device
            ref_pos_t = torch.tensor(ref_pos, device=policy_device)
            ref_neg_t = torch.tensor(ref_neg, device=policy_device)

            loss = dpo_loss(pol_pos_t, pol_neg_t, ref_pos_t, ref_neg_t, cfg.beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            if global_step % 10 == 0:
                print(f"[epoch {epoch+1}] step {global_step} | loss {loss.item():.4f} | pairs {len(ds)}")

            # wandb logging per step
            if run is not None:
                with torch.no_grad():
                    delta_policy = (pol_pos_t - pol_neg_t).mean().item()
                    delta_ref = (ref_pos_t - ref_neg_t).mean().item()
                    advantage = (pol_pos_t - pol_neg_t - (ref_pos_t - ref_neg_t)).mean().item()
                    # optional: average components from pairs if available
                    def _avg(field: str):
                        try:
                            vals = [float(x) for x in batch.get(field, [])]
                            if not vals:
                                return None
                            return sum(vals) / len(vals)
                        except Exception:
                            return None
                    avg_chosen_score = _avg("chosen_score")
                    avg_rejected_score = _avg("rejected_score")
                    avg_chosen_conf = _avg("chosen_conf")
                    avg_rejected_conf = _avg("rejected_conf")
                    avg_chosen_sim = _avg("chosen_sim")
                    avg_rejected_sim = _avg("rejected_sim")
                mem0 = mem1 = None
                if torch.cuda.is_available():
                    try:
                        mem0 = torch.cuda.memory_allocated(0)
                        if torch.cuda.device_count() > 1:
                            mem1 = torch.cuda.memory_allocated(1)
                    except Exception:
                        pass
                log_data = {
                    "train/loss": float(loss.item()),
                    "train/grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
                    "dpo/delta_policy_mean": float(delta_policy),
                    "dpo/delta_ref_mean": float(delta_ref),
                    "dpo/advantage_mean": float(advantage),
                    "step": int(global_step),
                    "epoch": int(epoch + 1),
                }
                if avg_chosen_score is not None:
                    log_data["pairs/avg_chosen_score"] = float(avg_chosen_score)
                if avg_rejected_score is not None:
                    log_data["pairs/avg_rejected_score"] = float(avg_rejected_score)
                if avg_chosen_conf is not None:
                    log_data["pairs/avg_chosen_conf"] = float(avg_chosen_conf)
                if avg_rejected_conf is not None:
                    log_data["pairs/avg_rejected_conf"] = float(avg_rejected_conf)
                if avg_chosen_sim is not None:
                    log_data["pairs/avg_chosen_sim"] = float(avg_chosen_sim)
                if avg_rejected_sim is not None:
                    log_data["pairs/avg_rejected_sim"] = float(avg_rejected_sim)
                if mem0 is not None:
                    log_data["gpu0/mem_alloc_bytes"] = int(mem0)
                if mem1 is not None:
                    log_data["gpu1/mem_alloc_bytes"] = int(mem1)
                try:
                    wandb.log(log_data, step=global_step)
                except Exception:
                    pass

    os.makedirs(cfg.out_dir, exist_ok=True)
    policy.save_pretrained(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    print(f"[done] saved policy to {cfg.out_dir}")

    # Finish wandb
    if run is not None:
        try:
            wandb.finish()
        except Exception:
            pass


def main(
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507",
    epochs: int = 1,
    lr: float = 5e-6,
    alpha: float = 0.5,
    beta: float = 0.1,
    num_candidates: int = 2,
    max_rows: int | None = 64,
    batch_size: int = 1,
    out_dir: str = "./dpo_out",
    ray_batch_size: int | None = None,
    wandb_enabled: bool = True,
    wandb_project: str | None = "THIP",
    wandb_run_name: str | None = None,
    wandb_mode: str | None = None,
):
    cfg = DPOConfig(
        model_name=model_name,
        epochs=epochs,
        lr=lr,
        alpha=alpha,
        beta=beta,
        num_candidates=num_candidates,
        max_rows=max_rows,
        batch_size=batch_size,
        out_dir=out_dir,
        ray_batch_size=ray_batch_size,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_mode=wandb_mode,
    )
    train_dpo(cfg)


if __name__ == "__main__":
    main()