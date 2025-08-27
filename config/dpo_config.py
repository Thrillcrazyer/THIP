from dataclasses import dataclass

@dataclass
class DPOConfig:
    model_name: str = "Qwen/Qwen3-4B-Thinking-2507"
    lr: float = 5e-6
    epochs: int = 1
    batch_size: int = 1
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