"""
Supervised Fine-Tuning (SFT) script for training language models.

This module provides functionality to fine-tune causal language models using LoRA
on mathematical reasoning datasets with supervised learning.
"""
import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from utils.chat_template import SYSTEM_PROMPT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfiguration:
    """Configuration for the SFT training process."""
    
    # Model settings
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Dataset settings
    dataset_path: str = "DeepMath-103k_id.csv"
    test_split_ratio: float = 0.05
    random_seed: int = 42
    
    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    # Training hyperparameters
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 10
    warmup_ratio: float = 0.03
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 256
    eval_steps: int = 256
    
    # System prompt type
    system_prompt_type: str = "simplerl"
    
    def __post_init__(self):
        """Initialize default LoRA target modules."""
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj"
            ]


class EnvironmentConfig:
    """Handles environment variable configuration."""
    
    @staticmethod
    def get_output_dir() -> str:
        """Get the output directory from environment or use default."""
        return os.environ.get(
            "SFT_OUTPUT", 
            "Result/DeepSeek-R1-Distill-Qwen-1.5B_SFT"
        )
    
    @staticmethod
    def get_max_samples() -> Optional[int]:
        """Get max samples for debugging from environment."""
        debug_n = os.environ.get("SFT_MAX_SAMPLES")
        if debug_n is not None:
            try:
                n = int(debug_n)
                return n if n > 0 else None
            except ValueError:
                logger.warning(f"Invalid SFT_MAX_SAMPLES value: {debug_n}")
                return None
        return None
    
    @staticmethod
    def get_max_steps() -> int:
        """Get max training steps from environment."""
        try:
            return int(os.environ.get("SFT_MAX_STEPS", "0"))
        except ValueError:
            return 0
    
    @staticmethod
    def use_gradient_checkpointing() -> bool:
        """Check if gradient checkpointing should be enabled."""
        return os.environ.get("SFT_GRADIENT_CHECKPOINTING", "0") == "1"
    
    @staticmethod
    def use_liger_kernel() -> bool:
        """Check if liger kernel should be used."""
        return os.environ.get("SFT_DISABLE_LIGER", "0") != "1"
    
    @staticmethod
    def is_distributed() -> bool:
        """Check if running in distributed mode."""
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return world_size > 1


def build_messages(example: Dict, system_prompt: str) -> List[Dict[str, str]]:
    """
    Build a message list for chat template formatting.
    
    Args:
        example: Dataset example containing 'question', 'r1_solution_1', 'final_answer'
        system_prompt: System prompt to use for the conversation
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    assistant_answer = (
        f"<think>{example['r1_solution_1']}</think>"
        f"boxed{{{example['final_answer']}}}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": assistant_answer},
    ]


def format_to_text(
    batch: Dict, 
    tokenizer: AutoTokenizer, 
    system_prompt: str
) -> Dict[str, str]:
    """
    Format a batch example to text using chat template.
    
    Args:
        batch: Dataset batch to format
        tokenizer: Tokenizer with chat template
        system_prompt: System prompt for the conversation
        
    Returns:
        Dictionary with 'text' key containing formatted text
    """
    messages = build_messages(batch, system_prompt)
    text = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return {"text": text}


def get_lora_config(config: TrainingConfiguration) -> LoraConfig:
    """
    Create LoRA configuration for parameter-efficient fine-tuning.
    
    Args:
        config: Training configuration containing LoRA parameters
        
    Returns:
        Configured LoraConfig object
    """
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules,
    )


def load_tokenizer(
    model_name: str, 
    use_distributed: bool = False
) -> AutoTokenizer:
    """
    Load and configure tokenizer for the model.
    
    Args:
        model_name: Name or path of the model
        use_distributed: Whether running in distributed mode
        
    Returns:
        Configured tokenizer
    """
    logger.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(
    model_name: str,
    use_distributed: bool = False,
    enable_gradient_checkpointing: bool = False,
    use_liger: bool = True
) -> AutoModelForCausalLM:
    """
    Load and configure the causal language model.
    
    Args:
        model_name: Name or path of the model
        use_distributed: Whether running in distributed mode
        enable_gradient_checkpointing: Whether to enable gradient checkpointing
        use_liger: Whether liger kernel is enabled
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_name}...")
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    
    # IMPORTANT: device_map='auto' causes issues with Liger kernel in multi-GPU setups
    # The model will be placed on devices by accelerate/trainer automatically
    # Only use device_map='auto' in single-GPU non-distributed scenarios
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if not use_distributed and num_gpus == 1:
        # Single GPU: safe to use device_map
        model_kwargs["device_map"] = "auto"
        logger.info("Single GPU detected: using device_map='auto'")
    elif not use_distributed and num_gpus > 1:
        # Multi-GPU without accelerate: Don't use device_map with Liger
        if not use_liger:
            model_kwargs["device_map"] = "auto"
            logger.info(f"Multi-GPU ({num_gpus}) without Liger: using device_map='auto'")
        else:
            logger.info(f"Multi-GPU ({num_gpus}) with Liger: letting trainer handle device placement")
    else:
        # Distributed training: accelerate handles device placement
        logger.info(f"Distributed training: accelerate will handle device placement")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Move to cuda:0 if no device_map was set and not distributed
    if not use_distributed and "device_map" not in model_kwargs:
        if torch.cuda.is_available():
            model.to("cuda:0")
            logger.info("Moved model to CUDA:0")
    
    # Configure gradient checkpointing
    if enable_gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
        logger.info("Gradient checkpointing will be enabled through training args")
    
    return model


def load_and_prepare_dataset(
    dataset_path: str,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    test_split_ratio: float = 0.05,
    random_seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare training and evaluation datasets.
    
    Args:
        dataset_path: Path to the CSV dataset file
        tokenizer: Tokenizer for formatting
        system_prompt: System prompt to use
        test_split_ratio: Ratio of data to use for evaluation
        random_seed: Random seed for reproducibility
        max_samples: Optional limit on number of samples (for debugging)
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    base_dataset = Dataset.from_pandas(df)
    
    split_dataset = base_dataset.train_test_split(
        test_size=test_split_ratio, 
        seed=random_seed
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(
        f"Train samples: {len(train_dataset)}, "
        f"Eval samples: {len(eval_dataset)}"
    )
    
    # Apply debug sample limiting if specified
    if max_samples is not None and max_samples > 0:
        logger.warning(
            f"[DEBUG] Limiting datasets to first {max_samples} samples "
            "for rapid iteration"
        )
        train_dataset = train_dataset.select(
            range(min(max_samples, len(train_dataset)))
        )
        eval_dataset = eval_dataset.select(
            range(min(max(1, max_samples // 10), len(eval_dataset)))
        )
        logger.info(
            f"[DEBUG] New sizes -> Train: {len(train_dataset)}, "
            f"Eval: {len(eval_dataset)}"
        )
    
    # Format datasets with chat template
    logger.info("Formatting datasets with chat template → text field...")
    train_dataset = train_dataset.map(
        lambda ex: format_to_text(ex, tokenizer, system_prompt),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda ex: format_to_text(ex, tokenizer, system_prompt),
        remove_columns=eval_dataset.column_names,
    )
    logger.info("Formatting complete")
    
    return train_dataset, eval_dataset


def create_training_config(
    config: TrainingConfiguration,
    output_dir: str,
    use_liger: bool,
    want_gc: bool,
    max_steps: int
) -> SFTConfig:
    """
    Create SFT training configuration.
    
    Args:
        config: Training configuration object
        output_dir: Directory to save outputs
        use_liger: Whether to use liger kernel
        want_gc: Whether to use gradient checkpointing
        max_steps: Maximum training steps (overrides epochs if > 0)
        
    Returns:
        Configured SFTConfig object
    """
    gradient_checkpointing_kwargs = (
        {"use_reentrant": False} if want_gc else None
    )
    
    return SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=0.0001,
        warmup_ratio=0.1,
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        report_to=["wandb"],
        bf16=True,
        lr_scheduler_type="cosine",
        packing=False,
        model_init_kwargs={"torch_dtype": torch.bfloat16},
        use_liger_kernel=use_liger,
        gradient_checkpointing=want_gc,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        ddp_find_unused_parameters=False,
    )


def main():
    """Main training function."""
    # Initialize configuration
    config = TrainingConfiguration()
    env_config = EnvironmentConfig()
    
    # Get environment-based settings
    output_dir = env_config.get_output_dir()
    max_samples = env_config.get_max_samples()
    max_steps = env_config.get_max_steps()
    use_gradient_checkpointing = env_config.use_gradient_checkpointing()
    use_liger = env_config.use_liger_kernel()
    is_distributed = env_config.is_distributed()
    
    # Get system prompt
    system_prompt = SYSTEM_PROMPT[config.system_prompt_type]
    
    # Log configuration
    logger.info("="*60)
    logger.info("Training Configuration")
    logger.info("="*60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Epochs: {config.num_train_epochs}")
    logger.info(f"Batch size: {config.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info(f"Max steps override: {max_steps if max_steps > 0 else 'None (use epochs)'}")
    logger.info(f"Max samples (debug): {max_samples if max_samples else 'None (use all)'}")
    logger.info(f"Distributed: {is_distributed}")
    logger.info(f"Gradient checkpointing: {use_gradient_checkpointing}")
    logger.info(f"Use Liger kernel: {use_liger}")
    logger.info("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load tokenizer and model
        tokenizer = load_tokenizer(config.model_name, is_distributed)
        model = load_model(
            config.model_name,
            is_distributed,
            use_gradient_checkpointing,
            use_liger  # Pass liger flag to model loader
        )
        
        # Load and prepare datasets
        train_dataset, eval_dataset = load_and_prepare_dataset(
            config.dataset_path,
            tokenizer,
            system_prompt,
            config.test_split_ratio,
            config.random_seed,
            max_samples
        )
        
        # Create training configuration
        training_args = create_training_config(
            config,
            output_dir,
            use_liger,
            use_gradient_checkpointing,
            max_steps
        )
        
        # Initialize trainer
        logger.info("Initializing SFT trainer...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=get_lora_config(config),
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save model and tokenizer
        logger.info("Saving model...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Training complete. Artifacts saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Parse command line arguments (for compatibility with accelerate launch)
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument("--config_file", type=str, help="Accelerate config file (handled by accelerate launcher)")
    args, unknown = parser.parse_known_args()
    
    # Run main training
    main()
