import os

import pandas as pd
from datasets import Dataset
from trl import (
    RLOOTrainer,
    RLOOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
)
from trl.rewards import accuracy_rewards
from reward import process_reward, accuracy_reward_old,think_format_reward# , accuracy_reward , think_format_reward
from utils.chat_template import SYSTEM_PROMPT
from utils.utils import prepare_split
import weave


os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    print("LOAD DEEPMATH DATA SET")
    
    deepmath=pd.read_csv('DeepMath-103k_id.csv')
    dataset = Dataset.from_pandas(deepmath)
    
    split_dataset = dataset.train_test_split(test_size=0.05)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print("LOAD Complete")

    sp = SYSTEM_PROMPT["simplerl"]

    print("MAPPING DATASET")
    train_dataset = prepare_split(train_dataset, sp)
    eval_dataset = prepare_split(eval_dataset, sp)
    print("MAPPING COMPLETE")
    
    trainer = RLOOTrainer(
    model=model_args.model_name_or_path,
    args=training_args,
    reward_funcs=[accuracy_reward_old, process_reward],
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    )
    
    
    try:
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        print(f"[WARN] Failed to resume checkpoint: {e}")
        trainer.train()
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)