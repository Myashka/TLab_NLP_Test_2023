import sys

import click
import torch
import yaml
from tqdm import tqdm
from yaml import CLoader

tqdm.pandas()

import os

from peft import LoraConfig
from transformers import TrainingArguments
from trl import DPOTrainer

from src.models.EnhancedDPOTrainer import EnhancedDPOTrainer
from src.data.make_constractive_dataset import make_datasets
from src.utils.load_model import load_model
from src.utils.set_random_seed import set_random_seed

os.environ["WANDB_PROJECT"] = "tk_nlp"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        args_config = yaml.load(f, Loader=CLoader)

    set_random_seed(args_config["training_arguments"]["seed"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    data_config = args_config["data"]
    lora_config = args_config.get("lora_config", None)
    model_config = args_config["model"]
    training_arguments = args_config["training_arguments"]

    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model_config["device_map"] = device_map

    model, tokenizer = load_model(model_config)

    peft_config = None
    if lora_config:
        peft_config = LoraConfig(task_type="CAUSAL_LM", **lora_config)

    train_dataset, eval_dataset = make_datasets(**data_config)

    max_steps = (
        len(train_dataset)
        // training_arguments["per_device_train_batch_size"]
        // training_arguments.get("gradient_accumulation_steps", 1)
        * training_arguments.get("num_train_epochs", 1)
    )
    training_arguments["max_steps"] = max_steps

    training_args = TrainingArguments(
        ddp_find_unused_parameters=False if ddp else None,
        **training_arguments,
    )

    alpha = args_config.get("alpha", None)
    label_smoothing = args_config.get("label_smoothing", 0.0)
    annealing = args_config.get("annealing", False)

    dpo_trainer = EnhancedDPOTrainer(
        model=model,
        args=training_args,
        alpha=alpha,
        label_smoothing=label_smoothing,
        annealing=annealing,
        beta=args_config["beta"],
        loss_type=args_config["loss_type"],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=32,
        max_length=288,
        generate_during_eval=args_config["generate_during_eval"],
    )

    if training_arguments["gradient_checkpointing"]:
        dpo_trainer.model.gradient_checkpointing_enable()
    dpo_trainer.model.config.use_cache = not training_arguments[
        "gradient_checkpointing"
    ]

    dpo_trainer.model.enable_input_require_grads()
    dpo_trainer.ref_model.enable_input_require_grads()

    # Ошибка safetensors
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     dpo_trainer.model = torch.compile(dpo_trainer.model)

    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(args_config["training_arguments"]["output_dir"])
    dpo_trainer.tokenizer.save_pretrained(
        args_config["training_arguments"]["output_dir"]
    )

    dpo_trainer.push_to_hub()


if __name__ == "__main__":
    main()
