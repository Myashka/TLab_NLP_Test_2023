import sys

import click
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from yaml import CLoader

from src.data.make_inference_dataset import make_inference_dataset
from src.models.model_inference import run_model
from src.utils.load_model import load_model
from src.utils.set_random_seed import set_random_seed


@click.command()
@click.option("--config_file", default="inf_config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    model, tokenizer = load_model(config["model"])
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    set_random_seed(config["seed"])

    test_dataset = make_inference_dataset(tokenizer=tokenizer, **config["data"])

    dataloader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        collate_fn=DataCollatorForTokenClassification(
            tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
    )

    run_model(
        model,
        dataloader,
        tokenizer,
        config["generate_config"],
        config["log_config"],
    )

if __name__ == "__main__":
    main()
