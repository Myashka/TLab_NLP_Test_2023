import argparse
import gc
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.set_random_seed import set_random_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_rewards(model, tokenizer, df, text_column, batch_size):
    rewards = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch_texts = df[text_column][i: i + batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, 1].tolist()
            rewards.extend(logits)

        gc.collect()

    return rewards

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    model.to(device)
    set_random_seed(42)
    model.eval()

    df = pd.read_csv(args.file_path, lineterminator="\n")
    print(f"Length of data: {len(df)}")

    rewards = get_rewards(model, tokenizer, df, args.text_column, args.batch_size)
    df['reward'] = rewards
    df.to_csv(args.file_path, index=False)
    print(f"Mean Reward: {np.mean(rewards)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for evaluating rewards")
    parser.add_argument("--text_column", type=str, required=True, help="Text column for reward calculation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--file_path", type=str, required=True, help="File path")
    parser.add_argument("--model_name", type=str, default="lvwerra/distilbert-imdb", help="Model name")
    
    args = parser.parse_args()
    main(args)
