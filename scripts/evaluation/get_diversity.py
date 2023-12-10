import argparse
from collections import defaultdict

import pandas as pd
from scipy.stats import entropy
from transformers import AutoTokenizer


def token_entropy(generations, tokenizer):
    stats = defaultdict(int)
    num_tokens = 0

    for example in generations:
        tokens = tokenizer.encode(example)
        for t in tokens:
            if t == tokenizer.pad_token_id:
                continue
            stats[t] += 1
            num_tokens += 1

    for k in stats.keys():
        stats[k] /= num_tokens

    return entropy(list(stats.values()))

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    df = pd.read_csv(args.file_path)
    text_column = args.text_column if args.text_column else df.columns[0]
    diversity_scores = df.groupby('gen_id')[text_column].apply(lambda x: token_entropy(x, tokenizer))
    print(f"Diversity Score (Entropy): {diversity_scores.mean()}")
    print(f"Mean Reward: {df['reward'].mean()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the diversity of generated texts")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the file containing generated texts")
    parser.add_argument("--tokenizer_name", type=str, default="lvwerra/gpt2-imdb", help="Tokenizer name or path")
    parser.add_argument("--text_column", type=str, help="Column name in the CSV file to analyze. Defaults to the first column if not specified.")
    
    args = parser.parse_args()
    main(args)
