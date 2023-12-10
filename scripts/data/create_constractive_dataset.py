import argparse
from itertools import combinations

import pandas as pd
from tqdm import tqdm


def create_comparison_dataframe(df):
    comparison_data = []

    for gen_id in tqdm(df['gen_id'].unique()):
        gen_df = df[df['gen_id'] == gen_id]
        if len(gen_df) == 1:
            continue

        for (idx1, row1), (idx2, row2) in combinations(gen_df.iterrows(), 2):
            if row1['reward'] > row2['reward']:
                chosen, rejected = row1['response'], row2['response']
            else:
                chosen, rejected = row2['response'], row1['response']

            comparison_data.append({'prompt': row1['prompt'], 'chosen': chosen, 'rejected': rejected})

    return pd.DataFrame(comparison_data)

def print_statistics(df):
    print(f"Total number of unique prompts: {df['prompt'].nunique()}")
    print(f"Total number of responses: {len(df)}")
    print(f"Average reward across all responses: {df['reward'].mean():.2f}")
    print(f"Minimum reward: {df['reward'].min()}, Maximum reward: {df['reward'].max()}")

def print_pair_statistics(df):
    pairs_per_group = df.groupby('gen_id').apply(lambda x: len(list(combinations(x.index, 2))))
    total_pairs = pairs_per_group.sum()
    print(f"Average number of pairs per group: {pairs_per_group.mean():.2f}")
    print(f"Total number of pairs in the comparison dataset: {total_pairs}")

def main(input_file, output_file):
    df = pd.read_csv(input_file)
    print_statistics(df)
    comparison_df = create_comparison_dataframe(df)
    print_pair_statistics(df)
    comparison_df.to_csv(output_file, index=False)
    print(f"Comparison data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create comparison dataset from responses")
    parser.add_argument("--input_file", type=str, help="Input CSV file path")
    parser.add_argument("--output_file", type=str, help="Output CSV file path")
    args = parser.parse_args()
    main(args.input_file, args.output_file)
