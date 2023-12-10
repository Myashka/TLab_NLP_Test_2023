import pandas as pd
from transformers import AutoTokenizer
import argparse

def filter_responses_by_token_count(df, tokenizer, min_token_count):
    # Функция для подсчета токенов
    def token_count(text):
        return len(tokenizer.encode(text))

    # Удаляем строки с NaN в столбце 'response'
    df = df.dropna(subset=['response'])

    # Применяем функцию к каждой строке и фильтруем DataFrame
    df['token_count'] = df['response'].apply(token_count)
    filtered_df = df[df['token_count'] >= min_token_count]
    return filtered_df.drop(columns=['token_count'])

def main(input_file, output_file, min_token_count):
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")

    df = pd.read_csv(input_file)
    original_len = len(df)
    filtered_df = filter_responses_by_token_count(df, tokenizer, min_token_count)

    print(f"Original number of rows: {original_len}")
    print(f"Number of rows after filtering: {len(filtered_df)}")

    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter DataFrame rows based on token count in 'response' column")
    parser.add_argument("--input_file", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output_file", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--min_token_count", type=int, required=True, help="Minimum token count threshold")
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.min_token_count)
