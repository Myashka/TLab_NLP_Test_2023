import argparse
from datasets import load_dataset, concatenate_datasets
from src.utils.set_random_seed import set_random_seed

def sample_balanced_dataset(dataset_name, split, num_samples, seed=42):
    # Загружаем тестовый набор данных
    dataset = load_dataset(dataset_name, split=split)

    # Фильтруем семплы с label 0 и label 1
    positive_samples = dataset.filter(lambda example: example['label'] == 1)
    negative_samples = dataset.filter(lambda example: example['label'] == 0)

    # Семплируем половину из каждого класса
    positive_subset = positive_samples.shuffle(seed=seed).select(range(num_samples // 2))
    negative_subset = negative_samples.shuffle(seed=seed).select(range(num_samples // 2))

    # Объединяем и перемешиваем
    balanced_dataset = concatenate_datasets([positive_subset, negative_subset]).shuffle(seed=seed)
    return balanced_dataset

def main(args):
    set_random_seed(args.seed)
    final_dataset = sample_balanced_dataset(args.dataset_name, args.split, args.num_samples, args.seed)
    final_dataset.to_csv(args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample a balanced dataset and save it to CSV.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset to load from Huggingface datasets.')
    parser.add_argument('--split', type=str, required=True, help='Split of the dataset to load from Huggingface datasets.')
    parser.add_argument('--num_samples', type=int, required=True, help='Total number of samples to include in the balanced dataset.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the balanced dataset CSV.')
    
    args = parser.parse_args()
    main(args)
