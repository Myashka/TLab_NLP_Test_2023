#!/bin/sh
# Level 1

### Перед тренировками ###
# wandb init
# huggingface-cli login

### В config необходимо менять пути ###

### В качетсве обученных моделей использовался один и тот же шаг валидации - 7000 ###

# инициализация 2286 тренировочных сэмплов для дальнейшей генерации
python scripts/data/init_dataset.py --dataset_name imdb --split train --num_samples 2286 --seed 42 --save_path ./data/raw/train_2286.csv
# 1000 тестовых сэмплов для замера награды и разнообразия
python scripts/data/init_dataset.py --dataset_name imdb --split test --num_samples 500 --seed 42 --save_path ./data/raw/test_500.csv

# Тест SFT политики
python scripts/model/inference.py --config_file ./configs/inference/test_sft_config.yaml
# Фильтрация от коротких ответов
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/sft.csv --output_file ./data/processed/test/sft.csv --min_token_count 10
# Подсчет наград SFT политики
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/sft.csv

# Генерация тренировочного набора данных (8 генераций на 1 prompt)
python scripts/model/inference.py --config_file ./configs/inference/make_N_samples_config.yaml
python scripts/data/filter_gen.py --input_file ./data/intermediate/sft_N_response.csv --output_file ./data/intermediate/sft_N_response_filtered.csv --min_token_count 10
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/intermediate/sft_N_response_filtered.csv
# Создание Constractive dataset
python scripts/data/create_constractive_dataset.py --input_file ./data/intermediate/sft_N_response_filtered.csv --output_file ./data/processed/train/constractive.csv

# Тренировки
python scripts/model/train.py --config_file ./configs/train/sigmoid_train.yaml
python scripts/model/train.py --config_file ./configs/train/hinge_train.yaml

# Тест политик
python scripts/model/inference.py --config_file ./configs/inference/test_sigmoid_config.yaml
python scripts/model/inference.py --config_file ./configs/inference/test_hinge_config.yaml

python scripts/data/filter_gen.py --input_file ./data/intermediate/test/hinge.csv --output_file ./data/processed/test/hinge.csv --min_token_count 10
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/sigmoid.csv --output_file ./data/processed/test/sigmoid.csv --min_token_count 10

python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/hinge.csv
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/sigmoid.csv

Level 2
python scripts/model/train.py --config_file ./configs/train/jsd_train.yaml
python scripts/model/train.py --config_file ./configs/train/fkl_train.yaml
# менять значения alpha [0.3, 0.5, 0.7]
python scripts/model/train.py --config_file ./configs/train/alpha_train.yaml

python scripts/model/inference.py --config_file ./configs/inference/test_jsd_config.yaml
python scripts/model/inference.py --config_file ./configs/inference/test_fkl_config.yaml
python scripts/model/inference.py --config_file ./configs/inference/test_alpha_03_config.yaml
python scripts/model/inference.py --config_file ./configs/inference/test_alpha_05_config.yaml
python scripts/model/inference.py --config_file ./configs/inference/test_alpha_07_config.yaml

python scripts/data/filter_gen.py --input_file ./data/intermediate/test/jsd.csv --output_file ./data/processed/test/jsd.csv --min_token_count 10
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/fkl.csv --output_file ./data/processed/test/fkl.csv --min_token_count 10
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/alpha_03.csv --output_file ./data/processed/test/alpha_03.csv --min_token_count 10
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/alpha_05.csv --output_file ./data/processed/test/alpha_05.csv --min_token_count 10
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/alpha_07.csv --output_file ./data/processed/test/alpha_07.csv --min_token_count 10

python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/jsd.csv
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/fkl.csv
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/alpha_03.csv
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/alpha_05.csv
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/alpha_07.csv

# Level 3

# значения beta [0.1, 0.3, 0.5]
python scripts/model/train.py --config_file ./configs/train/ipo_train.yaml
python scripts/model/train.py --config_file ./configs/train/cdpo_train.yaml
python scripts/model/train.py --config_file ./configs/train/ipo_annealing_train.yaml
python scripts/model/train.py --config_file ./configs/train/dpo_annealing_train.yaml

python scripts/model/inference.py --config_file ./configs/inference/test_ipo_01_config.yaml
python scripts/model/inference.py --config_file ./configs/inference/test_ipo_05_config.yaml
python scripts/model/inference.py --config_file ./configs/inference/test_ipo_03_config.yaml
python scripts/model/inference.py --config_file ./configs/inference/test_cdpo_config.yaml
python scripts/model/inference.py --config_file ./configs/inference/test_ipo_annealing_config.yaml
python scripts/model/inference.py --config_file ./configs/inference/test_dpo_annealing_config.yaml

python scripts/data/filter_gen.py --input_file ./data/intermediate/test/ipo_01.csv --output_file ./data/processed/test/ipo_01.csv --min_token_count 10
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/ipo_03.csv --output_file ./data/processed/test/ipo_03.csv --min_token_count 10
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/ipo_05.csv --output_file ./data/processed/test/ipo_05.csv --min_token_count 10
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/cdpo_015.csv --output_file ./data/processed/test/cdpo_015.csv --min_token_count 10
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/ipo_annealing.csv --output_file ./data/processed/test/ipo_annealing.csv --min_token_count 10
python scripts/data/filter_gen.py --input_file ./data/intermediate/test/dpo_annealing.csv --output_file ./data/processed/test/dpo_annealing.csv --min_token_count 10

python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/ipo_01.csv
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/ipo_03.csv
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/ipo_05.csv
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/cdpo_015.csv
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/ipo_annealing.csv
python scripts/evaluation/get_rewards.py --text_column response --file_path ./data/processed/test/dpo_annealing.csv


### ./notebooks/evaluation.ipynb - получение графиков ###