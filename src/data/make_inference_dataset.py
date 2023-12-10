from datasets import load_dataset
import torch

def make_inference_dataset(
    dataset_name,
    tokenizer,
    max_prompt_length,
    prompt_column,
):
    def promt_tokenize(example):
        tokenized_dict = tokenizer(
            example[prompt_column],
            padding="longest",
            max_length=max_prompt_length,
            truncation=True,
        )
        example['positive'] = torch.tensor(example['label']).unsqueeze(0)
        return tokenized_dict

    dataset = load_dataset("csv", data_files=dataset_name)["train"]
    dataset = dataset.map(promt_tokenize)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "positive"])
    return dataset
