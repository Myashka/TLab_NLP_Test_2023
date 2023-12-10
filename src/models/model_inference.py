import gc
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GenerationConfig

from src.utils.save_csv import save_csv


def generate_outputs(model, batch_ids, generation_config):
    with torch.autocast("cuda"):
        output_tokens = (
            model.generate(
                input_ids=batch_ids["input_ids"],
                attention_mask=batch_ids["attention_mask"],
                generation_config=generation_config,
            )
            .cpu()
            .numpy()
        )
    return output_tokens


def decode_outputs(tokenizer, output_tokens, input_ids, generation_config):
    input_ids = [
        ids for ids in input_ids for _ in range(generation_config.num_return_sequences)
    ]

    outputs = []
    prompts = []
    for sample_output_tokens, sample_input_ids in zip(output_tokens, input_ids):
        sample_output_tokens = sample_output_tokens[len(sample_input_ids) :]

        gen_response = tokenizer.decode(sample_output_tokens, skip_special_tokens=True)
        prompt = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
        outputs.append(gen_response)
        prompts.append(prompt)
    return prompts, outputs


def log_results(results, log_config):
    os.makedirs(log_config["dir"], exist_ok=True)
    save_csv(results, f"{log_config['dir']}/{log_config['file_name']}")


def run_model(
    model,
    dataloader_ids,
    tokenizer,
    generate_config,
    log_config,
):
    results = pd.DataFrame()
    generation_config = GenerationConfig(
        **generate_config, pad_token_id=tokenizer.pad_token_id
    )

    for i, batch_ids in enumerate(tqdm(dataloader_ids)):
        batch_ids = {k: v.to(model.device) for k, v in batch_ids.items()}
        output_tokens = generate_outputs(model, batch_ids, generation_config)
        prompts, outputs = decode_outputs(
            tokenizer, output_tokens, batch_ids["input_ids"], generation_config
        )

        id_sequence = range(
            i * len(outputs) // generation_config.num_return_sequences,
            (i + 1) * len(outputs) // generation_config.num_return_sequences,
        )

        ids = [
            number
            for number in id_sequence
            for _ in range(generation_config.num_return_sequences)
        ]

        labels = [value.item() for value in batch_ids['positive'] for _ in range(generation_config.num_return_sequences)]

        result_dict = dict()
        result_dict["gen_id"] = ids
        result_dict["prompt"] = prompts
        result_dict["response"] = outputs
        result_dict["label"] = labels

        result = pd.DataFrame(result_dict)

        results = pd.concat([results, result], ignore_index=True)

        del output_tokens, result
        torch.cuda.empty_cache()
        gc.collect()

        if (i + 1) % log_config["save_steps"] == 0:
            log_results(results, log_config)

            # Clear the results for the next iteration
            results = pd.DataFrame()
            gc.collect()

    if not results.empty:
        log_results(results, log_config)
