# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets


def get_preprocessed_uspto(dataset_config, tokenizer, split):
    print('split:', type(split), split)
#    temp = datasets.load_dataset('csv',data_files="/afs/crc.nd.edu/user/x/xhuang2/llama_test/llama-recipes/recipes/finetuning/datasets/combined_finetuning_data.csv")
    if split == 'train':
        temp = datasets.Dataset.from_csv("/afs/crc.nd.edu/user/x/xhuang2/llama_test/llama-recipes/recipes/finetuning/datasets/combined_finetuning_data_train.csv")
        dataset = temp
    else:
        temp = datasets.Dataset.from_csv("/afs/crc.nd.edu/user/x/xhuang2/llama_test/llama-recipes/recipes/finetuning/datasets/combined_finetuning_data_test.csv")
        dataset = temp

    prompt = (
            f"Format the data into JSON format:\n{{dialog}}\n---\nHere is the JSON format:\n"
    )

    def apply_prompt_template(sample):
        return {
            "prompt": prompt.format(dialog=sample["input"]),
            "summary": sample["label"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        summary = tokenizer.encode(sample["summary"] +  tokenizer.eos_token, add_special_tokens=False)

        sample = {
            "input_ids": prompt + summary,
            "attention_mask" : [1] * (len(prompt) + len(summary)),
            "labels": [-100] * len(prompt) + summary,
            }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
