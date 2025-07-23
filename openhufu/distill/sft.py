# openhufu/distill/sft.py

import os
import json
import torch
from typing import Dict, Any
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer


def prepare_dataset_for_sft(data_path: str, tokenizer) -> Dataset:
    """Prepare dataset for SFT training"""
    # Load dataset
    if os.path.isfile(data_path) and data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Use Hugging Face datasets to load dataset
        data = load_dataset(data_path)

    # Format data as needed
    formatted_data = []

    # Process based on dataset structure
    if isinstance(data, dict) and "train" in data:
        raw_data = data["train"]
    else:
        raw_data = data

    for item in raw_data:
        if isinstance(item, dict):
            if "instruction" in item and "output" in item:
                text = f"Human: {item['instruction']}\nAssistant: {item['output']}"
            elif "prompt" in item and "completion" in item:
                text = f"Human: {item['prompt']}\nAssistant: {item['completion']}"
            elif "input" in item and "output" in item:
                text = f"Human: {item['input']}\nAssistant: {item['output']}"
            else:
                # Default case, try to find question and answer keys
                keys = list(item.keys())
                if len(keys) >= 2:
                    text = f"Human: {item[keys[0]]}\nAssistant: {item[keys[1]]}"
                else:
                    continue
        else:
            continue

        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)


def train_model_with_sft(model, train_dataset: Dataset, output_dir: str, **training_args):
    """Train model using SFT with trl"""
    default_args = {
        "output_dir": output_dir,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "fp16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "optim": "paged_adamw_32bit"
    }

    # Update default args
    default_args.update(training_args)

    # Adjust parameters based on model size
    if model.model_name and "7B" in model.model_name:
        default_args["per_device_train_batch_size"] = 2
        default_args["gradient_accumulation_steps"] = 4
        default_args["learning_rate"] = 1e-4
        default_args["num_train_epochs"] = 2

    args = TrainingArguments(**default_args)

    model.model.print_trainable_parameters()
    
    trainer = SFTTrainer(
        model=model.model,
        args=args,
        train_dataset=train_dataset,
        processing_class=model.tokenizer,
    )

    trainer.train()
    model.save_model(output_dir)

    return model