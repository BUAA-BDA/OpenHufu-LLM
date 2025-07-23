# openhufu/distill/main.py

import os
import json
import torch
import random
import numpy as np
from typing import List

# Import local modules
from openhufu.distill.base import SLM, LLM
from openhufu.distill.sft import prepare_dataset_for_sft, train_model_with_sft
from openhufu.distill.distill import (
    generate_synthetic_data,
    evaluate_synthetic_data,
    filter_high_quality_data,
    synthetic_data_to_dataset
)

# Set random seed for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # Set random seed
    set_seed(42)

    # Configure parameters
    data_paths = ["path/to/silo1_data.json", "path/to/silo2_data.json"]
    prompt_templates = [
        "Write a detailed explanation about quantum computing: ",
        "Create a story about a futuristic society: ",
        "Describe the process of photosynthesis in plants: ",
        "Explain how neural networks work: "
    ]

    slm_output_dir = "outputs/slm_trained"
    llm_output_dir = "outputs/llm_trained"
    synthetic_data_path = "outputs/synthetic_data.json"
    filtered_data_path = "outputs/filtered_data.json"

    # Step 1: Initialize small model and perform SFT LoRA fine-tuning
    print("Step 1: Initializing small model and performing SFT fine-tuning...")
    slm = SLM()
    slm.add_lora()

    # Combine data from all silos for initial training
    for data_path in data_paths:
        train_dataset = prepare_dataset_for_sft(data_path, slm.tokenizer)
        slm = train_model_with_sft(slm, train_dataset, slm_output_dir)

    # Step 2: Use fine-tuned small model to generate synthetic data
    print("Step 2: Generating synthetic data...")
    synthetic_data = generate_synthetic_data(
        slm,
        prompt_templates,
        num_samples_per_prompt=20,
        temperature=0.8
    )

    # Save synthetic data
    with open(synthetic_data_path, "w") as f:
        json.dump(synthetic_data, f, indent=2)

    # Step 3: Initialize large model and evaluate synthetic data
    print("Step 3: Evaluating synthetic data quality...")
    llm = LLM()
    evaluated_data = evaluate_synthetic_data(llm, synthetic_data)

    # Step 4: Filter high quality synthetic data
    print("Step 4: Filtering high quality data...")
    filtered_data = filter_high_quality_data(evaluated_data, threshold=6)

    print(f"Generated synthetic data: {len(synthetic_data)} samples")
    print(f"High quality filtered data: {len(filtered_data)} samples")

    # Save filtered data
    with open(filtered_data_path, "w") as f:
        json.dump(filtered_data, f, indent=2)

    # Step 5: Use filtered synthetic data to fine-tune large model with SFT LoRA
    print("Step 5: Fine-tuning large model with synthetic data...")
    llm.add_lora()

    train_dataset = synthetic_data_to_dataset(filtered_data)
    llm = train_model_with_sft(llm, train_dataset, llm_output_dir)

    print("Knowledge transfer completed! Large model fine-tuning results saved at:", llm_output_dir)

if __name__ == "__main__":
    main()