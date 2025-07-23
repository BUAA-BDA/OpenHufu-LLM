# openhufu/distill/distill.py

import torch
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import Dataset




def generate_synthetic_data(model, prompts: List[str],
                           num_samples_per_prompt: int = 5,
                           max_new_tokens: int = 512,
                           temperature: float = 0.8) -> List[Dict[str, str]]:
    """Generate synthetic data using SLM"""
    synthetic_data = []

    for prompt in prompts:
        for _ in range(num_samples_per_prompt):
            inputs = model.tokenizer(prompt, return_tensors="pt").to(model.model.device)

            outputs = model.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
            )

            generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt, keep only generated content
            if prompt in generated_text:
                generated_text = generated_text[len(prompt):]

            synthetic_data.append({
                "prompt": prompt,
                "generated_text": generated_text.strip()
            })

    return synthetic_data

def evaluate_synthetic_data(model, synthetic_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Evaluate synthetic data quality using LLM"""
    evaluated_data = []

    for item in tqdm(synthetic_data, desc="Evaluating synthetic data"):
        generated_text = item["instruction"] + "\n" + item["output"]

        # Construct evaluation prompt
        eval_prompt = f"""Please evaluate the following text based on diversity and correctness.
Rate it from 1-10 (10 being the best).

Text to evaluate: {generated_text}

Rating (1-10): """

        inputs = model.tokenizer(eval_prompt, return_tensors="pt").to(model.model.device)

        with torch.no_grad():
            outputs = model.model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                temperature=0.1,
                num_return_sequences=1,
            )

        evaluation = model.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Try to extract the score
        try:
            # Extract score between 1-10
            score = int(evaluation.strip().split()[0])
            if score < 1:
                score = 1
            elif score > 10:
                score = 10
        except:
            # Default score
            score = 5

        item["score"] = score
        item["evaluation"] = evaluation
        evaluated_data.append(item)

    return evaluated_data

def filter_high_quality_data(evaluated_data: List[Dict[str, Any]], threshold: int = 6) -> List[Dict[str, Any]]:
    """Filter high quality data based on score threshold"""
    filtered_data = [item for item in evaluated_data if item["score"] >= threshold]
    return filtered_data

def synthetic_data_to_dataset(filtered_data: List[Dict[str, Any]]) -> Dataset:
    """Convert synthetic data to training dataset format"""
    formatted_data = []

    for item in filtered_data:
        text = f"Human: {item['instruction']}\nAssistant: {item['output']}"
        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)