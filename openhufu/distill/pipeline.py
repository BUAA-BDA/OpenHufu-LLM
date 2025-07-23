import yaml
import logging
import os
import random
import numpy as np
import torch
import json
import argparse

from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from base import SLM, LLM
from sft import prepare_dataset_for_sft, train_model_with_sft
from distill import (
    generate_synthetic_data,
    evaluate_synthetic_data,
    filter_high_quality_data,
    synthetic_data_to_dataset,
)
from dataset import preprocess_agnews_to_sft, select_examples, generate_new_data



def set_seed(seed: int):
    """Fix all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main(config_path: str):
    # ---------------------------------------------------------------------
    # 1. Load config & prepare environment
    # ---------------------------------------------------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    set_seed(cfg["seed"])

    # ---------------------------------------------------------------------
    # 2. Dataset preparation
    # ---------------------------------------------------------------------
    dataset = cfg["dataset"]["name"]
    train_ds_path = Path("data")/ cfg["dataset"] / "train.json"
    test_ds_path = Path("data") / cfg["dataset"] / "test.json"
    if dataset == "ag_news":
        train_ds, test_ds = preprocess_agnews_to_sft(train_csv=f"data/ag_news/train.csv", 
                                             test_csv=f"data/ag_news/test.csv",
                                             instruction_tmpl=cfg["dataset"]["instruction_tmpl"],
                                             output_tmpl=cfg["dataset"]["output_tmpl"])
        

    # ---------------------------------------------------------------------
    # 3. Stage-1: SLM (≈1.5B) LoRA fine-tuning
    # ---------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger.info("Stage-1 | Loading SLM …")
    slm = SLM(model_name=cfg["stage1"]["model_name"])
    slm.add_lora(lora_config="/home/koushurui/Docs/Code/OpenHufu-LLM/openhufu/distill/outputs/slm_lora")

    logger.info("Stage-1 | Preparing dataset …")
    train_dataset = prepare_dataset_for_sft(str(train_ds_path), slm.tokenizer)

    logger.info("Stage-1 | Fine-tuning …")
    train_model_with_sft(slm, train_dataset, cfg["stage1"]["output_dir"], **cfg["stage1"].get("lora", {}))

    # logger.info("Stage-1 | Evaluation …")
    # test_dataset = prepare_dataset_for_sft(test_ds_path, slm.tokenizer)
    # stage1_metrics = evaluate_lm(slm, test_dataset)
    # logger.info("Stage-1 | Metrics: %s", stage1_metrics)

    # ---------------------------------------------------------------------
    # 4. Distillation: generate + (optional) LLM scoring
    # ---------------------------------------------------------------------
    logger.info("Distillation | Generating synthetic data …")
    synthetic = generate_new_data(slm, prompt=cfg["distill"]["prompt"], dataset=cfg["dataset"]["name"], examples=train_ds, num_samples=cfg["distill"]["num_samples"])

    if cfg["distill"].get("with_llm_scoring", True):
        logger.info("Distillation | Evaluating synthetic data with LLM …")
        llm = LLM(model_name=cfg["stage2"]["model_name"])
        evaluated = evaluate_synthetic_data(llm, synthetic)
        synthetic = filter_high_quality_data(evaluated, threshold=cfg["distill"].get("score_threshold", 6))
        logger.info("Distillation | %d / %d samples kept after filtering.", len(synthetic), len(evaluated))

    # Save distilled data
    distilled_path = cfg["distill"].get("output_path", "outputs/distilled_data.json")
    os.makedirs(os.path.dirname(distilled_path), exist_ok=True)
    with open(distilled_path, "w", encoding="utf-8") as f:
        import json
        json.dump(synthetic, f, ensure_ascii=False, indent=2)

    # Convert to HF Dataset for training
    distilled_dataset = synthetic_data_to_dataset(synthetic)

    # ---------------------------------------------------------------------
    # 5. Stage-2: LLM (≈7B) LoRA fine-tuning on distilled data
    # ---------------------------------------------------------------------
    logger.info("Stage-2 | Loading LLM …")
    llm = LLM(model_name=cfg["stage2"]["model_name"]) if not llm else llm
    llm.add_lora()

    logger.info("Stage-2 | Fine-tuning …")
    train_model_with_sft(llm, distilled_dataset, cfg["stage2"]["output_dir"], **cfg["stage2"].get("lora", {}))

    # ---------------------------------------------------------------------
    # 6. Summarize
    # ---------------------------------------------------------------------
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full distillation pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)