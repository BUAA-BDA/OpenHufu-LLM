# openhufu/distill/base.py
import os
import torch
from typing import Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model


class BaseModel:
    """Base class for model integration"""

    def __init__(self, model_name: str, device_map: str = "cuda:0",
                 load_in_4bit: bool = True, use_flash_attn: bool = True):
        self.model_name = model_name

        # Configure quantization parameters
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="right"
        )

        # Set special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def add_lora(self, lora_config: Union[str, os.PathLike, LoraConfig, None] = None):
        """Add LoRA adapter to the model"""
        if lora_config is None:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
        elif isinstance(lora_config, (str, os.PathLike)):
            lora_config = LoraConfig.from_pretrained(lora_config)
            
        self.model = get_peft_model(self.model, lora_config)

    def save_model(self, output_dir: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


class SLM(BaseModel):
    """Small Language Model class"""

    def __init__(self, model_name: str = "Qwen/Qwen1.5-1.5B", **kwargs):
        super().__init__(model_name, **kwargs)

    def add_lora(self, lora_config: Union[str, os.PathLike, LoraConfig, None] = None):
        """Add LoRA adapter with SLM-specific parameters"""
        if lora_config is None:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
        super().add_lora(lora_config)


class LLM(BaseModel):
    """Large Language Model class"""

    def __init__(self, model_name: str = "Qwen/Qwen1.5-7B", **kwargs):
        super().__init__(model_name, **kwargs)

    def add_lora(self, lora_config: Union[str, os.PathLike, LoraConfig, None] = None):
        """Add LoRA adapter with LLM-specific parameters"""
        if lora_config is None:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
        super().add_lora(lora_config)