from peft import LoraConfig, get_peft_model
from .fed_client import FederatedClient

class FederatedClient_LoRA(FederatedClient):
    def __init__(self, config, cell):
        super().__init__(config, cell)
        self.lora_config = LoraConfig(
            r=self.config.lora.lora_r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=self.config.lora.task_type,
        )
        self.model = get_peft_model(self.model, self.lora_config)