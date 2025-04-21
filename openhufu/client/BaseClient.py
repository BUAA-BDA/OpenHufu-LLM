from openhufu.worker import Worker
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch
from datasets import load_dataset, load_from_disk
from collections import OrderedDict
import os
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from openhufu.utils import Prompter
import openhufu.private.utlis.defs as defs
# 客户端用的局部超参数：
# local_batch_size: int = 64 # 64,
# local_micro_batch_size: int = 8
# local_num_epochs: int = 10
# local_learning_rate: float = 3e-2
# local_val_set_size: int = 0
# local_save_steps: int = 3
# cutoff_len: int = 512

# # llm param
# train_on_inputs: bool = True
# group_by_length: bool = False
# device_map = 'auto'
class BaseClient(Worker):
    def __init__(self, id, config, com_manager):
        super().__init__(id, com_manager)
        # self.__register_all_callback()
        # self._register_handler()
        # self.model = config.model
        self.config = config
        self.model =AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=torch.float16,
            device_map=config.device_map,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.tokenizer.pad_token_id = (
            0
        )

        self.tokenizer.padding_side = "left"
        self.local_data = load_from_disk(os.path.join(self.config.local_data_path, f"client_{self.id + 1}_train"))
        self.prompter = Prompter(config.prompt_template_name)
        self.local_train_dataset = self.local_data.shuffle().map(self.__generate_and_tokenize_prompt)
        self.local_output_dir = os.path.join(self.config.output_dir, 'trainer_saved', str(self.id))
        self.epoch = 0

    def __tokenize(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.config.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def __generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.__tokenize(full_prompt)
        return tokenized_full_prompt
    
    def __init_local_training(self):
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))


    def __build_local_trainer(self,
                            # tokenizer,
                            # local_micro_batch_size,
                            # gradient_accumulation_steps,
                            # local_num_epochs,
                            # local_learning_rate,
                            # group_by_length,
                            # ddp=False
                            ):
        self.train_args = TrainingArguments(
            per_device_train_batch_size=self.config.local_micro_batch_size,
            gradient_accumulation_steps= self.config.local_batch_size // self.config.local_micro_batch_size,
            warmup_steps=0,
            num_train_epochs=self.config.local_num_epochs,
            learning_rate=self.config.local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps= None,
            save_steps=200,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            # load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=None,
            group_by_length=self.config.group_by_length,
            label_names= ['labels',], #标签名字
            dataloader_drop_last=False
        )
        self.local_trainer = Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset= None,
                                                  args=self.train_args,
                                                  data_collator=DataCollatorForSeq2Seq(
                                                      self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                                  ),
                                                  )
        
    def __train(self):
        self.local_trainer.train()
    
    def __terminate_local_training(self):
        # local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        # 发送本地数据个数
        local_lora_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(self.epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(local_lora_weight, single_output_dir + "/pytorch_model.bin")
        # torch.save(local_embedding_weight, self.tem_path + f"/local_embedding{self.client_id}.bin")
        # 无需还原
        # previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        # last_client_id = self.client_id
        return local_lora_weight,  # last_client_id

    def perform_local_train(self, epoch):
        self.epoch = epoch
        self.__init_local_training()
        self.__build_local_trainer()
        self.__train()
        lora_weight = self.__terminate_local_training()
        local_dataset_size = len(self.local_data)
        # self, id , channel: CellChannel , topic: CellChannelTopic, **kwargs
        self.com_manager.send_message(target=-1,channel=defs.CellChannel.CLIENT_MAIN, 
                                      topic=defs.CellChannelTopic.Share, client_id = self.id, lora = lora_weight, weight = local_dataset_size)

    def send_model_params_to_server(self, lora, epoch):
        # 每次重新创建一个Trainer
        set_peft_model_state_dict(self.model,lora,'default')
        self.perform_local_train(epoch)
        

    def _register_all_callback(self):
        # self.msg_handlers[defs.CellChannelTopic.Update] = self.send_model_params_to_server
        # self.msg_handlers
        print("execute son")
        self._register_handler(defs.CellChannelTopic.Update, self.send_model_params_to_server)
        self._register_handler(defs.CellChannelTopic.Start, self.perform_local_train)
    
   

   