import os
from openhufu.private.net.client_cell import ClientCell
from openhufu.private.utlis.config_class import ClientConfig
from openhufu.private.utlis.util import get_logger
from openhufu.private.utlis.factory import get_cell
from openhufu.worker.Worker import Worker
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import torch
from openhufu.utils import Prompter
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import openhufu.private.utlis.defs as defs
logger = get_logger("fed_client")

class FederatedClient(Worker):
    def __init__(self, config, cell):
        super().__init__(config.id, cell)
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
        self.logger = get_logger(__name__)
    
    # def _create_cell(self):
    #     # self.cell = ClientCell(config=self.config)
    #     self.cell = get_cell(config=self.config)
    #     self.cell.start()
    
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
        self.model.config.use_cache = False
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))


    def __build_local_trainer(self):
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
        print('nononononononon')
        single_output_dir = os.path.join(self.local_output_dir, str(self.epoch), "local_output_{}".format(self.id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(local_lora_weight, single_output_dir + "/pytorch_model.bin")
        # torch.save(local_embedding_weight, self.tem_path + f"/local_embedding{self.client_id}.bin")
        # 无需还原
        # previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        # last_client_id = self.client_id
        # print(local_lora_weight)
        return local_lora_weight,  # last_client_id

    def perform_local_train(self, epoch):
        self.epoch = epoch
        self.__init_local_training()
        self.__build_local_trainer()
        # self.__train()
        logger.info('build local trainer finished, start to train locally')
        lora_weight = self.__terminate_local_training()
        local_dataset_size = len(self.local_data)
        # self, id , channel: CellChannel , topic: CellChannelTopic, **kwargs
        # assert(isinstance(lora_weight, set))
        logger.info('local train finished, start to send param to server')
        print(f"state_dict_type:{type(lora_weight)})")
        print(lora_weight)
        self.send_message(target=-1,channel=defs.CellChannel.CLIENT_MAIN, 
                                      topic=defs.CellChannelTopic.Share, client_id = self.id, lora = lora_weight, weight = local_dataset_size)

    def send_model_params_to_server(self, update, epoch):
        # 每次重新创建一个Trainer
        logger.info("client recv update, do local train and upload param")
        set_peft_model_state_dict(self.model, update,'default')
        logger.info("ojohojsoidhaiosohdoasuhd")
        self.perform_local_train(epoch)
        

    def _register_all_callback(self):
        # self.msg_handlers[defs.CellChannelTopic.Update] = self.send_model_params_to_server
        # self.msg_handlers
        # print("execute son")
        self._register_handler(defs.CellChannelTopic.Update, self.send_model_params_to_server)
        self._register_handler(defs.CellChannelTopic.Start, self.perform_local_train)
        
        
    def set_up(self):    
        schema_location = self.config.host + ":" + str(self.config.port)
        self.config.addr = schema_location
        self.com_manager.start()
        
    
    def register(self):
        self.com_manager.register()
    
    
    def stop(self):
        self.com_manager.stop()