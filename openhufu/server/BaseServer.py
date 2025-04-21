from worker import Worker
import openhufu.private.utlis.defs as defs
from openhufu.aggregator.fedavg import FedAvg
from scheduler import client_selection
class BaseServer(Worker):

    def __init__(self, config, com_manager):
        super().__init__(-1, com_manager)
        self.config = config
        self.client_id2lora = dict()
        self.client_id2weight = dict()
        self.selected_clients = set()
        self.global_rounds_remain = config.epoch

    def agg_params(self, client_id, lora, weight):
        if client_id in self.selected_clients:
            self.client_id2lora[client_id] = lora
            self.client_id2weight[client_id] = weight
        
        if len(self.selected_clients) == len(self.client_id2weight):
            # all accept
            # selected_clients_set,id2params, id2weight, epoch
            if self.global_rounds_remain == 0:
                return
            agg_res = FedAvg(selected_clients_set=self.selected_clients, id2params=self.client_id2lora, 
                   id2weight=self.client_id2weight)
            self.selected_clients = client_selection(self.com_manager.get_all_client_id(), self.config.client_selection_frac)
            self.client_id2lora.clear()
            self.client_id2weight.clear()
            self.global_rounds_remain -= 1
            for i in self.selected_clients():
                self.com_manager.send_message(i, defs.CellChannel.CLIENT_MAIN, defs.CellChannelTopic.Update, update=agg_res, epoch=self.config.epoch - self.global_rounds_remain) 

   

    def __register_all_callback(self):
        # self.msg_handlers[defs.CellChannelTopic.Share] = self.agg_params
        self.__register_handler(defs.CellChannelTopic.Share, self.agg_params)

    def msg_handler(self, msg_type):
        return self.msg_handlers[msg_type]
    
    def deploy(self):
        self.selected_clients = client_selection(self.com_manager.get_all_client_id(), self.config.client_selection_frac)
        for i in self.selected_clients():
                self.com_manager.send_message(i, defs.CellChannel.CLIENT_MAIN, defs.CellChannelTopic.Update, epoch=self.config.epoch - self.global_rounds_remain) 