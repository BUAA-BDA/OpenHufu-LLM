

from openhufu.private.utlis.util import get_logger
from openhufu.private.utlis.defs import CellChannel, CellChannelTopic
from openhufu.private.utlis.config_class import ServerConfig
from openhufu.private.net.cell import Cell
from openhufu.private.net.message import Message
from openhufu.private.utlis.defs import CellChannel, CellChannelTopic, HeaderKey
from openhufu.private.utlis.factory import get_cell
from openhufu.worker.Worker import Worker
from openhufu.worker import Worker
import openhufu.private.utlis.defs as defs
from openhufu.aggregator.fedavg import FedAvg
from openhufu.scheduler import client_selection
import time
logger = get_logger("fed_server")

class FederatedServer(Worker):
    def __init__(self, config, cell):
        super().__init__(-1, cell)
        self.config = config
        self.client_id2lora = dict()
        self.client_id2weight = dict()
        self.selected_clients = set()
        self.global_rounds_remain = config.num_communication_rounds
        self.logger = get_logger(__name__)


    def agg_params(self, client_id, lora, weight):
        # assert(isinstance(lora, set))
        logger.info(f'server recv param from client_{client_id}, param type is{type(lora)}')
        if client_id in self.selected_clients:
            self.client_id2lora[client_id] = lora
            self.client_id2weight[client_id] = weight
        
        if len(self.selected_clients) == len(self.client_id2weight):
            # all accept
            # selected_clients_set,id2params, id2weight, epoch
            if self.global_rounds_remain == 0:
                return
            logger.info(f'server recv all param, perform agg')
            agg_res = FedAvg(selected_clients_set=self.selected_clients, id2params=self.client_id2lora, 
                   id2weight=self.client_id2weight, epoch=self.config.num_communication_rounds - self.global_rounds_remain)
            print(type(agg_res))
            self.selected_clients = client_selection(self.com_manager.get_all_client_id(), self.config.client_selection_frac)
            self.client_id2lora.clear()
            self.client_id2weight.clear()
            self.global_rounds_remain -= 1
            for i in self.selected_clients:   
                self.send_message(i, defs.CellChannel.CLIENT_MAIN, defs.CellChannelTopic.Update, update=agg_res, epoch=self.config.num_communication_rounds - self.global_rounds_remain) 


    def client_register(self, client_id):
        self.logger.info(f"Client {client_id} registered")
        # 在注册的时候把id分配给它
        self.selected_clients.add(client_id)
        self.send_message(client_id, defs.CellChannel.CLIENT_MAIN, defs.CellChannelTopic.Start, epoch=self.config.num_communication_rounds - self.global_rounds_remain)
        
        
    def _register_all_callback(self):
        # self.msg_handlers[defs.CellChannelTopic.Share] = self.agg_params
        self._register_handler(defs.CellChannelTopic.Share, self.agg_params)
        self._register_handler(defs.CellChannelTopic.Register, self.client_register)
    
    def msg_handler(self, msg_type):
        return self.msg_handlers[msg_type]
    
    def deploy(self):
        self.com_manager.start()
        # TODO: stop可能有问题 导致了死锁
        self.com_manager.stop()
        
    
        