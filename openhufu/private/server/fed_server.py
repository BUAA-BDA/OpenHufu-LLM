

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
        if client_id in self.selected_clients:
            self.client_id2lora[client_id] = lora
            self.client_id2weight[client_id] = weight
        
        if len(self.selected_clients) == len(self.client_id2weight):
            # all accept
            # selected_clients_set,id2params, id2weight, epoch
            if self.global_rounds_remain == 0:
                return
            agg_res = FedAvg(selected_clients_set=self.selected_clients, id2params=self.client_id2lora, 
                   id2weight=self.client_id2weight, epoch=self.config.num_communication_rounds - self.global_rounds_remain)
            self.selected_clients = client_selection(self.com_manager.get_all_client_id(), self.config.client_selection_frac)
            self.client_id2lora.clear()
            self.client_id2weight.clear()
            self.global_rounds_remain -= 1
            for i in self.selected_clients:
                
                self.send_message(i, defs.CellChannel.CLIENT_MAIN, defs.CellChannelTopic.Update, update=agg_res, epoch=self.config.num_communication_rounds - self.global_rounds_remain) 

    # def _register_server_cbs(self):
    #     # self.cell.register_request_cb(CellChannel.SERVER_MAIN, CellChannelTopic.Challenge, self.client_challenge)
    #     self.cell.register_request_cb(CellChannel.SERVER_MAIN, CellChannelTopic.Register, self.client_register)
        
    
    def client_register(self, client_id):
        # source_endpoint = message.get_from_headers(HeaderKey.SOURCE_ENDPOINT)
        # dest_endpoint = message.get_from_headers(HeaderKey.DESTINATION_ENDPOINT)
        self.logger.info(f"Client {client_id} registered")
        # 在注册的时候把id分配给它
        self.send_message(client_id, defs.CellChannel.CLIENT_MAIN, defs.CellChannelTopic.Start, epoch=self.config.num_communication_rounds - self.global_rounds_remain)
        
        
    def _register_all_callback(self):
        # self.msg_handlers[defs.CellChannelTopic.Share] = self.agg_params
        self._register_handler(defs.CellChannelTopic.Share, self.agg_params)
        self._register_handler(defs.CellChannelTopic.Register, self.client_register)
    
    def msg_handler(self, msg_type):
        return self.msg_handlers[msg_type]
    
    def deploy(self):
        self.com_manager.start()
        
        # self._register_server_cbs()
        # self.selected_clients = client_selection(self.com_manager.get_all_client_id(), self.config.client_selection_frac)
        # for i in self.selected_clients:
        #     self.logger.info(f'server send start to client{i}')
        #     self.send_message(i, defs.CellChannel.CLIENT_MAIN, defs.CellChannelTopic.Start, 
        #                     epoch=self.config.num_communication_rounds - self.global_rounds_remain)
        # while self.global_rounds_remain > 0:
        #     # 可以使用一个简单的等待，或者更好的是一个事件通知机制
        #     time.sleep(10)  # 每10秒检查一次状态
        # TODO: stop可能有问题 导致了死锁
        self.com_manager.stop()
        
    
        