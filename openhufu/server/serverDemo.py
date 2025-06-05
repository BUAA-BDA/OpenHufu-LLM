from openhufu.private.utlis.util import get_logger
from openhufu.private.utlis.defs import CellChannel, CellChannelTopic, HeaderKey
from openhufu.private.utlis.config_class import ServerConfig
from openhufu.private.net.cell import Cell
from openhufu.private.net.message import Message
from openhufu.private.utlis.factory import get_cell
from openhufu.worker import Worker
import openhufu.private.utlis.defs as defs
from openhufu.aggregator.fedavg import FedAvg
from openhufu.scheduler import client_selection

class UnifiedServer(Worker):
    """
    统一的服务器类，支持模拟模式和集群模式
    """
    def __init__(self, config, com_manager=None, mode="cluster"):
        """
        初始化服务器
        
        Args:
            config: 服务器配置
            com_manager: 通信管理器(集群模式需要)
            mode: 运行模式，可选值为"simulation"或"cluster"
        """
        self.mode = mode
        self.logger = get_logger(__name__)
        
        if mode == "simulation":
            # 模拟模式初始化
            self.config = config
            self.cell = None
        elif mode == "cluster":
            # 集群模式初始化
            super().__init__(-1, com_manager)
            self.config = config
            self.client_id2lora = dict()
            self.client_id2weight = dict()
            self.selected_clients = set()
            self.global_rounds_remain = config.num_communication_rounds
        else:
            raise ValueError(f"不支持的模式: {mode}, 请选择 'simulation' 或 'cluster'")
    
    # 模拟模式方法
    def _register_server_cbs(self):
        self.cell.register_request_cb(CellChannel.SERVER_MAIN, CellChannelTopic.Register, self.client_register)
    
    def client_register(self, message: Message):
        source_endpoint = message.get_from_headers(HeaderKey.SOURCE_ENDPOINT)
        dest_endpoint = message.get_from_headers(HeaderKey.DESTINATION_ENDPOINT)
        self.logger.info(f"Client {source_endpoint} registered")
    
    # 集群模式方法
    # def agg_params(self, client_id, lora, weight):
    #     if self.mode != "cluster":
    #         self.logger.warning("在非集群模式下调用了聚合参数方法")
    #         return
            
    #     if client_id in self.selected_clients:
    #         self.client_id2lora[client_id] = lora
    #         self.client_id2weight[client_id] = weight
        
    #     if len(self.selected_clients) == len(self.client_id2weight):
    #         # 所有客户端已接受
    #         if self.global_rounds_remain == 0:
    #             return
                
    #         agg_res = FedAvg(selected_clients_set=self.selected_clients, id2params=self.client_id2lora, 
    #                id2weight=self.client_id2weight, epoch=self.config.num_communication_rounds - self.global_rounds_remain)
                   
    #         self.selected_clients = client_selection(self.com_manager.get_all_client_id(), self.config.client_selection_frac)
    #         self.client_id2lora.clear()
    #         self.client_id2weight.clear()
    #         self.global_rounds_remain -= 1
            
    #         for i in self.selected_clients:
    #             self.send_message(i, defs.CellChannel.CLIENT_MAIN, defs.CellChannelTopic.Update, 
    #                              update=agg_res, epoch=self.config.num_communication_rounds - self.global_rounds_remain)

    # def _register_all_callback(self):
    #     if self.mode != "cluster":
    #         self.logger.warning("在非集群模式下调用了注册回调方法")
    #         return
    #     self._register_handler(defs.CellChannelTopic.Share, self.agg_params)
    
    # 统一的部署方法
    def deploy(self):
        
        self.cell = get_cell(self.config)
        self.cell.start()
        self._register_server_cbs()
        self.cell.stop()
        