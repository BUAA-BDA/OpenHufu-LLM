

from openhufu.private.utlis.defs import CellChannel, CellChannelTopic
from openhufu.private.utlis.config_class import BaseConfig, ServerConfig, ClientConfig
from openhufu.private.net.connection_manager import ConnManager
from openhufu.private.net.net_params import ConParams, DriverMode, DriverInfo
from openhufu.private.drivers.stream_grpc_driver import GrpcDriver
from openhufu.private.net.endpoint import Endpoint
from openhufu.private.net.message import Message
from openhufu.private.utlis.util import get_logger
from openhufu.private.utlis.defs import HeaderKey, CellChannelTopic


class Cell:
    msg_queue = []
    
    def __init__(self, config: BaseConfig):
        self.node_info : BaseConfig = config
        self.logger = get_logger(__name__)
        
        self.local_endpoint = Endpoint(name=self.node_info.name)
        
        self.conn_manager = ConnManager(self.local_endpoint)
        
        self.conn_manager.register_message_receiver(self)
        
        self.registered_cbs = {}
        
        self.is_client = None

        
        
    def _create_external_listener(self):
        driver = GrpcDriver()
        params = ConParams(host=self.node_info.host, port=self.node_info.port, addr=self.node_info.addr, scheme=self.node_info.schema)
        mode = DriverMode.SERVER
        self.conn_manager.add_connection_driver(driver=driver, params=params, mode=mode)
        
        
    def _create_external_connector(self):
        driver = GrpcDriver()
        params = ConParams(host=self.node_info.host, port=self.node_info.port, addr=self.node_info.addr, scheme=self.node_info.schema)
        mode = DriverMode.CLIENT
        self.conn_manager.add_connection_driver(driver=driver, params=params, mode=mode)
    
    
    def _start_cell_for_client(self):
        self._create_external_connector()
    
    
    def _start_cell_for_server(self):
        self._create_external_listener()
    
    
    def start(self):
        if self.is_client:
            self._start_cell_for_client()
        else:
            self._start_cell_for_server()

        self.conn_manager.start()
        
    
    def stop(self):
        self.conn_manager.stop()
        
    
    def register_request_cb(self, channel: CellChannel, topic: CellChannelTopic, cb):
        
        if not callable(cb):
            raise Exception("Callback is not callable")
        
        if topic not in self.registered_cbs.keys():
            self.registered_cbs[topic] = cb
            
            
    def process_message(self, message: Message):
        channel_topic = message.get_from_headers(HeaderKey.CHANNEL_TOPIC)
        if channel_topic == CellChannelTopic.Register:
            self.registered_cbs[CellChannelTopic.Register](message)
            
            
    def send_message(self, message: Message):
        self.conn_manager.send_message(message)