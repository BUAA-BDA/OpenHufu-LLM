

from openhufu.private.utlis.defs import CellChannel, CellChannelTopic
from openhufu.private.utlis.config_class import BaseConfig, ServerConfig, ClientConfig
from openhufu.private.net.connection_manager import ConnManager
from openhufu.private.net.net_params import ConParams, DriverMode, DriverInfo
from openhufu.private.drivers.stream_grpc_driver import GrpcDriver
from openhufu.private.net.endpoint import Endpoint


class Cell:
    msg_queue = []
    
    def __init__(self, config: BaseConfig):
        self.node_info : BaseConfig = config
        
        endpoint = Endpoint(name=self.node_info.name, addr=self.node_info.addr)
        
        self.conn_manager = ConnManager(self, endpoint)
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
        
        if channel not in self.registered_cbs:
            self.registered_cbs[topic] = cb
            
            
    def process_message(self, msg):
        # TODO: Implement message processing
        if True:
            self.registered_cbs[CellChannelTopic.Register](msg)