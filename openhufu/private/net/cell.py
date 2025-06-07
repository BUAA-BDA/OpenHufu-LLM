

from openhufu.private.utlis.defs import CellChannel, CellChannelTopic
from openhufu.private.utlis.config_class import BaseConfig, ServerConfig, ClientConfig
from openhufu.private.net.connection_manager import ConnManager
from openhufu.private.net.net_params import ConParams, DriverMode, DriverInfo
from openhufu.private.drivers.stream_grpc_driver import GrpcDriver
from openhufu.private.net.endpoint import Endpoint
from openhufu.private.net.message import Message
from openhufu.private.utlis.util import get_logger
from openhufu.private.utlis.defs import HeaderKey, CellChannelTopic

logger = get_logger("cell")

class Cell:
    msg_queue = []
    
    def __init__(self, config):
        self.node_info = config
        self.logger = get_logger(__name__)
        
        self.local_endpoint = Endpoint(name=self.node_info.name)
        
        self.conn_manager = ConnManager(self.local_endpoint)
        
        self.conn_manager.register_message_receiver(self)
        
        self.registered_cbs = {}
        self.id2worker = {}
        self.is_client = None
        self.id_counter = 0

        
        
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
        logger.info(f"handler 4 {topic} registered")
        if not callable(cb):
            raise Exception("Callback is not callable")
        
        if topic not in self.registered_cbs.keys():
            self.registered_cbs[topic] = cb
            
            
    def process_message(self, message: Message):
        try :
            channel_topic = message.get_from_headers(HeaderKey.CHANNEL_TOPIC)
            logger.info(f"recv message {channel_topic} ")
            if channel_topic == CellChannelTopic.Register:
                source_endpoint = message.get_from_headers(HeaderKey.SOURCE_ENDPOINT)
                client_id = self.id_counter  
                self.id_counter += 1
                self.id2worker[client_id] = source_endpoint
                self.registered_cbs[CellChannelTopic.Register](client_id)
            elif channel_topic in self.registered_cbs:
                # print(message['data'])
                # 取回来的data没什么问题
                filtered_dict = {k: v for k, v in message['data'].items() if k not in ['target','channel', 'topic']}
                for k,v in filtered_dict.items():
                    print(k)
                self.registered_cbs[channel_topic](**filtered_dict)
        except Exception as e:
            logger.error(f"process message failed because of {e}")
            
    def _send_message(self, message: Message):
        self.conn_manager.send_message(message)

    def send_message(self, target , channel: CellChannel , topic: CellChannelTopic, **kwargs):
        # print(self.registered_cbs[topic])
        # print(self.id2worker[target])
        # callback = getattr(self.id2worker[target], self.registered_cbs[topic], None)
        # logger.info(f"woker:{target} handle message:{topic} with {callback.__name__}")
        # callback(**kwargs)
        # target 2 endpoint
        logger.info(f'send message to worker {self.id2worker[target]}')
        message = Message(headers={HeaderKey.SOURCE_ENDPOINT: self.node_info.name,
                            # HeaderKey.DESTINATION_ENDPOINT: self.node_info.server_name,
                            HeaderKey.DESTINATION_ENDPOINT: self.id2worker[target],
                            HeaderKey.CHANNEL: channel,
                            HeaderKey.CHANNEL_TOPIC: topic}, 
                            data=kwargs)
        self.conn_manager.send_message(message)

    def get_all_client_id(self):
        return [key for key in self.id2worker.keys() if key != -1]