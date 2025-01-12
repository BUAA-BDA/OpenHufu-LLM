

from openhufu.private.utlis.util import get_logger
from openhufu.private.utlis.defs import CellChannel, CellChannelTopic
from openhufu.private.utlis.config_class import ServerConfig
from openhufu.private.net.cell import Cell

class FederatedServer:
    def __init__(self, config: ServerConfig):
        self.cell = None
        self.config : ServerConfig = config
        self.logger = get_logger(__name__)


    def _register_server_cbs(self):
        # self.cell.register_request_cb(CellChannel.SERVER_MAIN, CellChannelTopic.Challenge, self.client_challenge)
        self.cell.register_request_cb(CellChannel.SERVER_MAIN, CellChannelTopic.Register, self.client_register)
        
    
    def client_challenge(self, request):
        pass
    
    
    def client_register(self, request):
        self.logger.info("Client registered")
    
    
    def deploy(self):
        self.cell = Cell(self.config)
        self.cell.start()
        
        self._register_server_cbs()
        
        self.cell.stop()
        
    
        