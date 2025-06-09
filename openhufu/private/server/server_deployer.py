
from openhufu.private.utlis.config_class import ServerConfig
from openhufu.private.utlis.util import get_logger
from openhufu.private.server.fed_server import FederatedServer
from openhufu.private.utlis.factory import get_cell

class ServerDeployer:
    def __init__(self, config):
        self.config  = config 
        self.server = None
        self.logger = get_logger(__name__)


    def create_server(self):
        self.config.addr = self.config.host + ":" + str(self.config.port)
        cell = get_cell(config=self.config)
        server = FederatedServer(config=self.config, cell=cell)
        self.logger.info(f"Server {self.config} created")
        
        return server
    
    
    def deploy(self):
        self.server = self.create_server()
        
        self.server.deploy()
        