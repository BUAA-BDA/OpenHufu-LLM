
from openhufu.private.utlis.config_class import ServerConfig
from openhufu.private.utlis.util import get_logger
from openhufu.private.server.fed_server import FederatedServer


class ServerDeployer:
    def __init__(self, config: ServerConfig):
        self.config : ServerConfig = config 
        self.server = None
        self.logger = get_logger(__name__)


    def create_server(self):
        self.config.addr = self.config.host + ":" + str(self.config.port)
        
        server = FederatedServer(config=self.config)
        self.logger.info(f"Server {self.config} created")
        
        return server
    
    
    def deploy(self):
        self.server = self.create_server()
        
        self.server.deploy()
        