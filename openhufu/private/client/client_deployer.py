
from openhufu.private.utlis.config_class import ClientConfig
from openhufu.private.utlis.util import get_logger
from openhufu.private.client.fed_client import FederatedClient


class ClientDeployer:
    def __init__(self, config: ClientConfig):
        self.config : ClientConfig = config 
        self.logger = get_logger(__name__)


    def create_client(self):
        self.config.addr = self.config.host + ":" + str(self.config.port)
        
        client = FederatedClient(config=self.config)
        self.logger.info(f"Client {self.config} created")
        
        return client
 
    