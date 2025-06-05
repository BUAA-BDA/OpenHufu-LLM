
from openhufu.private.utlis.config_class import ClientConfig
from openhufu.private.utlis.util import get_logger
from openhufu.private.client.fed_client import FederatedClient
from openhufu.private.client.fed_client_lora import FederatedClient_LoRA
from openhufu.private.utlis.factory import get_cell

class ClientDeployer:
    def __init__(self, config: ClientConfig):
        self.config : ClientConfig = config 
        self.logger = get_logger(__name__)


    def create_client(self):
        self.config.addr = self.config.host + ":" + str(self.config.port)
        cell = get_cell(config=self.config)
        # cell.start()
        # cell先于client启动有没有影响
        client = FederatedClient_LoRA(config=self.config, cell=cell)
        self.logger.info(f"Client {self.config} created")
        
        return client
 
    