
from openhufu.private.net.client_cell import ClientCell
from openhufu.private.utlis.config_class import ClientConfig
from openhufu.private.utlis.util import get_logger
from openhufu.private.utlis.factory import get_cell
class FederatedClient:
    def __init__(self, config: ClientConfig):
        self.cell = None
        self.config : ClientConfig = config
        self.logger = get_logger(__name__)
    
    def _create_cell(self):
        # self.cell = ClientCell(config=self.config)
        self.cell = get_cell(config=self.config)
        self.cell.start()
        
        
        
    def set_up(self):    
        schema_location = self.config.host + ":" + str(self.config.port)
        self.config.addr = schema_location
        
        if not self.cell:
            self._create_cell()
        
    
    def register(self):
        self.cell.register()
    
    
    def stop(self):
        self.cell.stop()