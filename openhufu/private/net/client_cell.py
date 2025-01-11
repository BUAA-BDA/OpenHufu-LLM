

from openhufu.private.net.cell import Cell
from openhufu.private.utlis.config_class import ClientConfig

class ClientCell(Cell):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.is_client = True


    def register(self):
        pass