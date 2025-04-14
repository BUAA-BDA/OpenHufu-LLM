import os

from openhufu.private.net.cell import Cell
from openhufu.private.utlis.config_class import ClientConfig
from openhufu.private.net.message import Message
from openhufu.private.utlis.defs import HeaderKey, CellChannelTopic, CellChannel


class ClientCell(Cell):
    def __init__(self, config: ClientConfig):
        super().__init__(config)
        self.node_info : ClientConfig = config
        self.is_client = True

    def register(self):
        try:
            self.logger.info(f"Registering client {self.node_info.name} to server {self.node_info.server_name}")
            message = Message(headers={HeaderKey.SOURCE_ENDPOINT: self.node_info.name,
                            HeaderKey.DESTINATION_ENDPOINT: self.node_info.server_name,
                            HeaderKey.CHANNEL: CellChannel.CLIENT_MAIN,
                            HeaderKey.CHANNEL_TOPIC: CellChannelTopic.Register}, 
                            data=None)
            data = os.urandom(3 * 1024 * 1024 * 1024)
            message.set_data(data)
            self.send_message(message)
        except Exception as e:
            self.logger.error(f"Error registering client: {e}", exc_info=True)