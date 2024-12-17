

from openhufu.private.net.connection import Connection
from openhufu.private.net.connection_manager import ConnManager

class FrameReceiver:
    def __init__(self, connection: Connection, conn_manager: ConnManager):
        self.connection = connection
        self.conn_manager = conn_manager
    
    def process_frame(self, frame_data):
        self.conn_manager.process_frame(self.connection, frame_data)