import threading
from typing import Dict, List, Optional

from openhufu.private.net.connection_wrapper import ConnectionWrapper
from openhufu.private.net.endpoint import Endpoint
from openhufu.private.utlis.util import get_logger


logger = get_logger(__name__)

RESERVED_STREAM_ID = 16
MAX_CONN_PER_ENDPOINT = 1


class EndpointWrapper:
    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        self.stream_id : int = RESERVED_STREAM_ID
        self.lock = threading.Lock()
        self.connections : List[ConnectionWrapper] = []


    def add_connection(self, connection: ConnectionWrapper):
        with self.lock:
            while len(self.connections) >= MAX_CONN_PER_ENDPOINT:
                first = self.connections[0]
                first.connection.close()
                self.connections.pop(0)
                logger.info(f"Connection {first.connection.get_name()} closed from endpoint {self.endpoint.name}")
                
            self.connections.append(connection)
            
            
    def remove_connection(self, connection: ConnectionWrapper):
        pass
    
    
    def get_connection(self, stream_id: int) -> Optional[ConnectionWrapper]:
        if not self.connections:
            return None
        
        index = stream_id % len(self.connections)
        return self.connections[index]
    
    
    def next_stream_id(self) -> int:
        with self.lock:
            self.stream_id = (self.stream_id + 1) & 0xFFFF
            return self.stream_id
