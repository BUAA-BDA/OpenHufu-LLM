import msgpack

from openhufu.private.net.connection import Connection
from openhufu.private.net.endpoint import Endpoint


class ConnectionWrapper:
    def __init__(self, connection: Connection, local_endpoint: Endpoint):
        self.connection = connection
        self.local_endpoint = local_endpoint
        

    def get_name(self):
        return self.connection.get_name()
    
    
    def send_handshake(self):
        "send HELLO frame from server to client"
        
        data = {"endpoint_name": self.local_endpoint.name}
        self.send_dict(1, 1, data=data)
    
    
    def send_data(self, data):
        self.connection.send_frame(data)
        
    
    def send_dict(self, frame_type: int, stream_id: int, data: dict):
        "send a dict as payload"
        payload = msgpack.packb(data)
        
        self.send_frame(None, None, payload=payload)

    
    def send_frame(self, prefix, headers, payload):
        buffer: bytearray = bytearray(100)
        
        buffer[0:] = payload

        self.connection.send_frame(buffer)