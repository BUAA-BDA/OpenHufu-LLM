import msgpack

from openhufu.private.net.connection import Connection
from openhufu.private.net.endpoint import Endpoint
from openhufu.private.net.prefix import Prefix, FrameType, PREFIX_LEN
from openhufu.private.net.message import Message
from openhufu.private.utlis.defs import HeaderKey
from openhufu.private.utlis.util import get_logger
from openhufu.private.net.net_params import MAX_FRAME_SIZE


logger = get_logger(__name__)


class ConnectionWrapper:
    """
    
    
    """
    def __init__(self, connection: Connection, local_endpoint: Endpoint):
        self.connection = connection
        self.local_endpoint = local_endpoint
        

    def get_name(self):
        return self.connection.get_name()
    
    
    def get_endpoint_name(self):
        return self.local_endpoint.name
    
    
    def send_handshake(self, frame_type: FrameType):
        "send HELLO frame from client to server"
        try:
            data = None
            prefix = Prefix(0, 0, frame_type.value)
            message = Message({HeaderKey.SOURCE_ENDPOINT: self.local_endpoint.name}, data)
            self.send_data(prefix, message)
        except Exception as e:
            logger.error(f"Error sending handshake: {e}")
    
    def send_data(self, prefix: Prefix, message: Message):
        try:
            message_b = msgpack.packb(message)
            
            length = PREFIX_LEN + len(message_b)
            logger.info(f"Sending data of length {length}")
            if length > MAX_FRAME_SIZE:
                logger.info(f"Message too large, splitting into fragments")
                fragment_size = MAX_FRAME_SIZE - PREFIX_LEN
                for i in range(0, len(message_b), fragment_size):
                    buffer: bytearray = bytearray(fragment_size + PREFIX_LEN)
                    prefix_tmt = Prefix(PREFIX_LEN + len(message_b[i: i + fragment_size]), 0, prefix.frame_type, 1 if i + fragment_size < len(message_b) else 0)
                    prefix_tmt.to_buffer(buffer)
                    buffer[PREFIX_LEN:] = message_b[i: i + fragment_size]
                    logger.info(f"Sending fragment {i // fragment_size + 1} of {len(message_b) // fragment_size + 1}, frame length {len(buffer)}")
                    self.connection.send_frame(buffer)
            else:
                prefix.length = length
                buffer: bytearray = bytearray(prefix.length)
                prefix.to_buffer(buffer)
                buffer[PREFIX_LEN:] = message_b
                self.connection.send_frame(buffer)
        except Exception as e:
            logger.error(f"Error sending data: {e}")
        
        
    def close(self):
        self.connection.close()