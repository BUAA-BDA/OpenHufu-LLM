import threading
import msgpack
from abc import ABC, abstractmethod
from enum import Enum


from openhufu.private.net.net_params import DriverInfo
from openhufu.private.net.endpoint import Endpoint


count_lock = threading.Lock()
count = 0

def get_connection_name():
    global count
    with count_lock:
        count += 1
    return f"conn_{count:05d}"

    
class ConnState(Enum):
    CREATED = 1
    CONNECTED = 2
    CLOSED = 3
    

class FrameReceiver:
    @abstractmethod
    def process_frame(self, frame):
        pass
    

class Connection(ABC):
    """Connection class for handling network connections.

    Args:
        ABC (_type_): _description_
    """
    
    def __init__(self, driverInfo: DriverInfo):
        self.name = get_connection_name()
        self.state = ConnState.CREATED
        self.driverInfo = driverInfo
        self.frame_receiver = None
    
    
    def register_frame_receiver(self, receiver: FrameReceiver):
        self.frame_receiver = receiver
    
    
    def get_name(self):
        return self.name
    
    
    @abstractmethod
    def process_frames(self, request_iterator):
        pass
    
    
    @abstractmethod
    def send_frame(self, frame):
        pass
    


