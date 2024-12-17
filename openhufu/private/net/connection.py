import threading
from abc import ABC, classmethod
from dataclasses import dataclass
from enum import Enum

from openhufu.private.net.frame_receiver import FrameReceiver
from openhufu.private.net.net_params import DriverInfo

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
    
    @classmethod
    def process_frames(self, request_iterator):
        pass
    
    