
from abc import ABC
from dataclasses import dataclass
from enum import Enum


MAX_FRAME_SIZE = 2 * 1024 * 1024 * 1024 - 2 * 1024 * 1024 # 2GB - 2MB

@dataclass
class ConParams:
    host: str = ""
    port: int = 0
    addr: str = ""
    scheme: str = ""
    
    

class DriverMode(Enum):
    SERVER = 1
    CLIENT = 2
    
    
@dataclass
class DriverInfo:
    uid: str
    driver: "Driver"
    monitor: "ConnMonitor"
    params: ConParams 
    mode: DriverMode
    started: bool = False
    

    