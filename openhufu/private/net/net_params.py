
from abc import ABC
from dataclasses import dataclass
from enum import Enum

from openhufu.private.drivers.driver import Driver
from openhufu.private.net.connection_manager import ConnMonitor

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
    uid: str = ""
    driver: Driver = None
    monitor: ConnMonitor = None
    params: ConParams = None
    mode: DriverMode
    started: bool = False
