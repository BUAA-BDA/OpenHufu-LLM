import threading
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from enum import Enum

from openhufu.private.net.net_params import ConParams, DriverInfo
from openhufu.private.net.connection import Connection, ConnState


class ConnMonitor(ABC):
    @abstractmethod
    def state_change(self, connection: Connection):
        pass


class Driver:
    def __init__(self):
        self.connections : Dict[str, Connection] = {}
        self.conn_lock = threading.Lock()
        self.conn_monitor = None
        
        
    @abstractmethod
    def connect(self, connection: DriverInfo):
        pass
    
    
    @abstractmethod
    def listen(self, connection: DriverInfo):
        pass
    
    @abstractmethod
    def close(self):
        pass
    
    def add_connection(self, connection: Connection):
        connection.state = ConnState.CONNECTED
        with self.conn_lock:
            self.connections[connection.name] = connection    
        
        self._notify_monitor(connection)
    
    
    def close_connection(self, connection: Connection):
        connection.state = ConnState.CLOSED
        with self.conn_lock:
            if connection.name in self.connections:
                del self.connections[connection.name]
                
        self._notify_monitor(connection)
    
    
    def register_monitor(self, monitor: ConnMonitor):
        self.conn_monitor = monitor
    
        
    def _notify_monitor(self, connection: Connection):
        self.conn_monitor.state_change(connection)
        
    
    def close_all_connections(self):
        with self.conn_lock:
            for name in self.connections.keys():
                self.connections[name].close()
        
        


