import threading
from abc import classmethod
from dataclasses import dataclass
from typing import Dict

from openhufu.private.net.net_params import ConParams, DriverMode, DriverInfo
from openhufu.private.net.connection import Connection, ConnState
from openhufu.private.net.connection_manager import ConnMonitor


class Driver:
    def __init__(self):
        self.connections : Dict[str, Connection] = {}
        self.conn_lock = threading.Lock()
        self.conn_monitor = None
        
    @classmethod
    def connect(self, connection: DriverInfo):
        pass
    
    @classmethod
    def listen(self, connection: DriverInfo):
        pass
    
    def add_connection(self, connection: Connection):
        connection.state = ConnState.CONNECT
        with self.conn_lock:
            self.connections[connection.name] = connection    
        
        self._notify_monitor(connection)
    
    def close_connection(self, connection: Connection):
        connection.state = ConnState.CLOSED
        with self.conn_lock:
            if connection.name in self.connections:
                del self.connections[connection.name]
    
    def register_monitor(self, monitor: ConnMonitor):
        self.conn_monitor = monitor
        
    def _notify_monitor(self, connection: Connection):
        self.conn_monitor.state_change(connection)