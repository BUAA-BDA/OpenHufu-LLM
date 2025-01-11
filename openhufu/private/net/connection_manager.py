import threading
import msgpack
from abc import ABC, abstractmethod
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor

from openhufu.private.utlis.util import get_logger
from openhufu.private.net.connection import Connection, ConnState, FrameReceiver
from openhufu.private.drivers.driver import Driver, ConnMonitor
from openhufu.private.net.net_params import ConParams, DriverMode, DriverInfo
from openhufu.private.net.endpoint import Endpoint
from openhufu.private.net.connection_wrapper import ConnectionWrapper
from openhufu.private.net.endpoint_wrapper import EndpointWrapper


con_lock = threading.Lock()
con_count = 0
def get_connection_uid():
    with con_lock:
        con_count += 1
    return f"conn_{con_count:05d}"

        
    
class ConnManager(ConnMonitor):
    def __init__(self, cell, local_endpoint: Endpoint):
        super().__init__()
        self.lock = threading.Lock()
        
        self.logger = get_logger("ConnManager")
        self.driverInfos: Dict[str, DriverInfo]= {}
        
        self.started = False
        self.conn_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="conn_manager")
        self.frame_manager_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="frame_manager")
        self.message_receiver = cell
        
        self.local_endpoint = local_endpoint
        
        self.connections = Dict[str, ConnectionWrapper]
    
    
    def add_connection_driver(self, driver: Driver, params: ConParams, mode: DriverMode):
        uid = get_connection_uid()
        driver.register_monitor(self)
        driverInfo = DriverInfo(uid=uid, driver=driver, params=params, mode=mode)
        with self.lock:
            self.driverInfos[uid] = driverInfo
        
        if self.started:
            self._start_driver(driverInfo)
            
        
    def _start_driver(self, driverInfo: DriverInfo):
        if driverInfo.started:
            return

        self.conn_executor.submit(self.start_driver_task, driverInfo)
        
        
    def start(self):
        with self.lock:
            self.started = True
            for driverInfo in self.driverInfos.values():
                self._start_driver(driverInfo)
        
        
    def start_driver_task(self, driverInfo: DriverInfo):
        driverInfo.started = True
        
        if driverInfo.mode == DriverMode.CLIENT:
            starter = driverInfo.driver.connect
        else:
            starter = driverInfo.driver.listen
            
        try:
            # TODO: 记录目标地址
            starter(driverInfo)
        except Exception as e:
            self.logger.error(f"Error starting driver {driverInfo.uid}: {e}")
        
    
    def _handle_new_connection(self, connection: Connection):
        connWrapper = ConnectionWrapper(connection, self.local_endpoint)
        
        with self.lock:
            self.connections[connection.get_name()] = connWrapper
            
        connection.register_frame_receiver(FrameReceiverWrapper(connWrapper=connWrapper, conn_manager=self))
        
        # TODO: send remember the target address
        if connection.get_name() not in self.connections:
            with self.lock:
                self.connections[connection.get_name()] = connWrapper
        
        self.logger.info(f"New connection: {connection.get_name()}")
        
        # if on server end, send the endpoint msg to client.
        if connection.driverInfo.mode == DriverMode.SERVER:
            connWrapper.send_handshake()
        
    
    def _close_connection(self, connection: Connection):
        with self.lock:
            if connection.get_name() in self.connections:
                del self.connections[connection.get_name()]
               
                
    def state_change(self, connection: Connection):
        try:
            state = connection.state
            if state == ConnState.CONNECTED:
                self._handle_new_connection(connection)
            elif state == ConnState.CLOSED:
                self._close_connection(connection)
        except Exception as e:
           self.logger.error(f"Error handling connection state change: {e}")
            
            
    def process_frame(self, connWrapper: ConnectionWrapper, frame):
        self.frame_manager_executor.submit(self._process_frame_task, connWrapper, frame)
    
    
    def _process_frame_task(self, connWrapper: ConnectionWrapper, frame):
        # TODO: Unwrap the frame to message
        self._process_message(frame)
    
    
    def _process_message(self, messgae):
        self.message_receiver.process_message(messgae)

    
    def send_message(self, endpoint, frame):
        
        if self.endpoint_wrapper is None:
            raise Exception(f"Endpoint {endpoint} not found")
        
        
        
class FrameReceiverWrapper(FrameReceiver):
    def __init__(self, connWrapper: ConnectionWrapper, conn_manager: ConnManager):
        self.connWrapper = connWrapper
        self.conn_manager = conn_manager
    
    
    def process_frame(self, frame_data):
        self.conn_manager.process_frame(self.connWrapper, frame_data)