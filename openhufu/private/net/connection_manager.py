import threading
from abc import ABC, classmethod
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

from openhufu.private.net.connection import Connection, ConnState
from openhufu.private.drivers.driver import Driver
from openhufu.private.net.net_params import ConParams, DriverMode, DriverInfo
from openhufu.private.net.frame_receiver import FrameReceiver

con_lock = threading.Lock()
con_count = 0
def get_connection_uid():
    with con_lock:
        con_count += 1
    return f"conn_{con_count:05d}"



class ConnMonitor(ABC):
    @classmethod
    def state_change(self, connection: Connection):
        pass




class ConnManager(ConnMonitor):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.driverInfos: Dict[str, DriverInfo]= {}
        
        self.started = False
        self.conn_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="conn_manager")
        
        
    def add_connection_driver(self, driver: Driver, params: ConParams, mode: DriverMode):
        uid = get_connection_uid()
        driver.register_monitor(self)
        driverInfo = DriverInfo(uid=uid, driver=driver, params=params, mode=mode)
        with self.lock:
            self.driverInfos[uid] = driverInfo
        
        if self.started:
            self.start_connection_driver(driverInfo)
            
        
    def start_connection_driver(self, driverInfo: DriverInfo):
        if driverInfo.started:
            return

        self.conn_executor.submit(self.start_driver_task, driverInfo)
        
        
    def start_driver_task(self, connectionInfo: DriverInfo):
        connectionInfo.started = True
        
        if connectionInfo.mode == DriverMode.CLIENT:
            starter = connectionInfo.driver.connect
        else:
            starter = connectionInfo.driver.listen
            
        starter(connectionInfo)
        
    
    def handle_new_connection(self, connection: Connection):
        connection.register_frame_receiver(FrameReceiver(connection, self))
    
    
    def close_connection(self, connection: Connection):
        pass
 
    
    def state_change(self, connection: Connection):
        try:
            state = connection.state
            if state == ConnState.CONNECTED:
                self.handle_new_connection(connection)
            elif state == ConnState.CLOSED:
                self.close_connection(connection)
        except Exception as e:
            print("Error: ", e)
            
            
    def process_frame(self, connection: Connection, frame):
        pass
    
    
    def _process_message(self, messgae):
        pass
