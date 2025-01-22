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
from openhufu.private.net.prefix import Prefix, FrameType, PREFIX_LEN
from openhufu.private.net.message import Message
from openhufu.private.utlis.defs import HeaderKey
import time



con_lock = threading.Lock()
con_count = 0

def get_connection_uid():
    global con_count
    with con_lock:
        con_count += 1
    return f"conn_{con_count:05d}"

        
    
class ConnManager(ConnMonitor):
    def __init__(self, local_endpoint: Endpoint):
        super().__init__()
        self.lock = threading.Lock()
        
        self.logger = get_logger("ConnManager")
        self.driverInfos: Dict[str, DriverInfo]= {}
        
        self.started = False
        self.conn_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="conn_manager")
        self.frame_manager_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="frame_manager")
        self.message_receiver = None
        
        self.local_endpoint = local_endpoint
        
        self.connections : Dict[str, ConnectionWrapper] = {}
        self.endpoints : Dict[str, EndpointWrapper] = {}
    
    
    def register_message_receiver(self, cell):
        self.message_receiver = cell
    
    
    def add_connection_driver(self, driver: Driver, params: ConParams, mode: DriverMode):
        uid = get_connection_uid()
        driver.register_monitor(self)
        driverInfo = DriverInfo(uid=uid, driver=driver, params=params, mode=mode, monitor=self, started=False)
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
        
        
    def stop(self):
        with self.lock:
            self.started = False
            for driverInfo in self.driverInfos.values():
                driverInfo.driver.stop()
                
            self.driverInfos.clear()
            
        with self.lock:
            for conn in self.connections.values():
                conn.close()
            
        self.connections.clear()
        
        self.conn_executor.shutdown()
        self.frame_manager_executor.shutdown()
        

        
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
            
        connection.register_frame_receiver(FrameProcessor(connWrapper=connWrapper, conn_manager=self))
        
        # TODO: send remember the target address
        if connection.get_name() not in self.connections:
            with self.lock:
                self.connections[connection.get_name()] = connWrapper
        
        self.logger.info(f"New connection: {connection.get_name()}")
        
        # if on client end, send the endpoint msg to client.
        if connection.driverInfo.mode == DriverMode.CLIENT:
            self.logger.info(f"Sending handshake to {connection.get_name()}")
            connWrapper.send_handshake(FrameType.HELLO)
        
    
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
    
    
    def update_endpoint(self, message: Message, connWrapper: ConnectionWrapper):
        try:
            endpoint_name = message.get_from_headers(HeaderKey.SOURCE_ENDPOINT)
        except Exception as e:
            self.logger.error(f"Invalid message, no endpoint name in message: {message}")
            raise e
        
        if endpoint_name not in self.endpoints.keys():
            self.logger.info(f"New endpoint: {endpoint_name}")
            endpont_wrapper = EndpointWrapper(Endpoint(name=endpoint_name))
            self.endpoints[endpoint_name] = endpont_wrapper
        
        endpont_wrapper = self.endpoints[endpoint_name]
        endpont_wrapper.add_connection(connWrapper)
    
    
    def _process_frame_task(self, connWrapper: ConnectionWrapper, frame):
        # TODO: Unwrap the frame to message
        ct = threading.current_thread()
        self.logger.info(f"Processing frame from {connWrapper.get_name()} on thread {ct.name}")
        try:
            # todo concat the frame
            prefix : Prefix = Prefix.parse(frame)
            message = msgpack.unpackb(frame[PREFIX_LEN:])
            message = Message.from_dict(message)
        except Exception as e:
            self.logger.error(f"Error unpacking frame: {e}")
        
    
        if FrameType(prefix.frame_type) in {FrameType.HELLO, FrameType.HI}:
            if prefix.frame_type == FrameType.HELLO.value:
                self.logger.info(f"Received HELLO from {connWrapper.get_name()}, sending HI")
                connWrapper.send_handshake(FrameType.HI)
            else:
                self.logger.info(f"Received HI from {connWrapper.get_name()}")
            self.update_endpoint(message, connWrapper)
        else:
            self._process_message(message)
    
    
    def _process_message(self, messgae: Message):
        self.message_receiver.process_message(messgae)

    
    def send_message(self, message: Message):
        destination_endpoint = message.get_from_headers(HeaderKey.DESTINATION_ENDPOINT)
        
        timeout = 5
        wait_interval = 0.1
        start_time = time.time()
        with self.lock:
            while destination_endpoint not in self.endpoints:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    self.logger.error(f"Endpoint {destination_endpoint} not available")
                    raise Exception(f"Endpoint {destination_endpoint} not available")
                self.lock.release()
                time.sleep(wait_interval)
                self.lock.acquire()               
                
        endpoint_wrapper = self.endpoints.get(destination_endpoint)
        if endpoint_wrapper is None:
            self.logger.error(f"Endpoint {destination_endpoint} not found")
            raise Exception(f"Endpoint {destination_endpoint} not found")
        stream_id = endpoint_wrapper.next_stream_id()
        conn_wrapper = endpoint_wrapper.get_connection(stream_id)
        
        prefix = Prefix(0, stream_id=stream_id, frame_type=FrameType.DATA.value)
        conn_wrapper.send_data(prefix=prefix, message=message)
        
        
class FrameProcessor(FrameReceiver):
    def __init__(self, connWrapper: ConnectionWrapper, conn_manager: ConnManager):
        self.connWrapper = connWrapper
        self.conn_manager = conn_manager
        self.logger = get_logger("FrameProcessor")
        self.frame_buffer : Dict[int, List[bytearray]] = {}
    
    
    def process_frame(self, frame_data):
        prefix : Prefix = Prefix.parse(frame_data)
        if prefix.has_next == 1:
            self.logger.info(f"Appending frame")
            self.frame_buffer.setdefault(prefix.stream_id, []).append(frame_data[PREFIX_LEN:])
        else :
            if len(self.frame_buffer) > 0:
                self.logger.info(f"Concating frame")
                self.frame_buffer.setdefault(prefix.stream_id, []).append(frame_data[PREFIX_LEN:])
                tot_buf_len = sum([len(buf) for buf in self.frame_buffer.get(prefix.stream_id, [])])
                tot_buf: bytearray = bytearray(tot_buf_len + PREFIX_LEN)
                prefix.to_buffer(tot_buf)
                tot_buf[PREFIX_LEN:] = b"".join(self.frame_buffer.get(prefix.stream_id, []))
                frame_data = tot_buf
                del self.frame_buffer[prefix.stream_id]
            
            self.conn_manager.process_frame(self.connWrapper, frame_data)