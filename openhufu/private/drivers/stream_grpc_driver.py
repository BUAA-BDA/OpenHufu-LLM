import time
import grpc
import queue
from concurrent import futures
import threading
import signal

from openhufu.private.net.connection import Connection
from openhufu.private.net.net_params import DriverInfo
from openhufu.private.drivers.proto.grpc_stream_pb2_grpc import grpcStreamFuncServicer, add_grpcStreamFuncServicer_to_server, grpcStreamFuncStub
from openhufu.private.drivers.proto.grpc_stream_pb2 import Frame
from openhufu.private.drivers.driver import Driver
from openhufu.private.utlis.util import get_logger
from openhufu.private.net.net_params import MAX_FRAME_SIZE

logger = get_logger("stream_grpc_driver")

options = [
    ('grpc.max_send_message_length', MAX_FRAME_SIZE + 8),
    ('grpc.max_receive_message_length', MAX_FRAME_SIZE + 8)
]

class StreamConnection(Connection):
    """StreamConnection class for handling network connections.

    Args:
        Connection (_type_): _description_
    """
    seq_num = 0
    
    def __init__(self, driverInfo: DriverInfo):
        super().__init__(driverInfo=driverInfo)
        self.closed = False
        self.frame_queue = queue.Queue()
        self.queue_lock = threading.Lock()  
    
    
    def send_frame(self, frame):
        try:
            StreamConnection.seq_num += 1
            with self.queue_lock:
                self.frame_queue.put(Frame(seq=StreamConnection.seq_num, data=bytes(frame)))
        except Exception as e:
            logger.error(f"Error sending frame: {e}")
            
    
    def iter_queue(self):
        """迭代队列中的帧"""
        if self.closed:
            raise StopIteration
        
        while True: 
            try:
                yield self.frame_queue.get(timeout=10)
            except queue.Empty:
                # 如果队列为空且超时，记录警告并继续检查状态
                if self.closed:
                    logger.info("Connection closed, stopping frame iteration.")
                    raise StopIteration
                logger.warning("Frame queue is empty or processing timeout.")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Error iterating frame queue: {e}")
                raise StopIteration
    
            
    def process_frames(self, request_iterator):
        ct = threading.current_thread()
        
        try:
            for frame in request_iterator:
                if not isinstance(frame, Frame):
                    logger.error(f"Invalid item in frame_queue: expected Frame, got {type(frame).__name__}")
                    continue
                
                if self.frame_receiver:
                    self.frame_receiver.process_frame(frame.data)
                else:
                    logger.error("No frame receiver")
        except Exception as e:
            logger.error(f"Error processing frames: {e}")
 
 
    def close(self):
        self.closed = True
        # self.frame_queue.shutdown()
    
    
    def generate_output(self):
        try:
            ct = threading.current_thread()
            for i in self.iter_queue():
                if not isinstance(i, Frame):
                    error_message = f"Invalid item in frame_queue: expected Frame, got {type(i).__name__}"
                    logger.error(error_message)
                    continue
                    
                yield i
        except Exception as e:
            logger.error(f"Error generating output: {e}")
        finally:
            logger.info("Output generation complete")
        
    
class Servicer(grpcStreamFuncServicer):
    """Missing associated documentation comment in .proto file."""
    def __init__(self, server):
        self.server: Server = server    
        self.logger = get_logger("Servicer")
    
    
    def processStream(self, request_iterator, context):
        ct = threading.current_thread()
        
        self.logger.info(f"Processing stream on {ct.name}")
        # create a new connection
        try:
            connection = StreamConnection(driverInfo=self.server.driver.driverInfo)
            self.server.driver.add_connection(connection)
            t = threading.Thread(target=connection.process_frames, args=(request_iterator,), daemon=True)
            t.start()
            yield from connection.generate_output()
        except Exception as e:
            self.logger.error(f"Error processing stream: {e}")
            if t.is_alive():
                self.logger.warning("Thread still running, attempting to join with timeout.")
                t.join(timeout=5)  # 设置超时，避免永久阻塞
        finally:
            if t and t.is_alive():
                t.join(timeout=5)
            if connection:
                connection.close()
                self.server.driver.close_connection(connection)
    
class Server:
    def __init__(
        self,
        driver: Driver,
        driverInfo: DriverInfo,
        max_workers,
    ):
        self.driver: Driver = driver
        self.driverInfo = driverInfo
        self.max_workers = max_workers
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=options)
        self.logger = get_logger("Server")  
        
        # create a servicer
        self.servicer = Servicer(self)
        add_grpcStreamFuncServicer_to_server(self.servicer, self.grpc_server)
        
        # get the address of the server
        addr = self.driverInfo.params.addr
        try:
            self.grpc_server.add_insecure_port(address=addr)
        except Exception as e:
            self.logger.error(f"Error adding insecure port: {e}")
        finally:
            self.logger.info(f"Add insecure port: {addr}")
    
    
    def start(self):
        ct = threading.current_thread()
        self.grpc_server.start()
        self.logger.info(f"Server starting on {self.driverInfo.params.addr}, thread: {ct.name}")
        
        
        self.grpc_server.wait_for_termination()
    
    
    def stop(self):
        self.grpc_server.stop(grace=0.5)
        self.grpc_server = None
        


class GrpcDriver(Driver):
    def __init__(self):
        super().__init__()
        self.max_workers = 100
        self.server = None
        self.logger = get_logger("GrpcDriver")


    def connect(self, driverInfo: DriverInfo):
        params = driverInfo.params
        connection = None
        
        try:
            channel = grpc.insecure_channel(params.addr, options=options)

            stub = grpcStreamFuncStub(channel)
            connection = StreamConnection(driverInfo=driverInfo)
            self.add_connection(connection)
            received = stub.processStream(connection.generate_output())
            connection.process_frames(received)
        except grpc.RpcError as e:
            self.logger.error(f"gRPC connection error: {e}")
        except Exception as e:
            self.logger.error(f"Error connecting to {params.addr}: {e}")
        finally:
            if connection:
                connection.close()
                self.close_connection(connection)
    
    
    def listen(self, driverInfo: DriverInfo):
        self.driverInfo = driverInfo
        self.server = Server(
            self,
            self.driverInfo,
            self.max_workers
        )
        self.server.start()
        
        
    def stop(self):
        if self.server is not None:
            self.server.stop()
        self.close_all_connections()
        
    