import grpc
import queue
from concurrent import futures
from typing import Dict
import threading

from openhufu.private.net.connection import Connection, DriverInfo, ConParams
from openhufu.private.drivers.proto.grpc_stream_pb2_grpc import grpcStreamFuncServicer, add_grpcStreamFuncServicer_to_server, grpcStreamFuncStub
from openhufu.private.drivers.proto.grpc_stream_pb2 import Frame
from openhufu.private.drivers.driver import Driver


class StreamConnection(Connection):
    """StreamConnection class for handling network connections.

    Args:
        Connection (_type_): _description_
    """
    seq_num = 0
    
    def __init__(self, driverInfo: DriverInfo):
        super().__init__(driverInfo=driverInfo)
        self.frame_queue = queue.Queue()
        self.queue_lock = threading.Lock()  
    
    def send_frame(self, frame):
        try:
            StreamConnection.seq_num += 1
            with self.queue_lock:
                self.frame_queue.put(Frame(seq=StreamConnection.seq_num, data=bytes(frame)))
        except Exception as e:
            print("Error: ", e)
            
    def process_frames(self, request_iterator):
        ct = threading.current_thread()
        
        try:
            for frame in request_iterator:
                assert isinstance(frame, Frame)
                
                if self.frame_receiver:
                    self.frame_receiver.process_frame(frame.data)
                else:
                    print("No frame receiver registered")
        except Exception as e:
            print("Error: ", e)
        
    def close(self):
        pass
    
    def generate_output(self):
        ct = threading.current_thread()
        for i in self.frame_queue:
            assert isinstance(i, Frame)
            yield i

    
class Servicer(grpcStreamFuncServicer):
    """Missing associated documentation comment in .proto file."""
    def __init__(self, server):
        self.server: Server = server    
        
    
    def processStream(self, request_iterator, context):
        ct = threading.current_thread()
        
        # create a new connection
        try:
            connection = StreamConnection(driverInfo=self.server.driver.driverInfo)
            self.server.driver.add_connection(connection)
            t = threading.Thread(target=connection.process_frames, args=(request_iterator,))
            t.start()
            yield from connection.generate_output()
        except Exception as e:
            print("Error: ", e)
        finally:
            if t:
                t.join()
            if connection:
                connection.close()
                self.server.driver.close_connection(connection)
    
class Server:
    def __init__(
        self,
        driver,
        driverInfo: DriverInfo,
        max_workers,
    ):
        self.driver = driver
        self.driverInfo = driverInfo
        self.max_workers = max_workers
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        
        # create a servicer
        self.servicer = Servicer(self)
        add_grpcStreamFuncServicer_to_server(self.servicer, self.grpc_server)
        
        # get the address of the server
        addr = self.driverInfo.params.addr
        try:
            self.grpc_server.add_insecure_port(address=addr)
        except Exception as e:
            print("Error: ", e)
    
    def start(self):
        self.grpc_server.start()
        self.grpc_server.wait_for_termination()
    
    def stop(self):
        self.grpc_server.stop(grace=0.5)
        self.grpc_server = None
        


class GrpcDriver(Driver):
    def __init__(self):
        super().__init__()
        self.max_workers = 100

    def connect(self, driverInfo: DriverInfo):
        params = driverInfo.params
        connection = None
        
        try:
            channel = grpc.insecure_channel(params.addr)

            stub = grpcStreamFuncStub(channel)
            connection = StreamConnection(driverInfo=driverInfo)
            self.add_connection(connection)
            received = stub.processStream(connection.generate_output())
            connection.process_frames(received)
        except Exception as e:
            print("Error: ", e)
        finally:
            if connection:
                connection.close()
                self.close_connection(connection)
    
    def listen(self, connectionInfo: DriverInfo):
        self.connectionInfo = connectionInfo
        self.server = Server(
            self,
            self.connectionInfo,
            self.max_workers
        )
        self.server.start()