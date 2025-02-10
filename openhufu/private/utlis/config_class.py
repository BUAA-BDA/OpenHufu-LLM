from dataclasses import dataclass

@dataclass
class BaseConfig:
    name: str = ""
    host: str = "localhost"
    port: int = 0
    addr: str = ""
    schema: str = "grpc"
    
@dataclass
class ServerConfig(BaseConfig):
    name: str = "server"
    host: str = "localhost"
    port: int = 8002
    

@dataclass
class ClientConfig(BaseConfig):
    server_name: str = "server"

@dataclass
class StandaloneConfig(object):
    pass
