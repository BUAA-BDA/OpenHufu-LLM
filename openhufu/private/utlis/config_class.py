from dataclasses import dataclass

@dataclass
class BaseConfig:
    name: str
    host: str
    port: int
    addr: str = ""
    schema: str = "grpc"
    

class ServerConfig(BaseConfig):
    name: str = "server"
    host: str = "localhost"
    port: int = 8002
    

class ClientConfig(BaseConfig):
    pass
