from enum import Enum


class CellChannel:
    SERVER_MAIN = "server_main"
    CLIENT_MAIN = "client_main" 
    
    
class CellChannelTopic:
    Challenge = "challenge"
    Register = "register"
    

class HeaderKey:
    # endpoint name
    SOURCE_ENDPOINT = "source_endpoint"
    DESTINATION_ENDPOINT = "destination_endpoint"
    
    # message type
    CHANNEL_TOPIC = "channel_topic"
    CHANNEL = "channel"
    