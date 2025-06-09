import openhufu.private.utlis.defs as defs
from openhufu.private.utlis.defs import CellChannel, CellChannelTopic
import logging
from openhufu.private.utlis.util import get_logger

logger = get_logger("worker")
class Worker(object):
    def __init__(self, id, com_manager):
        self.id = id
        self.msg_handlers = dict()
        self.com_manager = com_manager
        self._register_all_callback()

    def get_handler(self, msg_type: str):
        return self.msg_handlers[msg_type]
    
    def send_message(self, target , channel: CellChannel , topic: CellChannelTopic, **kwargs):
        logger.info(f"worker:{self.id} send topic:{topic} to worker:{target}")
        self.com_manager.send_message(target, channel, topic, **kwargs)

    def __test(self, msg):
        print(f'I am Client, I rec message from {msg.sender}')

    
    def _register_all_callback(self):
         print("execute father") 
    
    def _register_handler(self, topic: defs.CellChannelTopic, callback):
        self.msg_handlers[topic] = callback
        self.com_manager.register_request_cb(None, topic, callback)

    def deploy(self):
        pass