import openhufu.private.utlis.defs as defs
class Worker(object):
    def __init__(self, id, com_manager):
        self.id = id
        self.msg_handlers = dict()
        self.com_manager = com_manager
        self._register_all_callback()

    def get_handler(self, msg_type: str):
        return self.msg_handlers[msg_type]
    

    def __test(self, msg):
        print(f'I am Client, I rec message from {msg.sender}')

    
    def _register_all_callback(self):
         print("execute father") 
    
    def _register_handler(self, topic: defs.CellChannelTopic, callback):
        self.msg_handlers[topic] = callback
        self.com_manager.register_request_cb(None, topic, callback)

    def deploy(self):
        pass