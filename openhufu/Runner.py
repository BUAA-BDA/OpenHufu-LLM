from abc import ABC, abstractmethod
from collections import deque
import heapq
import numpy as np
from Message import Message
class BaseRunner(ABC):

    def __init__(self, data, config):
        self.data = data
        self.mode = config['mode']
        self.config = config


    @abstractmethod
    def run(self):
        print(f'run')
        # raise Exception("Not Implement")

class StandaloneRunner(BaseRunner):
    def __init__(self, data, config):
        super().__init__(data, config)
        # self.shared_comm_queue = deque()
        from CommunicatonManager import StandaloneManager
        self.com_manager = StandaloneManager()
        import server
        print(config['server']['name'])
        server_class = getattr(server, 'BaseServer', None)
        print(server_class)
        import client
        client_class = getattr(client, config['client']['name'], None)
        # self.shared_comm_queue = deque()
        # 模型不同？
        self.server = server_class(config, self.com_manager)
        self.client_list = []
        for i in range(config['client']['client_num']):
            _ = client_class(i, config, self.com_manager)
            self.client_list.append(_)


    def run(self):
        # target在哪里确认 流程怎么描述
        # client向server发test server回应
        self.client_list[0].com_manager.send_message(Message('test', 0, -1, 0, 0, 'I am Client'))
        # def __init__(self, msg_type, sender=-1, receiver=-1, seq=-1, timestamp=0, content=None):
        # while self.com_manager.has_unhandled_msg():
        #     for msg in self.com_manager.msg_queue:
        #         self.client_list[msg.receiver].msg_handler[msg.msg_type](msg)
        #     # self.com_manager.clients_handle_msg()
        #     # self.com_manager.server_handle_msg()
        #     # 这么写不行 反射不过去
        #     for _, msg in self.com_manager.server_msg_buffer:
        #
        #         self.server.msg_handler(msg.msg_type)(msg)
        while self.com_manager.clients_have_unhandled_msg():
            msg = self.com_manager.client_recveive_msg()
            self.client_list[msg.receiver.id].msg_handler(msg.msg_type)(msg)
        while self.com_manager.server_has_unhandled_msg():
            msg = self.com_manager.server_receive_msg()
            self.server.msg_handler(msg.msg_type)(msg)





