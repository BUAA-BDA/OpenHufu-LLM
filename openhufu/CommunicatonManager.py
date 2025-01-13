from abc import ABC, abstractmethod
from collections import deque
import heapq


class BaseManager(ABC):

    @abstractmethod
    def send_message(self, msg):
        pass

    @abstractmethod
    def receive_message(self):
        pass


class StandaloneManager(BaseManager):

    def __init__(self):
        self.msg_queue = deque()
        self.server_msg_buffer = []

    def send_message(self, msg):
        if msg.receiver == -1:
            heapq.heappush(self.server_msg_buffer, (msg.timestamp, msg))
        else:
            # to client
            self.msg_queue.append(msg)

    def receive_message(self):
        # 得知道自己是谁
        pass

    def has_unhandled_msg(self) -> bool:
        return len(self.msg_queue) > 0 or len(self.server_msg_buffer) > 0

    def server_has_unhandled_msg(self):
        return len(self.server_msg_buffer) > 0

    def server_receive_msg(self):
        _, msg = heapq.heappop(self.server_msg_buffer)
        return msg

    def clients_have_unhandled_msg(self):
        return len(self.msg_queue) > 0

    def client_receive_msg(self):
        return self.msg_queue.popleft()


