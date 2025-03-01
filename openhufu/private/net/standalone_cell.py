from openhufu.private.utlis.defs import CellChannel, CellChannelTopic
from openhufu.private.utlis.util import get_logger
from openhufu.private.utlis.defs import CellChannel, CellChannelTopic
from openhufu.utils import IDGenerator

class StandaloneCell(object):
    _instance = None
    # client类里只保存 id列表  映射由cell完成
    # 这个映射由创建的时候填入

    @classmethod
    def get_singleton(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self, config):
        self.logger = get_logger(__name__)
        self.config = config
        self.registered_cbs = {}
        self.id2worker = dict()

    def register_request_cb(self, channel: CellChannel, topic: CellChannelTopic, cb):
        # topic起不同的就可以 用不到channel
        if not callable(cb):
            raise Exception("Callback is not callable")
        # 注册名字 这样同类不同对象的callback就只用注册一次了
        if topic not in self.registered_cbs.keys():
            self.registered_cbs[topic] = cb.__name__

    def start(self):
        pass

    def stop(self):
        pass

    def add_participant(self, part, id=IDGenerator.next_id()):
        self.id2worker[id] = part

    # def send_message(id, CellChannel.SERVER_MAIN, CellChannelTopic.Finish, None)
    def send_message(self, id, channel: CellChannel, topic: CellChannelTopic, data):
        callback = getattr(self.id2worker[id], self.registered_cbs[topic], None)
        callback(data)




