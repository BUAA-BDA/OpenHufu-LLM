from openhufu.private.utlis.factory import get_cell
from openhufu.private.utlis.defs import CellChannel, CellChannelTopic
from openhufu.private.utlis.defs import CellChannelTopic, CellChannel

class serverDemo(object):

    def __init__(self, cfg):
        self.config = cfg
        self.client_id_list = None

    def set_client_id_list(self, client_id_list):
        self.client_id_list = client_id_list

    def deploy(self):
        # TODO: global rounds怎么办？
        # self.cell = Cell(self.config)
        self.cell = get_cell(self.config)
        self.cell.start()
        self._register_server_cbs()
        self._start_train() # 模拟训练的函数
        self.cell.stop()

    def _start_train(self):
        # server向client发信息的时候带编号, 编号默认server
        # 把client编号注册给server才行
        for i in self.client_id_list:
            print("client id:", i)

        for i in self.client_id_list:
            self.cell.send_message(i, CellChannel.SERVER_MAIN, CellChannelTopic.Finish,None)

    def _challenge_handler(self, data):
        print(f"server rec challenge")

    def _register_server_cbs(self):
        self.cell.register_request_cb(CellChannel.SERVER_MAIN, CellChannelTopic.Challenge, self._challenge_handler)