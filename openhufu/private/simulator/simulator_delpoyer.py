from openhufu.private.utlis.factory import get_cell
from openhufu.utils import load_class, IDGenerator


class SimulatorDeployer:

    def __init__(self, config):
        # 留一个 用来创建server和client 作用对比之前的runner
        # 根据配置从软件包里反射
        # 这里要根据学长的创建方法更改
        self.config = config
        self.cell = None

        self.server = None
        self.id2client = dict()




    def deploy(self):
        self.cell = get_cell(self.config)
        import openhufu.server
        server_class = getattr(openhufu.server, 'BaseServer', None)
        # server_class = load_class('openhufu.server', 'Server')
        import openhufu.client
        client_class = getattr(openhufu.client, 'BaseClient', None)
        server = server_class(self.config, self.cell)
        self.cell.add_participant(server)
        client_id_list = list()
        for i in range(0, self.config.client.num):
            # TODO: yaml的格式要设计 至少要保证与server client配置有关的内容在cluster和单机模拟下一致
            id = IDGenerator.next_id()
            part = client_class(id, self.config, self.cell)
            part.deploy()
            client_id_list.append(id)
            self.cell.add_participant(id=id, part=part)
            
        # server.set_client_id_list(client_id_list)
        server.deploy()

