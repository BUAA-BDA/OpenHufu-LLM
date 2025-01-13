from worker import Worker

class BaseClient(Worker):
    def __init__(self, id, config, com_manager):
        super().__init__(id, com_manager)
        self.msg_handlers = dict()
        self.register_handler()

    def test(self, msg):
        print(f'I am Client, I rec message from {msg.sender}')

    def register_handler(self):
        self.msg_handlers['test'] = self.test

    def msg_handler(self, msg_type):
        return self.msg_handlers[msg_type]