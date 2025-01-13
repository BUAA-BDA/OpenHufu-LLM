class Worker(object):
    def __init__(self, id, com_manager, model=None):
        self.id = id
        self.model = model
        self.handler_dict = dict()
        self.com_manager = com_manager

    def model(self):
        return self.model

    def handler(self, msg_type: str):
        return self.handler_dict[msg_type]
