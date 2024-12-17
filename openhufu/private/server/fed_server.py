from openhufu.private.net.cell import Cell


class FederatedServer:
    def __init__(self, cell: Cell):
        self.cell = cell

    def start(self):
        self.cell.start()