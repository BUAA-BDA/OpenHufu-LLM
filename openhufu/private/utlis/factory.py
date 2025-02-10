from ..net.standalone_cell import StandaloneCell
from ..net.cell import Cell
from ..net.client_cell import ClientCell

global buffer
# def get_cell(config):
#     if isinstance(config, StandaloneConfig):
#         return StandaloneCell.get_singleton(config)
#     elif isinstance(config, ServerConfig):
#         return Cell(config)
#     else:
#         return ClientCell(config)

def get_cell(config):
    if config.mode.lower() == 'standalone' or config.mode.lower() == 'simulator':
        return StandaloneCell.get_singleton(config)
    elif config.role == 'server':
        return Cell(config)
    else:
        return ClientCell(config)

