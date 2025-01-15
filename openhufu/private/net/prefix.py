import struct
from dataclasses import dataclass
from enum import Enum

from openhufu.private.utlis.util import get_logger

logger = get_logger(__name__)
PREFIX_STRUCT = struct.Struct(">IHBB")
PREFIX_LEN = PREFIX_STRUCT.size



class FrameType(Enum):
    HELLO = 0
    HI = 1
    DATA = 2



@dataclass
class Prefix:
    """
    Prefix of a frame
    length: length of the frame, including the prefix, in bytes, 4 bytes
    header_len: length of the header, 2 bytes
    stream_id: stream id, 2 bytes
    frame_type: type of the frame, HELLO/DATA, 1 byte
    """
    
    length : int = 0
    stream_id : int = 0
    frame_type : int = 0
    has_next : int = 0
    
    @staticmethod
    def parse(data):
        if len(data) < PREFIX_LEN:
            logger.error(f"Invalid frame, data length {len(data)} < {PREFIX_LEN}")
            raise ValueError(f"Invalid frame, data length {len(data)} < {PREFIX_LEN}")
        length, stream_id, frame_type, has_next = PREFIX_STRUCT.unpack(data[:PREFIX_LEN])
        return Prefix(length, stream_id, frame_type, has_next)
    

    def to_buffer(self, buffer):
        return PREFIX_STRUCT.pack_into(buffer, 0, self.length,  self.stream_id, self.frame_type, self.has_next)
    