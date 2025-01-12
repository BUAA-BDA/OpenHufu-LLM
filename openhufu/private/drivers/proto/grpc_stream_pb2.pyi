from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Frame(_message.Message):
    __slots__ = ("seq", "data")
    SEQ_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    seq: int
    data: bytes
    def __init__(self, seq: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...
