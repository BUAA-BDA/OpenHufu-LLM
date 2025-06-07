# openhufu/private/utlis/TensorEncoder.py
import msgpack
import torch
import numpy as np

# 递归编码器
class TensorEncoder:
    # def __init__(self):
    #     self.default_encoder = msgpack.Packer().default
        
    # def __call__(self, obj):
    #     if isinstance(obj, torch.Tensor):
    #         return {
    #             "__tensor__": True,
    #             "data": obj.cpu().detach().numpy().tolist(),
    #             "dtype": str(obj.dtype),
    #             "shape": list(obj.shape)
    #         }
    #     elif isinstance(obj, dict):
    #         return {k: self(v) for k, v in obj.items()}
    #     elif isinstance(obj, list) or isinstance(obj, tuple):
    #         return [self(item) for item in obj]
    #     return self.default_encoder(obj)

    def __call__(self, obj):
        if isinstance(obj, torch.Tensor):
            return {
                "__tensor__": True,
                "data": obj.cpu().detach().numpy().tolist(),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape)
            }
        elif isinstance(obj, dict):
            return {k: self(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [self(item) for item in obj]
        # 直接返回其他类型，让 msgpack 自己处理
        return obj

# 递归解码器
def tensor_decoder(obj):
    if isinstance(obj, dict):
        if "__tensor__" in obj:
            data = obj["data"]
            dtype_str = obj["dtype"]
            shape = obj["shape"]
            
            # 将dtype字符串转换为torch的dtype
            dtype_map = {
                "torch.float16": torch.float16,
                "torch.float32": torch.float32,
                "torch.float64": torch.float64,
                "torch.int64": torch.int64,
                "torch.int32": torch.int32,
                "torch.int16": torch.int16,
                "torch.int8": torch.int8,
                "torch.uint8": torch.uint8,
                "torch.bool": torch.bool
            }
            dtype = dtype_map.get(dtype_str, torch.float32)
            
            # 从列表重建Tensor
            return torch.tensor(data, dtype=dtype).reshape(shape)
        else:
            # 处理普通字典，递归解码其值
            return {k: tensor_decoder(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # 处理列表，递归解码其元素
        return [tensor_decoder(item) for item in obj]
    return obj

# 使用自定义编码器和解码器
def custom_packb(data):
    return msgpack.packb(data, default=TensorEncoder())

def custom_unpackb(packed_data):
    return msgpack.unpackb(packed_data, object_hook=tensor_decoder, raw=False)