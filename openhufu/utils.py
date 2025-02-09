import os
import requests
import time
from yacs.config import CfgNode as CN

def download_url(url, path):
    file_name = os.path.basename(url)
    if '.' in file_name:
        file_base_name = os.path.splitext(file_name)[0]
    else:
        timestamp = int(time.time())
        file_name = f"download_{timestamp}.tmp"
        file_base_name = file_name

    dir_path = os.path.join(path, file_base_name)
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, file_name)

    response = requests.get(url)
    with open(file_path, "wb") as file:
        file.write(response.content)

    return file_path


def get_file_path_without_name(file_path):
    return os.path.dirname(file_path)


class IDGenerator:
    _count = -1

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count


def load_config(path: str):
    # yaml 2 tree
    cfg = CN()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(path)
    return cfg


import importlib


def load_class(module_path, class_name):
    """
    根据模块路径和类名动态加载类。

    参数:
        module_path (str): 包含目标类的模块路径。
        class_name (str): 要加载的类的名称。

    返回:
        class: 动态加载的类。
    """
    try:
        # 加载模块
        module = importlib.import_module(module_path)

        # 获取类
        cls = getattr(module, class_name)

        return cls
    except (ImportError, AttributeError) as e:
        raise ImportError(f"无法加载类 '{class_name}' 从模块 '{module_path}': {e}")


