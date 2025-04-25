import os
import requests
import time
from yacs.config import CfgNode as CN
from typing import Union
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
    _count = -2

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


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        # if not template_name:
        #     # Enforce the default here, so the constructor can be called with '' and will not break.
        #     template_name = "alpaca"
        # # file_name = osp.join("templates", f"{template_name}.json")
        # file_name = f"{template_name}.json"
        # if not osp.exists(file_name):
        #     raise ValueError(f"Can't read {file_name}")
        # with open(file_name) as fp:
        #     self.template = json.load(fp)
        # if self._verbose:
        #     print(
        #         f"Using prompt template {template_name}: {self.template['description']}"
        #     )
        self.template = {
            "prompt_input": "以下是一个描述任务的指令和一个提供进一步上下文的输入。 写一个适当的回答来完成请求。\n\n### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回答:\n",
            "prompt_no_input": "以下是一个描述任务的指令。 请写一个适当的回答来完成请求。\n\n### 指令:\n{instruction}\n\n### 回答:\n",
            "response_split": "### 回答:"    
        }

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

