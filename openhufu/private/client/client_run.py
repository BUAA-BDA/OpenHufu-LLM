import yaml
import argparse
from pathlib import Path

from openhufu.private.client.client_deployer import ClientDeployer
from openhufu.private.utlis.config_class import BaseConfig
from openhufu.private.utlis.util import get_logger
from openhufu.utils import load_config
import sys
import traceback
import threading

logger = get_logger(__name__) 

def parse_args():
    parser = argparse.ArgumentParser(description="Federated Server")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        type=str,
        help="path to the config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    def show_thread_exception_hook(args):
        print(f'Thread exception (ignored): {args.exc_type.__name__}: {args.exc_value}',
              file=sys.stderr)
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
    
    # 设置全局线程异常处理器
    threading.excepthook = show_thread_exception_hook
    args = parse_args()
    config = load_config(args.config)
    print(config)
    deployer = ClientDeployer(config=config)
    
    client = deployer.create_client()
    
    client.set_up() # 会安装cell

    logger.info("Client registering")
    client.register()
    
    client.stop()
    # client.perform_local_train()