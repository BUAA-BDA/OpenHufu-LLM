import yaml
import argparse
from pathlib import Path

from openhufu.private.server.server_deployer import ServerDeployer
# from openhufu.private.utlis.util import load_config
from openhufu.utils import load_config
import sys
import traceback
import threading

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
    deployer = ServerDeployer(config=config)
    server = deployer.deploy()
