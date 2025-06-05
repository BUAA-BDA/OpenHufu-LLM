import yaml
import argparse
from pathlib import Path

from openhufu.private.server.server_deployer import ServerDeployer
# from openhufu.private.utlis.util import load_config
from openhufu.utils import load_config

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
    args = parse_args()
    config = load_config(args.config)
    deployer = ServerDeployer(config=config)
    server = deployer.deploy()
