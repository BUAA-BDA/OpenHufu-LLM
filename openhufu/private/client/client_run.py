import yaml
import argparse
from pathlib import Path

from openhufu.private.client.client_deployer import ClientDeployer
from openhufu.private.utlis.config_class import BaseConfig
from openhufu.private.utlis.util import get_logger
from openhufu.utils import load_config


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
    args = parse_args()
    config : BaseConfig = load_config(args.config)
    print(config)
    deployer = ClientDeployer(config=config)
    
    client = deployer.create_client()
    
    client.set_up()

    logger.info("Client registering")
    client.register()
    
    client.stop()