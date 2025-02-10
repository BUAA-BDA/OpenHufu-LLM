import argparse
from openhufu.utils import load_config
from openhufu.private.simulator.simulator_delpoyer import SimulatorDeployer
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

if __name__=="__main__":
    args = parse_args()
    cfg = load_config(args.config)
    deployer = SimulatorDeployer(cfg)
    deployer.deploy()