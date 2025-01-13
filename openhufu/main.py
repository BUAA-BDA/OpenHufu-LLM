import argparse
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg')
    args = parser.parse_args()
    config = utils.get_config(args.cfg)
    print(config)
    runner = utils.get_runner(config['mode'])(None, config)
    runner.run()