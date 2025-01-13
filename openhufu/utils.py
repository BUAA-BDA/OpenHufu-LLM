import os
import requests
import time
import yaml
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

def get_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
        print(config)
        return config

def get_runner(mode):
       if mode.lower() == 'standalone':
           from Runner import StandaloneRunner
           return StandaloneRunner