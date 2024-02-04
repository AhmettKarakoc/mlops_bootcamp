import socket
import yaml
import os


def find_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn’t even have to be reachable
        s.connect(("10.255.255.255", 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = "127.0.0.1"
    finally:
        s.close()
    return ip_address


def get_all_configs():
    with open(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml"), "r"
    ) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict


def get_env_configs(env):
    all_configs_dict = get_all_configs()
    return all_configs_dict[env]
