import yaml
from safetensors import safe_open


def load_parameters(file_path):
    with open(file_path) as f:
        data = yaml.safe_load(f)
    return data


def load_tensor(file, key, device='cpu', framework='pt'):
    with safe_open(file, device=device, framework=framework) as f:
        tensor = f.get_tensor(key)
    return tensor