import yaml


def load_parameters(file_path):
    with open(file_path) as f:
        data = yaml.safe_load(f)
    return data