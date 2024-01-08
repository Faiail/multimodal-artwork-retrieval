from torchvision.transforms import v2
import nlpaug.augmenter.word as naw
import argparse
import hyperopt
import json


def get_distribution(key, value):
    if value['type'] == 'choice':
        return hyperopt.hp.choice(key, value['range'])
    if value['type'] == 'uniform':
        return hyperopt.hp.uniform(key, *value['range'])
    raise ValueError(f'{value["type"]} not supported')


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--params_path')
    return argparser.parse_args()


def convert_image_preprocess(params):
    AUGMENTATIONS = v2.__dict__
    out = []
    for entry, val in params.item():
        out.append(
            AUGMENTATIONS[entry](**val)
        )
    return out


def convert_text_preprocess(params):
    AUGMENTATIONS = naw.__dict__
    out = []
    for entry, val in params.item():
        out.append(
            AUGMENTATIONS[entry](**val)
        )
    return out


def save_params(path, params):
    with open(path, 'w+') as f:
        json.dump(params, f)