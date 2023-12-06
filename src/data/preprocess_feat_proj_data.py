import os
from src.utils import load_ruamel
import pandas as pd
from enum import Enum
import json


class Columns(Enum):
    ID = 'ID'
    NAME = 'NAME'


def preprocess_split(mapping_ids, mapping_file, central_node_type):
    mapping_file[Columns.NAME] = mapping_file[Columns.NAME].map(
        lambda x: f'{x[:-4]}.safetensors'
    )
    ids = map(int, mapping_ids[central_node_type].keys())
    return mapping_file.loc[list(ids), [Columns.NAME]]


def main():
    params = load_ruamel('../../configs/preprocess_feat_proj_data.yaml')
    out_dir = params['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    for split in os.listdir(params['split_path']):
        with open(f'{params["split_path"]}/{split}/{split}_map.json') as f:
            map_ids = json.load(f)
        mapping = pd.read_csv(f'{params["mapping_file"]}/{params["central_node_type"]}_entidx2name.csv',
                              header=None,
                              names=[Columns.ID, Columns.NAME])
        mapped = preprocess_split(
            mapping_ids=map_ids,
            mapping_file=mapping,
            central_node_type=params['central_node_type']
        )
        mapped.to_csv(f'{out_dir}/{split}.csv', index=False)


if __name__ == '__main__':
    main()
