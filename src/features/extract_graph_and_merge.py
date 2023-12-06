import os
import torch
from src.utils import load_ruamel, load_tensor
from src.models.gae.HeteroGAE import HeteroGAE
import json
import pandas as pd
from src.data.preprocess_feat_proj_data import Columns
from safetensors.torch import save_file
import shutil
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def load_data(data_dir, device):
    datasets = {}
    for split in os.listdir(data_dir):
        graph = torch.load(f'{data_dir}/{split}/{split}_data.pt').to(device)
        datasets[split] = {
            'data': ToUndirected()(graph).to(device),
            'map_ids': load_json(f'{data_dir}/{split}/{split}_map.json')
        }
    return datasets


def extract_and_merge(model, datasets, mapping_file, source_embedding_dir, dest_embedding_dir, delete, verbose):
    os.makedirs(dest_embedding_dir, exist_ok=True)
    for split in datasets:
        if verbose:
            print(f'Merging {split} split')
        # encode information
        data = datasets[split]['data']
        map_ids = datasets[split]['map_ids']['artwork']
        z = model.encode(data.x_dict, data.edge_index_dict)['artwork']
        names = mapping_file.loc[list(map(int, map_ids.keys()))]
        names[Columns.NAME] = names[Columns.NAME].map(lambda x: f'{x[:-4]}.safetensors')
        names_iter = tqdm(names.iterrows()) if verbose else names.iterrows()
        for ix, row in names_iter:
            image = load_tensor(f'{source_embedding_dir}/{row[Columns.NAME]}', key='image')
            text = load_tensor(f'{source_embedding_dir}/{row[Columns.NAME]}', key='text')
            tensors = {
                'image': image,
                'text': text,
                'graph': z[map_ids[str(row[Columns.ID])]].cpu(),
            }
            save_file(tensors, f'{dest_embedding_dir}/{row[Columns.NAME]}')
    if delete:
        if verbose:
            print(f'Deleting {source_embedding_dir} directory...')
        shutil.rmtree(source_embedding_dir)


def main():
    params = load_ruamel('../../configs/extract_graph_and_merge.yaml')
    device = params['device']
    mapping_file = pd.read_csv(params['mapping_file'], header=None, names=[Columns.ID, Columns.NAME])
    datasets = load_data(params['data_dir'], device)
    model = HeteroGAE(
        metadata=datasets['train']['data'].metadata(),
        **params['model'],
    )
    model = model.to(device)
    state_dict = torch.load(params['checkpoint'])
    # lazy initialization
    model(datasets['test']['data'].x_dict, datasets['test']['data'].edge_index_dict)
    model.load_state_dict(state_dict)

    extract_and_merge(
        model=model,
        datasets=datasets,
        mapping_file=mapping_file,
        **params['general']
    )


if __name__ == '__main__':
    main()