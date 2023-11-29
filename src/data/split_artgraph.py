import os
from artgraph.graph_splitter import ArtGraphInductiveSplitter
from artgraph.load_artgraph import ArtGraph
from src.utils import load_parameters
import torch
import json


def main():
    params = load_parameters('../../configs/split_artgraph.yaml')
    artgraph = ArtGraph(**params['dataset'])[0]
    print(artgraph)
    splitter = ArtGraphInductiveSplitter(dataset=artgraph, **params['splitter'])
    data = splitter.transform()

    out_dir = params['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    print('Saving...')
    for (data_set, map_id), split in zip(data, ['train', 'val', 'test']):
        os.makedirs(f'{out_dir}/{split}', exist_ok=True)
        torch.save(data_set, f'{out_dir}/{split}/{split}_data.pt')
        with open(f'{out_dir}/{split}/{split}_map.json', 'w+') as f:
            json.dump(map_id, f)
    print('Done!')


if __name__ == '__main__':
    main()