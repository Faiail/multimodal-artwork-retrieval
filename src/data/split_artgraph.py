import os
from artgraph.graph_splitter import ArtGraphInductiveSplitter
from artgraph.load_artgraph import ArtGraph
from src.utils import load_parameters
import torch


def main():
    params = load_parameters('../../configs/split_artgraph.yaml')
    artgraph = ArtGraph(**params['dataset'])[0]
    print(artgraph)
    splitter = ArtGraphInductiveSplitter(dataset=artgraph, **params['splitter'])
    data = splitter.transform()

    out_dir = params['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    print('Saving...')
    for data_set, split in zip(data, ['train_data', 'val_data', 'test_data']):
        torch.save(data_set, f'{out_dir}/{split}.pt')
        print(f'{split} saved in {out_dir}/{split}.pt')
    print('Done!')


if __name__ == '__main__':
    main()