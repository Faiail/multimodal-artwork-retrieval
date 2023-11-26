from enum import Enum
import numpy as np
import pandas as pd
import shutil
import torch
import torch_geometric as pyg
import os

from torch.onnx._internal.diagnostics.infra.sarif import Artifact
from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)
import src.utils as utils


class LabelEncoder(Enum):
    SCALAR = 0
    ONE_HOT = 1
    W2V = 2


class VisFeatEncoder(Enum):
    CLIP = 0
    OPEN_CLIP = 1
    TIMM_VIT = 2
    RESNET = 3


class FileMapping(Enum):
    MAPPING = 'mapping'
    RAW = 'raw'
    MAPPING_SUFFIX = '_entidx2name'
    RELATIONS = 'relations'


class ArtGraph(InMemoryDataset):
    def __init__(
            self, root,
            label_feats_root,
            vis_feats_root,
            labels=LabelEncoder.W2V,
            vis_feats=VisFeatEncoder.OPEN_CLIP,
    ):
        self.labels = labels
        self.vis_feats = vis_feats
        self.root = root
        self.label_feats_root = label_feats_root
        self.vis_feats_root = vis_feats_root

        assert self.labels in list(LabelEncoder)
        assert self.features in list(VisFeatEncoder)
        super().__init__(root, transform=None, pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])
        for f in os.listdir(fr'{root}/processed'):
            os.remove(fr'{root}/processed/{f}')
        os.rmdir(fr'{root}/processed')
        
        
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        file_names = [
            'node-feat', 'node-label', 'relations', 'split',
            'num-node-dict.csv'
        ]

        return file_names

    @property
    def processed_file_names(self):
        return 'none.pt'
    
    def download(self):
        return

    def __get_artwork_features(self):
        features = torch.empty(size=(116475, 512))
        mapping = pd.read_csv(f'{self.root}/{FileMapping.MAPPING}/artwork{FileMapping.MAPPING_SUFFIX}.csv',
                              header=None)
        for ix, name in mapping.values:
            name = name[:-4] + '.safetensors'
            features[ix] = utils.load_tensor(f'{self.vis_feats_root}/{name}', key='image')
        return features

    def __get_node_types(self):
        return list(
            filter(
                lambda x: x != 'artwork', map(
                    lambda x: x[:-len(FileMapping.MAPPING_SUFFIX)], os.listdir(f'{self.root}/{FileMapping.MAPPING}')
                )
            )
        )

    def process(self):
        data = pyg.data.HeteroData()
        data['artwork'].x = self.__get_artwork_features()
            
        path = os.path.join(self.raw_dir, 'num-node-dict.csv')
        num_nodes_df = pd.read_csv(path)
        num_nodes_df.rename(columns={"training": "training_node"}, inplace=True)
        nodes_type = self.__get_node_types()  # add map training node

        if self.preprocess == LabelEncoder.SCALAR:
            for feature, node_type in enumerate(nodes_type):
                ones = [feature + 1] * num_nodes_df[node_type].tolist()[0]
                data_tensor = torch.tensor(ones)
                data_tensor = torch.reshape(data_tensor, (num_nodes_df[node_type].tolist()[0], 1))
                data[node_type].x = data_tensor
        elif self.preprocess == LabelEncoder.ONE_HOT:
            for node_type in nodes_type:
                data[node_type].x = torch.eye(num_nodes_df[node_type].tolist()[0])

        for edge_type in os.listdir(fr'{self.raw_dir}\relations'):
            sub, verb, obj = edge_type.split("___")
            path = fr'{self.raw_dir}\relations\\{edge_type}\edge.csv'
            edge_index = pd.read_csv(path, header=None, dtype=np.int64).values
            edge_index = torch.from_numpy(edge_index).t().contiguous()
            if obj == 'training':
                obj = 'training_node'
            data[(sub, verb, obj)].edge_index = edge_index

        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        torch.save(self.collate([data]), self.processed_paths[0])

    @property
    def num_features(self):
        return self.data['artist'].x.shape[1]


# if __name__ == '__main__':