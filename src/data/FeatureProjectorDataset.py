from torch.utils.data import Dataset
from src.utils import load_tensor
import pandas as pd
from enum import Enum
from PIL import Image


class DataModality(Enum):
    IMAGE = 'image'
    TEXT = 'text'
    GRAPH = 'graph'


class Mode(Enum):
    EMBEDDING = 'embedding'
    RAW = 'raw'


class FeatureProjectorDataset(Dataset):
    def __init__(
            self,
            mode,
            source_modality,
            dest_modality,
            data,
            data_source_dir=None,
            data_dest_dir=None,
            preprocess_source=None,
            preprocess_dest=None,
    ):
        super().__init__()
        self.mode=mode
        self.source_modality = source_modality
        self.dest_modality = dest_modality
        self.data = pd.read_csv(data)
        self.data_source_dir = data_source_dir
        self.data_dest_dir = data_dest_dir
        self.preprocess_source = preprocess_source
        self.preprocess_dest = preprocess_dest

    def __get_source(self, item):
        if self.mode == Mode.EMBEDDING.value:
            return load_tensor(
                file=f'{self.data_source_dir}/{self.data.iloc[item, 0]}',
                key=self.source_modality,
            )
        assert self.mode == Mode.RAW.value
        fname = self.data.iloc[item, 0]
        if self.source_modality == DataModality.IMAGE.value:
            image_fname = fname.split('.')[0] + '.jpg'
            x = Image.open(f'{self.data_source_dir}/{image_fname}')
            if self.preprocess_source:
                x = self.preprocess_source(x)
        elif self.source_modality == DataModality.TEXT.value:
            x = self.data.iloc[item, 1]
            preprocess, tokenizer = (
                self.preprocess_source.get('preprocess', None),
                self.preprocess_source.get('tokenizer', None),
            )
            if preprocess:
                for step in preprocess:
                    x = step(x)
            x = tokenizer(x).squeeze(dim=0)
        elif self.source_modality == DataModality.GRAPH.value:
            x = load_tensor(
                file=f'{self.data_source_dir}/{self.data.iloc[item, 0]}',
                key=self.source_modality,
            )
        return x

    def __get_dest(self, item):
        return load_tensor(
            file=f'{self.data_dest_dir}/{self.data.iloc[item, 0]}',
            key=self.source_modality,
        )

    def __getitem__(self, item):
        x = self.__get_source(item)
        y = self.__get_dest(item)
        return x, y

    def __len__(self):
        return len(self.data)
