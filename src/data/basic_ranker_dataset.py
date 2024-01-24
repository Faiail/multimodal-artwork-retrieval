from torch.utils.data import Dataset
from typing import Union, Optional, List
import pandas as pd
from torchvision.transforms import Compose
from PIL import Image
from src.data.predictor_datasets import PredictorTestDataset, CatalogDataset
import torch


class BasicRankerDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            dataset: Union[pd.DataFrame, str],
            names: Union[pd.DataFrame, str],
            preprocess: Compose = None,
    ):
        self.image_dir = image_dir
        self.dataset = dataset if isinstance(dataset, pd.DataFrame) else pd.read_csv(dataset, index_col=0)
        self.names = names if isinstance(names, pd.DataFrame) else pd.read_csv(names)
        self.preprocess = preprocess

    def _get_image(self, image_id):
        image_fname = self.names.iloc[image_id, 0]
        image_fname = image_fname.replace('.safetensors', '.jpg')
        image = Image.open(f'{self.image_dir}/{image_fname}').convert('RGB')
        return self.preprocess(image) if self.preprocess else image

    def _get_text(self, raw_id):
        return {
            'comments': self.names.iloc[raw_id]['caption'],
            'titles': self.names.iloc[raw_id]['title'],
        }

    def __getitem__(self, item):
        data_dict = {}
        # get image from the first artwork
        data_dict['images'] = self._get_image(self.dataset.iloc[item, 0])
        # get the text from the second one
        data_dict.update(self._get_text(self.dataset.iloc[item, 1]))
        data_dict['gt'] = 0 if self.dataset.iloc[item, 2] < .5 else 1
        return data_dict

    def __len__(self):
        return len(self.dataset)


class BasicGarciaCatalogueDataset(CatalogDataset):
    def __init__(
            self,
            data_t,
            names: Union[str, pd.DataFrame],
            data_keys: List[str],
            out_keys: List[str],
            data_dir: Optional[str] = None,
            preprocess: Optional[Compose] = None,
    ):
        super().__init__()
        self.data_t = data_t
        self.names = names if isinstance(names, pd.DataFrame) else pd.read_csv(names)
        self.data_keys = data_keys
        self.out_keys = out_keys
        self.data_dir = data_dir
        self.preprocess = preprocess

    def _load_image(self, raw):
        image = Image.open(f'{self.data_dir}/{raw[self.data_keys[0]]}').convert('RGB')
        return image if not self.preprocess else self.preprocess(image)

    def __getitem__(self, item):
        raw = self.names.iloc[item]
        raw = raw[self.data_keys]
        if self.data_t == 'image':
            assert len(self.out_keys) == 1
            return {self.out_keys[0]: self._load_image(raw)}
        return {
            y: raw[x] for x, y in zip(self.data_keys, self.out_keys)
        }

    def __len__(self):
        return len(self.names)


class BasicGarciaPredictorDataset(PredictorTestDataset):
    def __init__(
            self,
            data_t,
            data_keys,
            out_keys,
            names: Union[pd.DataFrame, str],
            data_dir: str = None,
            preprocess: Optional[Compose] = None,
    ):
        super().__init__()
        self.data_t = data_t
        self.data_keys = data_keys
        self.out_keys = out_keys
        self.names = names if isinstance(names, pd.DataFrame) else pd.read_csv(names)
        self.data_dir = data_dir
        self.preprocess = preprocess

    def __getitem__(self, item):
        raw = self.names.iloc[item]
        raw = raw[self.data_keys]
        label = torch.zeros(size=(len(self.names), ))
        label[item] = 1
        if self.data_t == 'image':
            assert len(self.out_keys) == 1
            return {self.out_keys[0]: self._load_image(raw)}, label
        return {
            y: raw[x] for x, y in zip(self.data_keys, self.out_keys)
        }, label

    def __len__(self):
        return len(self.names)