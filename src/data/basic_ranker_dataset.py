from torch.utils.data import Dataset
from typing import Union
import pandas as pd
from torchvision.transforms import Compose
from PIL import Image


class BasicRankerDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            dataset: Union[pd.DataFrame, str],
            names: Union[pd.DataFrame, str],
            preprocess: Compose = None,
    ):
        self.image_dir = image_dir
        self.dataset = dataset if isinstance(dataset, pd.DataFrame) else pd.read_csv(dataset)
        self.names = names if isinstance(names, pd.DataFrame) else pd.read_csv(names)
        self.preprocess = preprocess

    def _get_image(self, image_id):
        image_fname = self.names.iloc[image_id, 0]
        image_fname.replace('.safetensors', '.jpg')
        image = Image.open(f'{self.image_dir}/{image_fname}').convert('RGB')
        return self.preprocess(image) if self.preprocess else image

    def _get_text(self, raw_id):
        return {
            'comment': self.names.iloc[raw_id]['caption'],
            'title': self.names.iloc[raw_id]['title'],
        }

    def __getitem__(self, item):
        data_dict = {}
        # get image from the first artwork
        data_dict['image'] = self._get_image(self.dataset.iloc[item, 0])
        # get the text from the second one
        data_dict.update(self._get_text(self.dataset.iloc[item, 1]))
        data_dict['gt'] = self.dataset.iloc[item, 2]
        return data_dict
