import torch.utils.data
from torch.utils.data import Dataset
from typing import Union, Optional
import pandas as pd
from torch.utils.data.dataset import T_co
from torchvision.transforms import Compose
from PIL import Image
from src.utils import load_tensor


class ContextNetDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            data: Union[str, pd.DataFrame],
            emb_dir: str,
            emb_key: str,
            preprocess: Optional[Compose] = None,
    ):
        self.image_dir = image_dir
        self.data = data if isinstance(data, pd.DataFrame) else pd.read_csv(data)
        self.emb_dir = emb_dir
        self.emb_key = emb_key
        self.preprocess = preprocess

    def _load_image(self, fname):
        img = Image.open(f"{self.image_dir}/{fname}").convert("RGB")
        if self.preprocess:
            img = self.preprocess(img)
        return img

    def __getitem__(self, item):
        image_fname, label = self.data.iloc[item]
        return {
            "images": self._load_image(image_fname.replace(".safetensors", ".jpg")),
            "embeddings": load_tensor(file=f"{self.emb_dir}/{image_fname}", key=self.emb_key),
            "gt": label,
        }

    def __len__(self):
        return len(self.data)


class ContextNetRankerDataset(Dataset):
    def __init__(
            self,
            image_dir: str,
            dataset: Union[pd.DataFrame, str],
            names: Union[pd.DataFrame, str],
            attribute_names: Union[pd.DataFrame, str],
            preprocess: Compose = None,

    ) -> None:
        self.image_dir = image_dir
        self.dataset = dataset if isinstance(dataset, pd.DataFrame) else pd.read_csv(dataset, index_col=0)
        self.names = names if isinstance(names, pd.DataFrame) else pd.read_csv(names)
        self.attribute_names = attribute_names if isinstance(attribute_names, pd.DataFrame) else pd.read_csv(attribute_names)
        self.preprocess = preprocess

    def _load_image(self, fname):
        img = Image.open(f"{self.image_dir}/{fname}").convert("RGB")
        if self.preprocess:
            img = self.preprocess(img)
        return img

    def _load_comment_title(self, index):
        return self.names.iloc[index]['caption'], self.names.iloc[index]['title']

    def _load_attributes(self, index):
        return tuple(self.attribute_names.iloc[index, 1:].tolist())

    def __getitem__(self, item) -> dict:
        a, b, score = self.dataset.iloc[item]
        a, b = int(a), int(b)
        image = self._load_image(self.names.iloc[a, 0])
        comment, title = self._load_comment_title(b)
        attributes = self._load_attributes(b)
        score = 1 if score >= .5 else -1
        return {
            "image": image,
            "comment": comment,
            "title": title,
            "attributes": attributes,
            "score": score,
        }

    def __len__(self) -> int:
        return len(self.dataset)


def collate_fn(batched_input):
    image = torch.stack([x["image"] for x in batched_input])
    comment = [x["comment"] for x in batched_input]
    title = [x["title"] for x in batched_input]
    attributes = [x["attributes"] for x in batched_input]
    score = torch.as_tensor([x["score"] for x in batched_input]).float()
    return {
        "image": image,
        "comment": comment,
        "title": title,
        "attributes": attributes,
        "score": score,
    }
