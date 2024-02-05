from torch.utils.data import Dataset
from typing import Union, Optional
import pandas as pd
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