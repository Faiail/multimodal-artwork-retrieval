from torch.utils.data import Dataset
from src.utils import load_tensor
import pandas as pd


class FeatureProjectorDataset(Dataset):
    def __init__(
            self,
            source_modality,
            dest_modality,
            data,
            data_dir,
            preprocess=None,
    ):
        super().__init__()
        self.source_modality = source_modality
        self.dest_modality = dest_modality
        self.data = pd.read_csv(data)
        self.data_dir = data_dir
        self.preprocess = preprocess

    def __getitem__(self, item):
        x = load_tensor(
            file=f'{self.data_dir}/{self.data.iloc[0, 0]}',
            key=self.source_modality,
        )
        y = load_tensor(
            file=f'{self.data_dir}/{self.data.iloc[0, 0]}',
            key=self.dest_modality,
        )
        if self.preprocess:
            x = self.preprocess(x)
            y = self.preprocess(y)
        return x, y

    def __len__(self):
        return len(self.data)