from torch.utils.data import Dataset
from src.data.utils import DataModality, Mode, Score
from typing import List, Any, Dict
import pandas as pd
import random
from src.utils import load_tensor
import torch
from PIL import Image


class CompleteSiameseDataset(Dataset):
    def __init__(
        self,
        possible_modalities: List[Any],
        mode: Dict[str, str],
        data_dirs: Dict[str, str],
        dataset: pd.DataFrame,
        names: pd.DataFrame,
        score_strategy: str,
        preprocess=None,
        max_retry: int = 3,
    ) -> None:
        super().__init__()
        if preprocess is None:
            preprocess = dict()
        self.possible_modalities = possible_modalities
        self.mode = mode
        self.data_dirs = data_dirs
        self.dataset = (
            dataset
            if isinstance(dataset, pd.DataFrame)
            else pd.read_csv(dataset, index_col=0)
        )
        self.names = names if isinstance(names, pd.DataFrame) else pd.read_csv(names)
        self.preprocess = preprocess if preprocess else {}
        self.score_strategy = score_strategy
        self.max_retry = max_retry
        
        assert len(possible_modalities) == len(data_dirs) == len(mode)
            
    def _get_image(self, fname):
        if self.mode[DataModality.IMAGE.value] == Mode.RAW.value:
            image_fname = fname.split('.')[0] + '.jpg'
            img = Image.open(f'{self.data_dirs[DataModality.IMAGE.value]}/{image_fname}').convert('RGB')
            preprocess = self.preprocess.get(DataModality.IMAGE.value, None)
            if preprocess:
                for t in range(self.max_retry):
                    x = preprocess(img)
                    if not torch.isnan(x).any():
                        break
            assert not torch.isnan(x).any(), "nan value"
            return x
        return load_tensor(
            file=f'{self.data_dirs[DataModality.IMAGE.value]}/{fname}',
            key=DataModality.IMAGE.value,
        )

    def _get_text(self, fname):
        if self.mode[DataModality.TEXT.value] == Mode.RAW.value:
            if not isinstance(self.data_dirs[DataModality.TEXT.value], pd.DataFrame):
                self.data_dirs[DataModality.TEXT.value] = pd.read_csv(
                    self.data_dirs[DataModality.TEXT.value],
                    index_col=0,
                )
            x = self.data_dirs[DataModality.TEXT.value].loc[fname].tolist()[0]
            preprocess = self.preprocess.get(DataModality.TEXT.value, None)
            if preprocess:
                for step in preprocess:
                    x = step.augment(x)
            return x
        return load_tensor(
            file=f'{self.data_dirs[DataModality.TEXT.value]}/{fname}',
            key=DataModality.TEXT.value,
        )

    def _get_graph(self, fname):
        assert self.mode[DataModality.GRAPH.value] == Mode.EMBEDDING.value
        return load_tensor(
            file=f'{self.data_dirs[DataModality.GRAPH.value]}/{fname}',
            key=DataModality.GRAPH.value,
        )

    def get_modality(self, fname, modality):
        if modality == DataModality.IMAGE.value:
            return self._get_image(fname)
        if modality == DataModality.TEXT.value:
            return self._get_text(fname)
        if modality == DataModality.GRAPH.value:
            return self._get_graph(fname)

    def get_score(self, score):
        if self.score_strategy == Score.DISCRETE.value:
            return 1 if score >= 0.5 else 0
        if self.score_strategy == Score.REAL.value:
            return score
        raise ValueError(f'Cannot handle {self.score_strategy} score strategy.')

    def __getitem__(self, index) -> Any:
        x, y, score = self.dataset.iloc[index]
        x, y = self.names.iloc[[x, y], 0].tolist()
        x = {mod: self.get_modality(x, mod) for mod in self.possible_modalities}
        y = {mod: self.get_modality(y, mod) for mod in self.possible_modalities}
        
        score = self.get_score(score)
        return {
            "x": x,
            "y": y,
            "score": score,
        }

    def __len__(self):
        return len(self.dataset)
