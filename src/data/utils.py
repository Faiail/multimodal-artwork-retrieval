import os
from safetensors.torch import save_file
import pandas as pd
from enum import Enum


class DataModality(Enum):
    IMAGE = 'image'
    TEXT = 'text'
    GRAPH = 'graph'


class Mode(Enum):
    EMBEDDING = 'embedding'
    RAW = 'raw'


class Score(Enum):
    DISCRETE = 'discrete'
    REAL = 'real'


def save_embeddings(tensor, file_path):
    if not os.path.exists(file_path):
        save_file(tensor, f'{file_path}.safetensors')


def safe_saving(df: pd.DataFrame, out_path):
    directory = '/'.join(out_path.split('/')[:-1])
    os.makedirs(directory, exist_ok=True)
    df.to_csv(out_path, index=False)