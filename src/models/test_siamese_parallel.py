from src.models.artwork_siamese_network.fusion_module import FusionModule
from src.models.artwork_siamese_network.ArtworkSiameseNetwork import (
    ArtworkSiameseNetwork,
)
from src.models.feature_projector.FeatureProjector import FeatureProjector
from src.data.SiameseDataset import SiameseCatalogueDataset, SiameseTestDataset
from src.utils import load_ruamel
from src.models.utils import parse_args
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torchmetrics
import os
from typing import Any, Tuple
import json
from accelerate import Accelerator


class Tester:
    def __init__(self, params) -> None:
        self.parameters = params
        self.accelerator = Accelerator()

    def _init(self):
        self._init_model()

    def _init_model(self):
        model_params = self.parameters.get("model")
        self.model = ArtworkSiameseNetwork(**model_params.get("params"))
        self.model = self.accelerator.prepare(self.model)
        self.accelerator.load_state(model_params["state_dict"])


def main():
    params_path = parse_args().params_path
    parameters = load_ruamel(params_path)
    tester = Tester(params=parameters)
    
