import torch
from .fusion_module import FusionModule
from src.models.feature_projector import FeatureProjector
from src.data import DataModality
from enum import Enum


class ResultDict(Enum):
    PRED = "pred"
    FUSED = "fused"
    

class CompleteArtwrokSiameseNetwork(torch.nn.Module):
    """New class that is able to perform everything:
    Text2Image, Image2Text, Artwork Alignment
    """

    def __init__(
        self,
        text2image: FeatureProjector,
        text2graph: FeatureProjector,
        image2text: FeatureProjector,
        image2graph: FeatureProjector,
        fusion_module: FusionModule,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.text2image = text2image
        self.text2graph = text2graph
        self.image2text = image2text
        self.image2graph = image2graph
        self.fusion_module = fusion_module
        self.model = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(
                in_features=self.fusion_module.out_channels * 2, out_features=1
            ),
        )

    def encode(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        if DataModality.IMAGE.value not in x.keys():
            x = self._encode_image(x)
        elif DataModality.TEXT.value not in x.keys():
            x = self._encode_text(x)
        _, x = zip(*sorted(x.items()))
        return x

    def forward(
        self,
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
        return_fused: bool = False,
    ) -> torch.Tensor:
        x = self.encode(x)
        y = self.encode(y)

        fused_a = self.fusion_module(*x)
        fused_b = self.fusion_module(*y)
        shared = torch.cat([fused_a, fused_b], dim=-1)
        out = self.model(shared)
        return (
            out
            if not return_fused
            else {
                ResultDict.PRED: out,
                ResultDict.FUSED: (fused_a, fused_b),
            }
        )

    def _encode_text(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            DataModality.IMAGE.value: x[DataModality.IMAGE.value],
            DataModality.TEXT.value: self.image2text(x[DataModality.IMAGE.value]),
            DataModality.GRAPH.value: (
                self.image2graph(x[DataModality.IMAGE.value])
                if DataModality.GRAPH.value not in x.keys()
                else x[DataModality.GRAPH.value]
            ),
        }

    def _encode_image(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            DataModality.IMAGE.value: self.text2image(x[DataModality.TEXT.value]),
            DataModality.TEXT.value: x[DataModality.TEXT.value],
            DataModality.GRAPH.value: (
                self.text2graph(x[DataModality.TEXT.value])
                if DataModality.GRAPH.value not in x.keys()
                else x[DataModality.GRAPH.value]
            ),
        }
