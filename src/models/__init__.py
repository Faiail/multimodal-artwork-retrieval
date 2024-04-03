from src.models.competitors.basic_garcia.ranker import Ranker
from src.models.artwork_siamese_network.ArtworkSiameseNetwork import (
    ArtworkSiameseNetwork,
    FusionModule,
)
from src.models.feature_projector.FeatureProjector import FeatureProjector
from .artwork_siamese_network.complete_artwork_siamese_network import (
    CompleteArtwrokSiameseNetwork,
)
from typing import Any


def build_complete_siamese_network(
    params: dict[str, Any]
) -> CompleteArtwrokSiameseNetwork:
    text2image = build_feature_projector(params.get("text2image"))
    text2graph = build_feature_projector(params.get("text2graph"))
    image2text = build_feature_projector(params.get("image2text"))
    image2graph = build_feature_projector(params.get("image2graph"))
    fusion_module = build_fusion_module(params.get("fusion_module"))
    return CompleteArtwrokSiameseNetwork(
        text2image=text2image,
        text2graph=text2graph,
        image2text=image2text,
        image2graph=image2graph,
        fusion_module=fusion_module,
        dropout=params.get("dropout"),
    )


def build_feature_projector(params: dict[str, Any]) -> FeatureProjector:
    return FeatureProjector(**params)


def build_fusion_module(params: dict[str, Any]) -> FusionModule:
    return FusionModule(**params)


model_registry = {
    "complete_siamese_network": build_complete_siamese_network,
}


from .artwork_siamese_network.complete_artwork_siamese_network import ResultDict