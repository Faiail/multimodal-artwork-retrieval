import torch
from torchvision.models import ResNet
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Union
import joblib
from torchvision.models import ResNet152_Weights, resnet152


class TfidfEncoder(TfidfVectorizer):
    def __init__(
            self,
            max_features=None,
            stop_words=None,
    ):
        super().__init__(max_features=max_features, stop_words=stop_words)

    @property
    def vector_size(self):
        return len(self.get_feature_names_out())


def get_text_encoder(instance):
    if isinstance(instance, TfidfEncoder):
        return instance
    return joblib.load(instance)


def _init_resnet(resnet):
    mapping_resnet = {'v1': ResNet152_Weights.IMAGENET1K_V1,
                      'v2': ResNet152_Weights.IMAGENET1K_V2}
    if not isinstance(resnet, dict):
        return resnet
    resnet['weights'] = mapping_resnet.get(resnet['weights'], ResNet152_Weights.IMAGENET1K_V2)
    return resnet152(**resnet)


class Ranker(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            resnet: Union[ResNet, dict],
            comment_tf_idf_vectorizer: Union[TfidfEncoder, str],
            title_tf_idf_vectorizer: Union[TfidfEncoder, str],
            frozen: bool = True,
            device: str = 'cuda'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.comment_tf_idf_vectorizer = get_text_encoder(comment_tf_idf_vectorizer)
        self.title_tf_idf_vectorizer = get_text_encoder(title_tf_idf_vectorizer)
        self.frozen = frozen
        self.device = device

        resnet = _init_resnet(resnet)
        image_projector = torch.nn.Linear(
            in_features=resnet.fc.in_features,
            out_features=self.hidden_dim
        )
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        if self.frozen:
            for p in resnet.parameters():
                p.requires_grad = False
        self.image_encoder = torch.nn.Sequential(*[
            resnet,
            torch.nn.Flatten(),
            image_projector,
            torch.nn.Tanh(),
            torch.nn.LayerNorm(hidden_dim),
        ])

        text_projector = torch.nn.Linear(
            in_features=self.comment_tf_idf_vectorizer.vector_size + self.title_tf_idf_vectorizer.vector_size,
            out_features=self.hidden_dim
        )

        self.text_encoder = torch.nn.Sequential(*[
            text_projector,
            torch.nn.Tanh(),
            torch.nn.LayerNorm(hidden_dim),
        ])

    def encode_text(self, raw_comment: List[str], raw_title) -> torch.Tensor:
        x_t = self.title_tf_idf_vectorizer.transform(raw_title).toarray()
        x_t = torch.as_tensor(x_t, device=self.device, dtype=torch.float)
        x_c = self.comment_tf_idf_vectorizer.transform(raw_comment).toarray()
        x_c = torch.as_tensor(x_c, device=self.device, dtype=torch.float)
        x = torch.cat([x_c, x_t], dim=1)
        return self.text_encoder(x)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def forward(self, images, comments, titles):
        return self.encode_image(images), self.encode_text(comments, titles)

