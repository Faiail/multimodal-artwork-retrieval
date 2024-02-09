import torch
from typing import Union, Optional
from torchvision.models import ResNet
from src.models.competitors.contextnet.context_net import ContextNet, _init_resnet
from src.models.competitors.basic_garcia.ranker import TfidfEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class OneHotAttributeEncoder:
    def __init__(self, sparse=False):
        self.sparse = sparse
        self.encoder = OneHotEncoder(sparse=sparse)

    def fit(self, vocabulary):
        self.encoder.fit(vocabulary)

    def transform(self, x):
        return self.encoder.transform(x)

    def fit_transform(self, x):
        return self.encoder.fit_transform(x)

    @property
    def vector_size(self):
        dummy_input = np.array(['hello']).reshape(1, -1)
        return self.encoder.transform(dummy_input).shape[1]


class ContextNetRanker(torch.nn.Module):
    def __init__(
            self,
            resnet: Union[dict, ResNet],
            context_net: ContextNet,
            context_net_out_dim: int,
            title_vectorizer: TfidfEncoder,
            comment_vectorizer: TfidfEncoder,
            attribute_vectorizer: OneHotAttributeEncoder,
            hidden_dim: Optional[int] = 128,
            frozen: Optional[bool] = True,
            device: Optional[str] = 'cuda'
    ):
        super().__init__()
        resnet = _init_resnet(resnet)
        resnet_features = resnet.fc.in_features
        resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
        if frozen:
            for p in resnet.parameters():
                p.requires_grad = False
        self.resnet = resnet
        self.context_net = context_net
        self.title_vectorizer = title_vectorizer
        self.comment_vectorizer = comment_vectorizer
        self.attribute_vectorizer = attribute_vectorizer
        self.hidden_dim = hidden_dim
        self.device = device

        self.img_proj = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=(resnet_features + context_net_out_dim),
                out_features=hidden_dim
            ),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(hidden_dim),
        )

        self.text_proj = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.comment_vectorizer.vector_size +
                            self.title_vectorizer.vector_size +
                            self.attribute_vectorizer.vector_size,
                out_features=hidden_dim
            ),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(hidden_dim),
        )

    def encode_image(self, img):
        classes, _ = self.context_net(img)
        img_feats = self.resnet(img)
        shared = torch.cat([img_feats, classes], dim=-1).to(self.device)
        return self.img_proj(shared)

    def encode_text(self, com, tit, att):
        comments = torch.from_numpy(self.comment_vectorizer.transform(com)).to(self.device)
        titles = torch.from_numpy(self.title_vectorizer.transform(tit)).to(self.device)
        attributes = torch.from_numpy(self.attribute_vectorizer.transform(att)).to(self.device)
        shared = torch.cat([comments, titles, attributes], dim=-1).to(self.device)
        return self.text_proj(shared)

    def forward(self, img, com, tit, att):
        img_feats = self.encode_image(img)
        txt_feats = self.encode_text(com, tit, att)
        return img_feats, txt_feats


