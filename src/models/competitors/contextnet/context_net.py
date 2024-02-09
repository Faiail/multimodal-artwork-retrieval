import torch
from typing import Union, Optional
from torchvision.models import ResNet, ResNet50_Weights, resnet50


def _init_resnet(resnet):
    mapping_resnet = {'v1': ResNet50_Weights.IMAGENET1K_V1,
                      'v2': ResNet50_Weights.IMAGENET1K_V2}
    if not isinstance(resnet, dict):
        return resnet
    resnet['weights'] = mapping_resnet.get(resnet['weights'], ResNet50_Weights.IMAGENET1K_V2)
    return resnet50(**resnet)


class ContextNet(torch.nn.Module):
    def __init__(
            self,
            resnet: Union[ResNet, dict],
            out_dim: int,
            hidden_dim: int,
            frozen: Optional[bool] = True,
    ):
        super().__init__()
        self.frozen = frozen
        resnet = _init_resnet(resnet)
        if frozen:
            for p in resnet.parameters():
                p.requires_grad = False
            for p in resnet.layer4.parameters():
                p.requires_grad = True
        resnet_dim = resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-1], torch.nn.Flatten())
        self.classifier = torch.nn.Linear(resnet_dim, out_dim)
        self.proj = torch.nn.Linear(resnet_dim, hidden_dim)

    def forward(self, x):
        vis_emb = self.resnet(x)
        pred_class = self.classifier(vis_emb)
        graph_proj = self.proj(vis_emb)
        return [pred_class, graph_proj]
