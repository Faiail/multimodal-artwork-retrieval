import torch
from torch.nn.modules.loss import _WeightedLoss
from torchvision.ops import sigmoid_focal_loss


class BinaryFocalLoss(_WeightedLoss):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = 'none',
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
