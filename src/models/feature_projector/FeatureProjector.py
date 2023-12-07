import torch
from enum import Enum, IntEnum


class Strategy(IntEnum):
    SAME = 0
    HALF = 1
    DOUBLE = 2


def get_act(act):
    if act == 'relu':
        return torch.nn.ReLU()
    if act == 'tanh':
        return torch.nn.Tanh()
    raise ValueError('Act parameter must be in [relu, tanh]')


class FeatureProjector(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            strategy,
            n_layers,
            act,
            dropout,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strategy = strategy
        self.strategy_func = self._get_strategy_func()
        self.n_layers = n_layers
        self.act = get_act(act)
        self.dropout = dropout
        self.model = self._get_model()

    def _get_model(self):
        layers = []
        in_channels = self.in_channels
        for _ in range(self.n_layers - 1):
            layers.append(
                torch.nn.Linear(in_channels, self.strategy_func(in_channels))
            )
            layers.append(self.act)
            layers.append(torch.nn.Dropout(p=self.dropout))
            in_channels = self.strategy_func(in_channels)
        layers.append(
            torch.nn.Linear(in_channels, self.out_channels)
        )
        layers.append(torch.nn.Tanh())  # final layer activation
        return torch.nn.ModuleList(layers)

    def _get_strategy_func(self):
        if self.strategy == Strategy.SAME:
            return lambda x: x
        if self.strategy == Strategy.HALF:
            return lambda x: x // 2
        if self.strategy == Strategy.DOUBLE:
            return lambda x: x * 2
        raise ValueError(f'Strategy must be in {[Strategy.SAME, Strategy.DOUBLE, Strategy.DOUBLE]}')

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
