import torch
from src.models.artwork_siamese_network.fusion_module import FusionModule
from src.models.feature_projector.FeatureProjector import Strategy


class ArtworkSiameseNetwork(torch.nn.Module):
    def __init__(
            self,
            fusion_in_channels,
            fusion_hidden_channels,
            fusion_ff_channels,
            fusion_out_channels,
            fusion_num_heads,
            dropout,
            strategy,
            num_layers,
    ):
        super().__init__()
        self.model_in_channels = fusion_out_channels
        self.strategy = strategy
        self.num_layers = num_layers
        self.dropout = dropout
        self.strategy_func = self._get_strategy_func()
        self.fusion_module = FusionModule(
            in_channels=fusion_in_channels,
            hidden_channels=fusion_hidden_channels,
            ff_channels=fusion_ff_channels,
            out_channels=fusion_out_channels,
            num_heads=fusion_num_heads,
            dropout=dropout,
        )
        self.model = self._init_model()

    def _get_strategy_func(self):
        if self.strategy == Strategy.SAME:
            return lambda x: x
        if self.strategy == Strategy.HALF:
            return lambda x: x // 2
        if self.strategy == Strategy.DOUBLE:
            return lambda x: x * 2
        raise ValueError(f'Strategy must be in {[Strategy.SAME, Strategy.DOUBLE, Strategy.DOUBLE]}')

    def _init_model(self):
        in_channels = self.model_in_channels * 2
        model = []
        for _ in range(self.num_layers):
            model.append(
                torch.nn.Linear(in_channels, self.strategy_func(in_channels))
            )
            model.append(
                torch.nn.Dropout(p=self.dropout)
            )
            model.append(torch.nn.ReLU())
            in_channels = self.strategy_func(in_channels)

        model.append(torch.nn.Linear(in_channels, 1))

        return torch.nn.Sequential(*model)

    def forward(self, x1, x2):
        fused_a = self.fusion_module(*x1)
        fused_b = self.fusion_module(*x2)
        shared = torch.cat([fused_a, fused_b])
        return self.model(shared)