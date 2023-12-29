import torch


class FusionModule(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            ff_channels,
            out_channels,
            dropout,
            num_heads,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.ff_channels = ff_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_heads = num_heads

        self._init_model()

    def _init_model(self):
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(self.in_channels, self.hidden_channels),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.ReLU(),
        )
        self.attention = torch.nn.MultiheadAttention(
            embed_dim=self.hidden_channels,
            num_heads=self.num_heads,
            dropout=self.dropout,
        )
        self.fforward = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_channels, self.ff_channels),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(self.ff_channels, self.hidden_channels),
            torch.nn.Dropout(p=self.dropout),
        )
        self.layer_norm1 = torch.nn.LayerNorm(self.hidden_channels)
        self.layer_norm2 = torch.nn.LayerNorm(self.hidden_channels)
        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_channels, self.out_channels),
            torch.nn.Dropout(p=self.dropout),
            torch.nn.Tanh(),
        )

    def forward(self, *args, return_attention_weights=False):
        x = torch.stack(args, dim=0)
        x = self.proj(x)

        attn_x, attn_w = self.attention(x, x, x)
        x = x + attn_x
        x = self.layer_norm1(x)

        ff_x = self.fforward(x)
        x = x + ff_x
        x = self.layer_norm2(x)

        x = self.out_layer(x)
        x = torch.mean(x, dim=0)

        if return_attention_weights:
            return x, attn_w
        return x
