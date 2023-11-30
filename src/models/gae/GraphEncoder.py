import torch_geometric as pyg
import torch


class GraphAttentionEncoder(torch.nn.Module):
    def __init__(
            self,
            num_layers=2,
            hidden_channels=[64, 64],
            num_heads=[4, 4],
            dropouts=[.2, .2],
            add_self_loops=False,
            shared_weights=True,
    ):
        super().__init__()
        self._num_layers = num_layers
        self._hidden_channels = hidden_channels
        self._num_heads = num_heads
        self._dropouts = dropouts
        self._add_self_loops = add_self_loops
        self._shared_weights = shared_weights
        assert num_layers == len(num_heads) == len(dropouts) == len(hidden_channels)

        self._convs = []

        for i in range(num_layers):
            self._convs.append(
                pyg.nn.GATv2Conv(
                    (-1, -1),
                    out_channels=self._hidden_channels[i],
                    heads=self._num_heads[i],
                    dropout=self._dropouts[i],
                    add_self_loops=self._add_self_loops,
                    share_weights=self._shared_weights,
                    concat=False,
                )
            )

            self._convs = torch.nn.ModuleList(self._convs)

    def forward(
            self,
            x,
            edge_index,
    ):
        for conv in self._convs:
            x = conv(x, edge_index)
            x = x.relu()
        return x


class HeteroGraphAttentionEncoder(torch.nn.Module):
    def __init__(
            self,
            metadata,
            source_node,
            dest_node,
            final_dimension=512,
            num_layers=2,
            hidden_channels=[64, 64],
            num_heads=[4, 4],
            dropouts=[.2, .2],
            add_self_loops=False,
            shared_weights=True,
    ):
        super().__init__()
        self.metadata = metadata
        self.source_node = source_node
        self.dest_node = dest_node
        self.final_dimension = final_dimension
        self.encoder = GraphAttentionEncoder(
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            dropouts=dropouts,
            add_self_loops=add_self_loops,
            shared_weights=shared_weights,
        )
        self.encoder = pyg.nn.to_hetero(self.encoder, self.metadata)
        self.source_lin = torch.nn.Linear(hidden_channels[-1], # * num_heads[-1],
                                          self.final_dimension)
        self.dest_lin = torch.nn.Linear(hidden_channels[-1], # * num_heads[-1],
                                        self.final_dimension)

    def forward(
            self,
            x,
            edge_index_dict,
    ):
        z = self.encoder(x, edge_index_dict)
        source = z[self.source_node]
        dest = z[self.dest_node]
        return {
            self.source_node: self.source_lin(source),
            self.dest_node: self.dest_lin(dest)
        }



