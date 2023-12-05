import torch_geometric as pyg
import torch
from src.models.gae.GraphEncoder import HeteroGraphAttentionEncoder
from src.models.gae.HeteroGraphDecoder import HeteroInnerProductDecoder
from sklearn.metrics import average_precision_score, roc_auc_score
from copy import deepcopy


class HeteroGAE(torch.nn.Module):
    def __init__(
            self,
            metadata,
            source_node,
            rel,
            dest_node,
            encoder_num_layers,
            encoder_hidden_channels,
            encoder_num_heads,
            encoder_dropouts,
            encoder_add_self_loops,
            encoder_shared_weights,
            final_dimension,
    ):
        super().__init__()
        self._metadata = metadata
        self._source_node = source_node
        self._rel = rel
        self._dest_node = dest_node

        self.encoder = HeteroGraphAttentionEncoder(
            metadata=metadata,
            source_node=source_node,
            dest_node=dest_node,
            final_dimension=final_dimension,
            num_layers=encoder_num_layers,
            hidden_channels=encoder_hidden_channels,
            num_heads=encoder_num_heads,
            dropouts=encoder_dropouts,
            add_self_loops=encoder_add_self_loops,
            shared_weights=encoder_shared_weights,
        )
        self.decoder = HeteroInnerProductDecoder(
            source_node=source_node,
            rel=rel,
            dest_node=dest_node
        )

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_index, sigmoid=True):
        return self.decoder(z, edge_index, sigmoid)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        EPS = 1e-15
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS
        ).mean()

        if neg_edge_index is None:
            neg_edge_index = deepcopy(pos_edge_index)
            neg_edge_index[self._source_node, self._rel, self._dest_node] = pyg.utils.negative_sampling(
                edge_index=pos_edge_index[self._source_node, self._rel, self._dest_node],
                num_nodes=(z[self._source_node].size(0), z[self._dest_node].size(0))
            )
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index=None):
        if neg_edge_index is None:
            neg_edge_index = deepcopy(pos_edge_index)
            neg_edge_index[self._source_node, self._rel, self._dest_node] = pyg.utils.negative_sampling(
                edge_index=pos_edge_index[self._source_node, self._rel, self._dest_node],
                num_nodes=(z[self._source_node].size(0), z[self._dest_node].size(0))
            )
        device = z[list(z.keys())[0]].device
        pos_y = torch.ones(pos_edge_index[self._source_node, self._rel, self._dest_node].size(1)).to(device)
        neg_y = torch.zeros(neg_edge_index[self._source_node, self._rel, self._dest_node].size(1)).to(device)
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


if __name__ == '__main__':
    artgraph = torch.load('../../../data/processed/artgraph_split_open_clip/train/train_data.pt')
    artgraph = pyg.transforms.ToUndirected()(artgraph).to('cuda')

    model = HeteroGAE(
        metadata=artgraph.metadata(),
        source_node='artwork',
        rel='hasstyle',
        dest_node='style',
        encoder_dropouts=[.2, .2],
        encoder_num_heads=[2, 2],
        encoder_shared_weights=True,
        encoder_add_self_loops=False,
        encoder_hidden_channels=[128, 128],
        encoder_num_layers=2,
        final_dimension=512,
    ).to('cuda')

    print(model.test(model.encode(artgraph.x_dict, artgraph.edge_index_dict), artgraph.edge_index_dict))
    test_set = torch.load('../../../data/processed/artgraph_split_open_clip/test/test_data.pt')
    test_set = pyg.transforms.ToUndirected()(test_set).to('cuda')
    print(model.test(model.encode(test_set.x_dict, test_set.edge_index_dict), test_set.edge_index_dict))

