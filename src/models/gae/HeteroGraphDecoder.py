import torch


class HeteroInnerProductDecoder(torch.nn.Module):
    def __init__(
            self,
            source_node,
            rel,
            dest_node,
    ):
        super().__init__()
        self._source_node = source_node
        self._rel = rel
        self._dest_node = dest_node

    def forward(
            self,
            z,
            edge_index,
            sigmoid=True,
    ):
        triplets = edge_index[self._source_node, self._rel, self._dest_node]
        value = (z[self._source_node][triplets[0]] * z[self._dest_node][triplets[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(
            self,
            z,
            sigmoid=True,
    ):
        adj = torch.matmul(z[self._source_node], z[self._dest_node])
        return torch.sigmoid(adj) if sigmoid else adj
