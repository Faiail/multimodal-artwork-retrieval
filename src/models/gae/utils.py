import torch
import torch_geometric as pyg


def negative_sampling(triplets, size_source, size_dest):
    out = torch.empty(triplets.size())
    source = torch.randint(low=0, high=size_source, size=out.size(1))
    dest = []
    
