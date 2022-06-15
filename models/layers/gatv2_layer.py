import torch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import knn_graph

class GATv2Layer(torch.nn.Module):

    def __init__(self, out_channels, k, heads=3):
        super(GATv2Layer, self).__init__()

        self.k = k
        self.gatv2 = GATv2Conv(in_channels=-1, out_channels=out_channels, heads=heads, concat=False)

    def forward(self, x):
        edge_index = knn_graph(x, self.k, loop=False)
        return self.gatv2(x, edge_index)
