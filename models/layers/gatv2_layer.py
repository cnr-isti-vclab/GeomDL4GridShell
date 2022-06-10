import torch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import knn_graph

class GATv2Layer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k):
        super(GATv2Layer, self).__init__()

        self.k = k
        self.gatv2 = GATv2Conv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        edge_index = knn_graph(x, self.k, loop=False)
        return self.gatv2(x, edge_index)