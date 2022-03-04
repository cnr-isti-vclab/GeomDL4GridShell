import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import DynamicEdgeConv

class DGCNNLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k, aggr='mean'):
        super(DGCNNLayer, self).__init__()

        mlp = Sequential(   Linear(2 * in_channels, out_channels),
                            ReLU(), 
                            Linear(out_channels, out_channels)  )

        self.dgcnn = DynamicEdgeConv(nn=mlp, k=k, aggr=aggr)

    def forward(self, x):
        return self.dgcnn(x)
