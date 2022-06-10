import torch
from torch.nn import Sequential, ReLU
from torch_geometric.nn import Linear, DynamicEdgeConv

class DGCNNLayer(torch.nn.Module):

    def __init__(self, out_channels, k, aggr='mean'):
        super(DGCNNLayer, self).__init__()

        mlp = Sequential(   Linear(-1, out_channels),
                            ReLU(), 
                            Linear(out_channels, out_channels)  )

        self.dgcnn = DynamicEdgeConv(nn=mlp, k=k, aggr=aggr)

    def forward(self, x):
        return self.dgcnn(x)
