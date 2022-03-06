from turtle import forward
import torch
from torch.nn import Sequential,Linear, ReLU
from models.layers.dgcnn_layer import DGCNNLayer

class DGCNNDisplacerNet(torch.nn.Module):

    def __init__(self, input_channels, k, aggr='mean', out_channels_list=[64, 128, 256, 512]):
        super(DGCNNDisplacerNet, self).__init__()

        self.no_dgcnn_layers = len(out_channels_list)

        # First DGCNN layer.
        self.layer_1 = DGCNNLayer(input_channels, out_channels_list[0], k, aggr)

        # Following DGCNN layers.
        for layer in range(len(out_channels_list) - 1):
            setattr(self, 'layer_' + str(layer + 2), DGCNNLayer(out_channels_list[layer], out_channels_list[layer + 1], k, aggr))

        # Shared mlp.
        self.mlp = Sequential(  Linear(input_channels + sum(out_channels_list), 512),
                                ReLU(),
                                Linear(512, 32),
                                ReLU(),
                                Linear(32, 3)   )

    def forward(self, x, mask):
        # List of dgcnn layer outputs.
        out_list = [x]

        # Applying consecutive dgcnn layers.
        for layer in range(1, self.no_dgcnn_layers + 1):
            current_dgcnn_layer = getattr(self, 'layer_' + str(layer))
            out_list.append(current_dgcnn_layer(out_list[-1]))

        # Chaining all layer outputs.
        dgcnn_out = torch.cat(out_list, dim=1)

        # Removing constrainted vertices from deep feature tensor.
        reduced_dgcnn_out = dgcnn_out[mask, :]

        # Processing dgcnn_out via shared mlp.
        return self.mlp(reduced_dgcnn_out)

    @staticmethod
    def uniform_weight_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.05, 0.05)
            m.bias.data.fill_(0.)

