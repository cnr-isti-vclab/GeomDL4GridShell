from turtle import forward
import torch
from torch.nn import Sequential,Linear, ReLU, Softplus
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
                                Linear(512, 256),
                                ReLU(),
                                Linear(256, 3)  )

    def forward(self, x):
        # List of dgcnn layer outputs.
        self.out_list = [x]

        # Applying consecutive dgcnn layers.
        for layer in range(1, self.no_dgcnn_layers + 1):
            current_dgcnn_layer = getattr(self, 'layer_' + str(layer))
            self.out_list.append(current_dgcnn_layer(self.out_list[-1]))

        # Chaining all layer outputs.
        dgcnn_out = torch.cat(self.out_list, dim=1)

        # Processing dgcnn_out via shared mlp.
        return self.mlp(dgcnn_out)

    def get_knn(self, x, k, target_idx):
        # Computing layer results.
        self.forward(x)

        # Initializing output lists.
        knn_positions = []

        for layer in self.out_list:
            topk = torch.topk(torch.norm(layer[target_idx] - layer, dim=1), k=k, largest=False)
            knn_positions.append(topk.indices)

        return knn_positions

    @staticmethod
    def weight_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -1e-4, 1e-4)
            m.bias.data.fill_(0.)

