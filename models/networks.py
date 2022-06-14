import torch
from torch.nn import Sequential, ReLU
from torch_geometric.nn import Linear
from models.layers.dgcnn_layer import DGCNNLayer
from models.layers.gatv2_layer import GATv2Layer

class DGCNNDisplacerNet(torch.nn.Module):

    def __init__(self, k, aggr='mean', out_channels_list=[64, 128, 256, 512]):
        super(DGCNNDisplacerNet, self).__init__()

        self.no_dgcnn_layers = len(out_channels_list)

        # First DGCNN layer.
        self.layer_1 = DGCNNLayer(out_channels_list[0], k, aggr)

        # Following DGCNN layers.
        for layer in range(len(out_channels_list) - 1):
            setattr(self, 'layer_' + str(layer + 2), DGCNNLayer(out_channels_list[layer + 1], k, aggr))

        # Shared mlp.
        self.mlp = Sequential(  Linear(-1, 512),
                                ReLU(),
                                Linear(-1, 256),
                                ReLU(),
                                Linear(-1, 3)  )

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

    @staticmethod
    def weight_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.uniform_(m.weight, -1e-4, 1e-4)
            m.bias.data.fill_(0.)



class GATv2DisplacerNet(torch.nn.Module):

    def __init__(self, k, out_channels_list=[64, 128, 256, 512]):
        super(GATv2DisplacerNet, self).__init__()

        self.no_gat_layers = len(out_channels_list)

        # First GATv2 layer.
        self.layer_1 = GATv2Layer(out_channels_list[0], k)

        # Following DGCNN layers.
        for layer in range(len(out_channels_list) - 1):
            setattr(self, 'layer_' + str(layer + 2), GATv2Layer(out_channels_list[layer + 1], k))

        # Shared mlp.
        self.mlp = Sequential(  Linear(-1, 512),
                                ReLU(),
                                Linear(-1, 256),
                                ReLU(),
                                Linear(-1, 3)  )

    def forward(self, x):
        # List of dgcnn layer outputs.
        self.out_list = [x]

        # Applying consecutive dgcnn layers.
        for layer in range(1, self.no_gat_layers + 1):
            current_gat_layer = getattr(self, 'layer_' + str(layer))
            self.out_list.append(current_gat_layer(self.out_list[-1]))

        # Chaining all layer outputs.
        dgcnn_out = torch.cat(self.out_list, dim=1)

        # Processing dgcnn_out via shared mlp.
        return self.mlp(dgcnn_out)
