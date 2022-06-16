import torch
from torch.nn import Sequential, ReLU
from torch_geometric.nn import Linear
from models.layers.feature_transform_layer import FeatureTransformLayer
from models.layers.dgcnn_layer import DGCNNLayer
from models.layers.gatv2_layer import GATv2Layer

class DisplacerNet(torch.nn.Module):

    def __init__(self, k, mode='gat', out_channels_list=[64, 128, 256, 512], in_feature_mask=None):
        super(DisplacerNet, self).__init__()
        self.mode = mode

        self.no_dgcnn_layers = len(out_channels_list)

        # Feature transform layer, if requested.
        if in_feature_mask is not None:
            self.feature_transf = FeatureTransformLayer(mask=in_feature_mask, out_channels=256)

        # First layer.
        if self.mode == 'dgcnn':
            self.layer_1 = DGCNNLayer(out_channels_list[0], k, aggr='mean')
        elif self.mode == 'gat':
            self.layer_1 = GATv2Layer(out_channels_list[0], k)
        else:
            raise ValueError('Non valid layer mode inserted.')

        # Following layers.
        for layer in range(len(out_channels_list) - 1):
            if self.mode == 'dgcnn':
                setattr(self, 'layer_' + str(layer + 2), DGCNNLayer(out_channels_list[layer + 1], k, aggr='mean'))
            elif self.mode == 'gat':
                setattr(self, 'layer_' + str(layer + 2), GATv2Layer(out_channels_list[layer + 1], k))

        # Shared mlp.
        self.mlp = Sequential(  Linear(-1, 512),
                                ReLU(),
                                Linear(-1, 256),
                                ReLU(),
                                Linear(-1, 3)  )

    def forward(self, x):
        # Transforming input features, if requested.
        if hasattr(self, 'feature_transf'):
            x = self.feature_transf(x)

        # List of layer outputs.
        self.out_list = [x]

        # Applying consecutive layers.
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
