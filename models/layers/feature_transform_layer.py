import torch
from torch.nn import Sequential, Tanh, Sigmoid
from torch_geometric.nn import Linear

class FeatureTransformLayer(torch.nn.Module):

    def __init__(self, mask, out_channels, activation='tanh'):
        super(FeatureTransformLayer, self).__init__()
        self.mask = mask
        self.no_blocks = len(mask)

        # Initializing MLPs.
        for block in range(self.no_blocks):
            if activation == 'tanh':
                setattr(self, 'mlp_' + str(block), Sequential(Linear(-1, out_channels), Tanh()))
            elif activation == 'sigmoid':
                setattr(self, 'mlp_' + str(block), Sequential(Linear(-1, out_channels), Sigmoid()))
            else:
                raise ValueError('FeatureTransformLayer: inserted activation is not valid.')

    def forward(self, x):
        out_list = []

        for block in range(self.no_blocks):
            current_mlp = getattr(self, 'mlp_' + str(block))
            out_list.append(current_mlp(x[:, self.mask[block]]))
        
        return torch.cat(out_list, dim=1)

