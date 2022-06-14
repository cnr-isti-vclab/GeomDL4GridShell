import torch
from torch.nn import Sequential, Tanh
from torch_geometric.nn import Linear

class FeatureTransformLayer(torch.nn.Module):

    def __init__(self, no_blocks, out_channels):
        super(FeatureTransformLayer, self).__init__()
        self.no_blocks = no_blocks

        # Initializing homogenization MLPs.
        for block in range(self.no_blocks):
            setattr(self, 'mlp_' + str(block), Sequential(Linear(-1, out_channels), Tanh()))

    def forward(self, x, mask):
        out_list = []

        for block in range(self.no_blocks):
            current_mlp = getattr(self, 'mlp_' + str(block))
            out_list.append(current_mlp(x[:, mask[block]]))
        
        return torch.cat(out_list, dim=1)

