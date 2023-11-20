import argparse

class NetOptimizerOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Options for LacconianNetOptimizer.')
        self.parser.add_argument('--device', dest='device', type=str, default='cpu', help='Device for pytorch')
        self.parser.add_argument('--meshpath', dest='path', type=str, required=True, help='Path to starting mesh')
        self.parser.add_argument('--maxiter', dest='max_iter', type=int, default=1500, help='Maximum number of optimization steps')
        self.parser.add_argument('--lr', dest='lr', type=float, default=5e-4, help='Optimization learning rate')
        self.parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='Optimization momentum')
        self.parser.add_argument('--save', dest='save', action='store_true', default=True, help='True if mesh saving is required')
        self.parser.add_argument('--saveinterval', dest='save_interval', type=int, default=100, help='Mesh saving iterations interval')
        self.parser.add_argument('--saveprefix', dest='save_prefix', type=str, default='', help='Directory for saves.')
        self.parser.add_argument('--displayinterval', dest='display_interval', type=int, default=1, help='Loss display iterations interval')
        self.parser.add_argument('--savelabel', dest='save_label', type=str, required=True, help='Label for ouputs')
        self.parser.add_argument('--losstype', dest='loss_type', type=str, default='mean_beam_energy', help='Type of loss computed')
        self.parser.add_argument('--itertimes', dest='take_times', action='store_true', default=False, help='If iteration and backward times are required or not')
        self.parser.add_argument('--knn', dest='no_knn', type=int, default=16, help='Number of nearest neighbors in dgcnn layers')
        self.parser.add_argument('--transforminputfeatures', dest='transform_in_features', action='store_true', default='True', help='If cluster-wise feature encoding is required or not')
        self.parser.add_argument('--getloss', dest='get_loss', action='store_true', default='True', help='If loss vector for printing curves is required or not')
        self.parser.add_argument('--layermode', dest='layer_mode', default='gat', help='Which kind of layer is used for the graph module')
        self.parser.add_argument('--neighborlist', dest='neighbor_list', nargs='+', type=int, default=[])

    def parse(self):
        return self.parser.parse_args()