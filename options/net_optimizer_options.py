import argparse

class NetOptimizerOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Options for LacconianNetOptimizer.')
        self.parser.add_argument('--device', dest='device', type=str, default='cpu', help='Device for pytorch')
        self.parser.add_argument('--meshpath', dest='path', type=str, required=True, help='Path to starting mesh')
        self.parser.add_argument('--niter', dest='n_iter', type=int, default=100, help='Number of optimization steps')
        self.parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Optimization learning rate')
        self.parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='Optimization momentum')
        self.parser.add_argument('--save', dest='save', action='store_true', default=False, help='True if mesh saving is required')
        self.parser.add_argument('--saveinterval', dest='save_interval', type=int, default=25, help='Mesh saving iterations interval')
        self.parser.add_argument('--displayinterval', dest='display_interval', type=int, default=1, help='Loss display iterations interval')
        self.parser.add_argument('--savelabel', dest='save_label', type=str, default='deformation', help='Label for ouputs')
        self.parser.add_argument('--losstype', dest='loss_type', type=str, default='norm_vertex_deformations', help='Type of loss computed')
        self.parser.add_argument('--beamload', dest='beam_have_load', action='store_true', default=False, help='If beams give load to verts or not')
        self.parser.add_argument('--itertimes', dest='take_times', action='store_true', default=False, help='If iteration and backward times are required or not')
        self.parser.add_argument('--knn', dest='no_knn', type=int, default=16, help='Number of nearest neighbors in dgcnn layers')
        self.parser.add_argument('--transforminputfeatures', dest='transform_in_features', action='store_true', default='False')
        self.parser.add_argument('--getloss', dest='get_loss', action='store_true', default='False')

    def parse(self):
        return self.parser.parse_args()