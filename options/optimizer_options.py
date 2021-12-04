import argparse

class OptimizerOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Options for LacconianOptimizer.')
        self.parser.add_argument('--device', dest='device', type=str, default='cpu', help='Device for pytorch')
        self.parser.add_argument('--meshpath', dest='path', type=str, required=True, help='Path to starting mesh')
        self.parser.add_argument('--niter', dest='n_iter', type=int, default=100, help='Number of optimization steps')
        self.parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Optimization learning rate')
        #self.parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='Optimization momentum')
        self.parser.add_argument('--plot', dest='plot', action='store_true', default=False, help='True if polyscope plotting is required')
        self.parser.add_argument('--save', dest='save', action='store_true', default=False, help='True if mesh saving is required')
        self.parser.add_argument('--interval', dest='interval', type=int, default=25, help='Plotting/saving iterations interval')
        self.parser.add_argument('--savelabel', dest='savelabel', type=str, default='deformation', help='Label for ouputs')

    def parse(self):
        return self.parser.parse_args()