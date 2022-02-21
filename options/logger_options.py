import argparse

class WandbLoggerOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Options for WandbLogger.')
        self.parser.add_argument('--device', dest='device', type=str, default='cpu', help='Device for pytorch')
        self.parser.add_argument('--project', dest='project', required=True, type=str, default='cpu', help='Wandb project name')
        self.parser.add_argument('--niter', dest='n_iter', type=int, default=100, help='Number of optimization steps')
        self.parser.add_argument('--remeshing', dest='with_remeshing', action='store_true', default=False, help='If periodic remeshing is requested or not')
        self.parser.add_argument('--remeshinginterval', dest='remeshing_interval', type=int, default=1000, help='Isotrophic remeshing iterations interval')
     
    def parse(self):
        return self.parser.parse_args()