import argparse

class OptimizerOptions:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Options for LacconianOptimizer.')
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
        self.parser.add_argument('--initmode', dest='init_mode', type=str, default='stress_aided', help='Initial deformation rules')
        self.parser.add_argument('--beamload', dest='beam_have_load', action='store_true', default=False, help='If beams give load to verts or not')
        self.parser.add_argument('--itertimes', dest='take_times', action='store_true', default=False, help='If iteration and backward times are required or not')
        self.parser.add_argument('--laplaciansmooth', dest='with_laplacian_smooth', action='store_true', default=False, help='If laplacian mesh smoothing is requested or not')
        self.parser.add_argument('--normalconsistency', dest='with_normal_consistency', action='store_true', default=False, help='If mesh normal consistency is requested or not')
        self.parser.add_argument('--varfaceareas', dest='with_var_face_areas', action='store_true', default=False, help='If face area variance minimizing is requested or not')
        self.parser.add_argument('--laplsmoothlossperc', dest='laplsmooth_loss_perc', type=float, default=-1, help='laplacian smoothing percentual weight w.r.t. lacconian loss')
        self.parser.add_argument('--normconslossperc', dest='normcons_loss_perc', type=float, default=-1, help='normal consistency percentual weight w.r.t. lacconian loss')
        self.parser.add_argument('--varfaceareaslossperc', dest='varfaceareas_loss_perc', type=float, default=-1, help='face area variance percentual weight w.r.t. lacconian loss')
        self.parser.add_argument('--boundaryreg', dest='boundary_reg', action='store_true', default=False, help='If boundary normal consistency reg is employed or not')
        self.parser.add_argument('--remeshing', dest='with_remeshing', action='store_true', default=False, help='If periodic remeshing is requested or not')
        self.parser.add_argument('--remeshinginterval', dest='remeshing_interval', type=int, default=1000, help='Isotrophic remeshing iterations interval')
        self.parser.add_argument('--notsmoothedcloud', dest='see_not_smoothed', action='store_true', default=False, help='If we want not smoothed point cloud or not')


    def parse(self):
        return self.parser.parse_args()