import wandb
import os
import itertools
import pandas as pd
import torch
from LacconianOptimizer import LacconianOptimizer
from options.logger_options import WandbLoggerOptions

PARAMS = ['INDEX', 'MESH', 'LOSS', 'LR', 'MOMENTUM','LAPLACIAN_PERC', 'NORMCONS_PERC', 'VARAREA_PERC', 'BOUNDARY_REG']

MESHES = ['sphericalCup.ply', 'geodesic.ply']
LOSSES = ['mean_beam_energy']
LR = [1e-8, 1e-7, 1e-6]
MOMENTUM = [0.9]
LAPLACIAN_PERC = [0.1, 0.8]
NORMCONS_PERC = [0.1, 0.2, 0.8]
VARAREA_PERC = [0, 0.1]
BOUNDARY_REG = [True]

LIST_OF_LISTS = [MESHES, LOSSES, LR, MOMENTUM, LAPLACIAN_PERC, NORMCONS_PERC, VARAREA_PERC, BOUNDARY_REG]


class WandbLogger:
    
    def __init__(self, device, project, n_iter, with_remeshing, remeshing_interval):
        self.device = torch.device(device)
        self.project = project
        self.n_iter = n_iter
        self.with_remeshing = with_remeshing
        self.remeshing_interval = remeshing_interval

        # Making Results directory.
        if not os.path.exists('Results'):
            os.mkdir('Results')
            print("Directory Results created.")
        else:    
            print("Directory Results already exists.")

        # Making Meshes directory.
        if not os.path.exists('Results/Meshes'):
            os.mkdir('Results/Meshes')
            print("Directory Meshes created.")
        else:    
            print("Directory Meshes already exists.")
        
        # Building experiments.
        index = 0
        power_list = []
        for seq in itertools.product(*LIST_OF_LISTS):
            power_list.append((index, *seq))
            index +=1 

        # Keeping experiments number.
        self.no_experiments = len(power_list)
        print(str(self.no_experiments), 'experiments found.')

        # Aggregating to-do-experiments as pandas.DataFrame.
        self.experiments = pd.DataFrame(power_list, columns = PARAMS)
        self.experiments.to_excel("Results/experiments.xlsx", index = False)  
        self.experiments.to_csv("Results/experiments.csv", index = False)

    def start(self):
        # Executing row-wise wandb runs. 
        print('*** Now executing wandb runs. ***')
        for idx, row in self.experiments.iterrows():
            print('Run ' + str(idx+1) + ' of ' + str(self.no_experiments) + '.')
            self.make_run(row.to_dict())

    def make_run(self, row):
        # Starting wandb run.
        run = wandb.init(project=self.project, config=row, group=row['MESH'], entity='andfav', reinit=True)

        # Setting run name.
        name = str(row['INDEX']) + ' - ' + row['MESH']
        run.name = name
        run.save()

        # Defining metrics.
        wandb.define_metric('loss', summary='min')                      # Total loss, i.e. sum of all components.
        wandb.define_metric('structural_loss', summary='min')           # Structural loss component, specificed in row['LOSS'].
        wandb.define_metric('max_displacement_norm', summary='max')     # Max norm of vertex displacements from original mesh.
        wandb.define_metric('max_load_deformation_norm', summary='min') # Max norm of vertex load deformations.
        wandb.define_metric('laplacian_smoothing', summary='min')       # Laplacian smoothing loss component.
        wandb.define_metric('normal_consistency', summary='min')        # Normal consistency loss component.
        wandb.define_metric('var_face_areas', summary='min')            # Face area variance loss component.

        # Making current run mesh directory.
        path = 'Results/Meshes/' + str(row['INDEX'])
        if not os.path.exists(path):
            os.mkdir(path)

        # Optimizer settings.
        source_path = 'meshes/' + row['MESH']                   # Source mesh path.
        lr = row['LR']                                          # Optimizer learning rate.
        momentum = row['MOMENTUM']                              # Optimizer momentum.
        init_mode = 'uniform'                                   # Optimizer initialization mode.
        beam_have_load = True                                   # If beam load is computed or not.
        loss_type = row['LOSS']                                 # Which structural loss is computed.
        with_laplacian_smooth = True                            # If laplacian regularization is employed or not.
        with_normal_consistency = True                          # If normal consistency regularization is employed or not.
        with_var_face_areas = True                              # If face area variance regularization is employed or not.
        laplsmooth_loss_perc = row['LAPLACIAN_PERC']            # Laplacian regularization percentual on structural loss.
        normcons_loss_perc = row['NORMCONS_PERC']               # Normal consistency regularization percentual on structural loss.
        varfaceareas_loss_perc = row['VARAREA_PERC']            # Face area variance regularization percentual on structural loss.
        boundary_reg = row['BOUNDARY_REG']                      # If we want to regularize normals along boundary or not.
        n_iter = self.n_iter                                    # Number of iterations per experiment.
        save = True                                             # If we want to save meshes during iterations or not.
        save_interval = 50                                      # Iteration interval between mesh saves.
        display_interval = -1                                   # Iteration interval between loss displays.
        save_label = row['MESH'][:-4]                           # Label of saved mesh.
        see_not_smoothed = False                                # If we want to see not smoothed point cloud or not.
        take_times = False                                      # If we want to see iteration and backward times or not.
        save_prefix = path + '/'                                # Path of current run saves.

        optimizer = LacconianOptimizer(source_path, lr, momentum, self.device, init_mode, beam_have_load, loss_type, with_laplacian_smooth, with_normal_consistency, with_var_face_areas, laplsmooth_loss_perc, normcons_loss_perc, varfaceareas_loss_perc, boundary_reg)
        print('Optimizing (run ' + str(row['INDEX']+1) + ' of ' + str(self.no_experiments) + ' ) ...')
        optimizer.optimize(n_iter, save, save_interval, display_interval, save_label, take_times, self.with_remeshing, self.remeshing_interval, see_not_smoothed, save_prefix=save_prefix, wandb_run=run)
        

if __name__ == '__main__':
    parser = WandbLoggerOptions()
    options = parser.parse()
    wandb_logger = WandbLogger(options.device, options.project, options.n_iter, options.with_remeshing, options.remeshing_interval)
    wandb_logger.start()

