import wandb
import os
import itertools
import pandas as pd
import torch
from LacconianOptimizer import LacconianOptimizer
from options.logger_options import WandbLoggerOptions

PARAMS = ['INDEX', 'MESH', 'LOSS', 'LR', 'LAPLACIAN_PERC', 'NORMCONS_PERC']

MESHES = ['sphericalCup.ply', 'geodesic.ply']
LOSSES = ['mean_beam_energy']
LR = [1e-8, 1e-7, 1e-6]
LAPLACIAN_PERC = [0.1, 0.8]
NORMCONS_PERC = [0.1, 0.2, 0.8]

LIST_OF_LISTS = [MESHES, LOSSES, LR, LAPLACIAN_PERC, NORMCONS_PERC]


class WandbLogger:
    
    def __init__(self, device, project, n_iter):
        self.device = torch.device(device)
        self.project = project
        self.n_iter = n_iter

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
        wandb.define_metric('max_displacement_norm', summary='min')     # Max norm of vertex displacements from original mesh.
        wandb.define_metric('max_deformation_norm', summary='min')      # Max norm of vertex load deformations.
        wandb.define_metric('laplacian_smoothing', summary='min')       # Laplacian smoothing loss component.
        wandb.define_metric('normal_consistency', summary='min')        # Normal consistency loss component.

        # Making current run mesh directory.
        path = 'Results/Meshes/' + str(row['INDEX'])
        if not os.path.exists(path):
            os.mkdir(path)

        # Optimizer settings.
        source_path = 'meshes/' + row['MESH']                   # Source mesh path.
        lr = row['LR']                                          # Optimizer learning rate.
        init_mode = 'zeros'                                     # Optimizer initialization mode.
        beam_have_load = True                                   # If beam load is computed or not.
        loss_type = row['LOSS']                                 # Which structural loss is computed.
        with_laplacian_smooth = True                            # If laplacian regularization is employed or not.
        with_normal_consistency = True                          # If normal consistency regularization is employed or not.
        laplsmooth_loss_perc = row['LAPLACIAN_PERC']            # Laplacian regularization percentual on structural loss.
        normcons_loss_perc = row['NORMCONS_PERC']               # Normal consistency regularization percentual on structural loss.
        n_iter = self.n_iter                                    # Number of iterations per experiment.
        plot = False                                            # If we want to show meshes during iterations or not.
        save = True                                             # If we want to save meshes during iterations or not.
        plot_save_interval = 50                                 # Iteration interval between mesh saves/plots.
        display_interval = -1                                   # Iteration interval between loss displays.
        save_label = row['MESH'][:-4]                           # Label of saved mesh.
        take_times = False                                      # If we want to see iteration and backward times or not.
        save_prefix = path + '/'                                # Path of current run saves.

        optimizer = LacconianOptimizer(source_path, lr, self.device, init_mode, beam_have_load, loss_type, with_laplacian_smooth, with_normal_consistency, laplsmooth_loss_perc, normcons_loss_perc)
        print('Optimizing (run ' + str(row['INDEX']+1) + ' of ' + str(self.no_experiments) + ' ) ...')
        optimizer.start(n_iter, plot, save, plot_save_interval, display_interval, save_label, take_times, save_prefix=save_prefix, wandb_run=run)
        

if __name__ == '__main__':
    parser = WandbLoggerOptions()
    options = parser.parse()
    wandb_logger = WandbLogger(options.device, options.project, options.n_iter)
    wandb_logger.start()

