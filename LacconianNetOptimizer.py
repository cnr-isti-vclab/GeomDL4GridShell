import torch
import time
from LacconianCalculus import LacconianCalculus
from models.layers.featured_mesh import FeaturedMesh
from models.networks import DGCNNDisplacerNet
from options.net_optimizer_options import NetOptimizerOptions
from utils import save_mesh, export_vector


class LacconianNetOptimizer:

    def __init__(self, file, lr, momentum, device, beam_have_load, loss_type, no_knn):
        self.initial_mesh = FeaturedMesh(file=file, device=device)
        self.beam_have_load = beam_have_load
        self.device = device
        self.loss_type = loss_type
        self.lacconian_calculus = LacconianCalculus(device=device, mesh=self.initial_mesh, beam_have_load=beam_have_load)
        self.lr = lr
        self.momentum = momentum

        # Setting 10 decimal digits tensor display.
        torch.set_printoptions(precision=10)

        self.device = torch.device(device)

        self.make_optimizer(no_knn)

    def make_optimizer(self, no_knn):
        # Computing initial_mesh input features.
        self.initial_mesh.compute_mesh_input_features()

        # Initializing net model.
        self.model = DGCNNDisplacerNet(self.initial_mesh.input_features.shape[1], no_knn).to(self.device)

        # Initializing model weights.
        # self.model.apply(self.model.weight_init)

        # Building optimizer.
        # self.optimizer = torch.optim.Adam([ self.model.parameters ], lr=lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Building lr decay scheduler.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=15, verbose=True)

    def optimize(self, n_iter, save, save_interval, display_interval, save_label, take_times, save_prefix=''):
        # Initializing best loss.
        best_loss = torch.tensor(float('inf'), device=self.device)

        # Saving initial mesh with structural data.
        if save:
            filename = save_prefix + '[START]' + save_label + '.ply'
            quality = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
            save_mesh(self.initial_mesh, filename, v_quality=quality.unsqueeze(1))
            export_vector(self.initial_mesh.edges, 'edges.csv')
            export_vector(quality, save_prefix + '[START]load_' + save_label + '.csv')
            export_vector(self.lacconian_calculus.beam_energy, save_prefix + '[START]energy_' + save_label + '.csv')

        for current_iteration in range(n_iter):
            iter_start = time.time()

            # Putting grads to None.
            self.optimizer.zero_grad(set_to_none=True)

            # Computing mesh displacements via DGCNNNDisplacerNet.
            displacements = self.model(self.initial_mesh.input_features)
            offset = torch.zeros(self.initial_mesh.vertices.shape[0], 3, device=self.device)
            offset[self.lacconian_calculus.non_constrained_vertices, :] = displacements[self.lacconian_calculus.non_constrained_vertices]

            # Generating current iteration displaced mesh.
            iteration_mesh = self.initial_mesh.update_verts(offset)

            # Computing structural loss.
            structural_loss = self.lacconian_calculus(iteration_mesh, self.loss_type)

            # Computing boundary penalty term.
            constrained_vertices = torch.logical_not(self.lacconian_calculus.non_constrained_vertices)
            boundary_penalty = torch.mean(torch.norm(displacements[constrained_vertices], dim=1))
            with torch.no_grad():
                penalty_scale = 0.15 * structural_loss / boundary_penalty

            # Summing loss components.
            loss = structural_loss + penalty_scale * boundary_penalty

            # Saving current iteration mesh if requested.
            if current_iteration % save_interval == 0:
                if save:
                    filename = save_prefix + save_label + '_' + str(current_iteration) + '.ply'
                    quality = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
                    save_mesh(iteration_mesh, filename, v_quality=quality.unsqueeze(1))

            # Displaying loss if requested.
            if display_interval != -1 and current_iteration % display_interval == 0:
                print('*********** Iteration: ', current_iteration, ' Loss: ', loss, '***********')

            # Keeping data if loss is best.
            if loss < best_loss:
                best_loss = loss
                best_iteration = current_iteration

                if save:
                    best_mesh = iteration_mesh
                    best_displacements = quality
                    best_energy = self.lacconian_calculus.beam_energy

            # Computing gradients and updating optimizer
            back_start = time.time()
            loss.backward()
            back_end = time.time()
            self.optimizer.step()
            # self.scheduler.step(loss)

            # Deleting grad history in involved tensors.
            self.lacconian_calculus.clean_attributes()

            iter_end = time.time()

            # Displaying times if requested.
            if take_times:
                print('Iteration time: ' + str(iter_end - iter_start))
                print('Backward time: ' + str(back_end - back_start))
        
        # Saving best mesh, if mesh saving is enabled.
        if save and n_iter > 0:
            filename = save_prefix + '[BEST]' + save_label + '_' + str(best_iteration) + '.ply'
            save_mesh(best_mesh, filename, v_quality=best_displacements.unsqueeze(1))
            export_vector(best_displacements, '[BEST]load_' + save_label + str(best_iteration) + '.csv')
            export_vector(best_energy, '[BEST]energy_' + save_label + str(best_iteration) + '.csv')


if __name__ == '__main__':
    parser = NetOptimizerOptions()
    options = parser.parse()
    lo = LacconianNetOptimizer(options.path, options.lr, options.momentum, options.device, options.beam_have_load, options.loss_type, options.no_knn)
    lo.optimize(options.n_iter, options.save, options.save_interval, options.display_interval, options.save_label, options.take_times)

