import torch
import time
from LacconianCalculus import LacconianCalculus
from LaplacianSmoothing import LaplacianSmoothing
from NormalConsistency import NormalConsistency
from models.layers.mesh import Mesh
from options.optimizer_options import OptimizerOptions
from utils import save_mesh


class LacconianOptimizer:

    def __init__(self, file, lr, device, init_mode, beam_have_load, loss_type, with_laplacian_smooth, with_normal_consistency, laplsmooth_loss_perc, normcons_loss_perc):
        self.mesh = Mesh(file=file, device=device)
        self.loss_type = loss_type
        self.lacconian_calculus = LacconianCalculus(device=device, mesh=self.mesh, beam_have_load=beam_have_load)

        # Taking useful initial data.
        loss_0 = self.lacconian_calculus(self.loss_type)
        self.vertices_0 = torch.clone(self.mesh.vertices)
        eps = 1e-3

        # Finding laplacian smoothing loss scaling factor according to input percentage.
        if with_laplacian_smooth:
            self.laplacian_smoothing = LaplacianSmoothing(self.mesh, device)
            if laplsmooth_loss_perc == -1:
                self.laplsmooth_scaling_factor = 1
            else:
                laplacian_smooth_0 = self.laplacian_smoothing(self.mesh)
                self.laplsmooth_scaling_factor = laplsmooth_loss_perc * loss_0 / max(laplacian_smooth_0, eps)

        # Finding normal consistency loss scaling factor according to input percentage.
        if with_normal_consistency:
            self.normal_consistency = NormalConsistency(self.mesh, device)
            if normcons_loss_perc == -1:
                self.normcons_scaling_factor = 1
            else:
                normal_consistency_0 = self.normal_consistency()
                self.normcons_scaling_factor = normcons_loss_perc * loss_0 / max(normal_consistency_0, eps)

        self.device = torch.device(device)

        # Initializing displacements.
        if init_mode == 'stress_aided':
            self.lc = LacconianCalculus(file=file, device=device, beam_have_load=beam_have_load)
            self.displacements = -self.lc.vertex_deformations[self.lc.non_constrained_vertices, :3]
            self.displacements.requires_grad = True
        elif init_mode == 'uniform':
            self.displacements = torch.distributions.Uniform(0,1e-6).sample((len(self.mesh.vertices[self.lacconian_calculus.non_constrained_vertices]), 3))
            self.displacements = self.displacements.to(device)
            self.displacements.requires_grad = True
        elif init_mode == 'normal':
            self.displacements = torch.distributions.Normal(0,1e-6).sample((len(self.mesh.vertices[self.lacconian_calculus.non_constrained_vertices]), 3))
            self.displacements = self.displacements.to(device)
            self.displacements.requires_grad = True
        elif init_mode == 'zeros':
            self.displacements = torch.zeros(len(self.mesh.vertices[self.lacconian_calculus.non_constrained_vertices]), 3, device=self.device, requires_grad=True)

        # Building optimizer.
        self.optimizer = torch.optim.Adam([ self.displacements ], lr=lr)

    def start(self, n_iter, plot, save, plot_save_interval, display_interval, save_label, take_times, save_prefix='', wandb_run=None):
        # Initializing best loss.
        best_loss = torch.tensor(float('inf'), device=self.device)

        for iteration in range(n_iter):
            iter_start = time.time()

            # Putting grads to None.
            self.optimizer.zero_grad(set_to_none=True)

            # Initializing wandb log dictionary.
            log_dict = {}

            # Summing displacements to mesh vertices.
            self.mesh.vertices[self.lacconian_calculus.non_constrained_vertices, :] += self.displacements

            # VERTICES CHANGED: making on mesh loss-shared computations again.
            self.mesh.make_on_mesh_shared_computations()

            # Keeping max vertex displacement norm per iteration.
            max_displacement_norm = torch.max(torch.norm(self.mesh.vertices - self.vertices_0, p=2, dim=1))
            log_dict['max_displacement_norm'] = max_displacement_norm

            # Plotting/saving.
            if iteration % plot_save_interval == 0:
                if plot:
                    colors = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
                    self.mesh.plot_mesh(colors=colors)

                if save:
                    filename = save_prefix + save_label + '_' + str(iteration) + '.ply'
                    quality = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
                    save_mesh(self.mesh.vertices, self.mesh.faces, self.mesh.vertex_is_red, self.mesh.vertex_is_blue, filename, v_quality=quality.unsqueeze(1))

            # Computing loss by summing components.
            loss = 0

            # Lacconian loss.
            structural_loss = self.lacconian_calculus(self.loss_type)
            loss += structural_loss
            log_dict['structural_loss'] = structural_loss

            # Keeping max stress deformation.
            max_deformation_norm = torch.max(torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1))
            log_dict['max_load_deformation_norm'] = max_deformation_norm

            # Laplacian smoothing.
            if hasattr(self, 'laplacian_smoothing'):
                ls = self.laplacian_smoothing(self.mesh)
                log_dict['laplacian_smoothing'] = ls
                loss += self.laplsmooth_scaling_factor * ls

            # Normal consistency.
            if hasattr(self, 'normal_consistency'):
                nc = self.normal_consistency()
                log_dict['normal_consistency'] = nc
                loss += self.normcons_scaling_factor * nc

            log_dict['loss'] = loss

            # Displaying loss if requested.
            if display_interval != -1 and iteration % display_interval == 0:
                print('*********** Iteration: ', iteration, ' Loss: ', loss, '***********')

            # Keeping data if loss is best.
            if loss < best_loss:
                best_loss = loss
                best_iteration = iteration

                # Saving losses at best iteration.
                structural_loss_at_best_iteration = structural_loss
                max_displacement_norm_at_best_iteration = max_displacement_norm
                max_deformation_norm_at_best_iteration = max_deformation_norm
                if hasattr(self, 'laplacian_smoothing'):
                    laplacian_smoothing_at_best_iteration = ls
                if hasattr(self, 'normal_consistency'):
                    normal_consistency_at_best_iteration = nc

                # CAUTION: we do not store best_mesh_faces as meshing do not change.
                if save:
                    best_mesh_vertices = torch.clone(self.mesh.vertices)
                    best_quality = quality

            # Logging on wandb, if requested.
            if wandb_run is not None:
                wandb_run.log(log_dict)
                wandb_run.summary['best_iteration'] = best_iteration
                wandb_run.summary['structural_loss_at_best_iteration'] = structural_loss_at_best_iteration
                wandb_run.summary['max_displacement_norm_at_best_iteration'] = max_displacement_norm_at_best_iteration
                wandb_run.summary['max_load_deformation_norm_at_best_iteration'] = max_deformation_norm_at_best_iteration
                wandb_run.summary['laplacian_smoothing_at_best_iteration'] = laplacian_smoothing_at_best_iteration
                wandb_run.summary['normal_consistency_at_best_iteration'] = normal_consistency_at_best_iteration

            # Computing gradients and updating optimizer
            back_start = time.time()
            loss.backward()
            back_end = time.time()
            self.optimizer.step()

            # Deleting grad history in all re-usable attributes.
            self.lacconian_calculus.clean_attributes()

            iter_end = time.time()

            # Displaying times if requested.
            if take_times:
                print('Iteration time: ' + str(iter_end - iter_start))
                print('Backward time: ' + str(back_end - back_start))

        # Saving best mesh, if mesh saving is enabled.
        if save and n_iter > 0:
            filename = save_prefix + '[BEST]' + save_label + '_' + str(best_iteration) + '.ply'
            save_mesh(best_mesh_vertices, self.mesh.faces, self.mesh.vertex_is_red, self.mesh.vertex_is_blue, filename, v_quality=best_quality.unsqueeze(1))

if __name__ == '__main__':
    parser = OptimizerOptions()
    options = parser.parse()
    lo = LacconianOptimizer(options.path, options.lr, options.device, options.init_mode, options.beam_have_load, options.loss_type, options.with_laplacian_smooth, options.with_normal_consistency, options.laplsmooth_loss_perc, options.normcons_loss_perc)
    lo.start(options.n_iter, options.plot, options.save, options.plot_save_interval, options.display_interval, options.save_label, options.take_times)
