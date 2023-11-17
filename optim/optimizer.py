import torch
import time
from optim.structural_calculus import StructuralCalculus
from optim.laplacian_smoothing import LaplacianSmoothing
from optim.normal_consistency import NormalConsistency
from models.layers.mesh import Mesh
from options.optimizer_options import OptimizerOptions
from utils.utils import save_mesh, isotrophic_remesh, save_cloud


class StructuralOptimizer:

    def __init__(self, file, lr, momentum, device, init_mode, loss_type, with_laplacian_smooth, with_normal_consistency, with_var_face_areas, laplsmooth_loss_perc, normcons_loss_perc, varfaceareas_loss_perc, boundary_reg):
        self.initial_mesh = Mesh(file=file, device=device)
        self.device = device
        self.loss_type = loss_type
        self.Structural_calculus = StructuralCalculus(device=device, mesh=self.initial_mesh)
        self.init_mode = init_mode
        self.boundary_reg = boundary_reg
        self.lr = lr
        self.momentum = momentum

        # Taking useful initial data.
        loss_0 = self.Structural_calculus(self.initial_mesh, self.loss_type)
        eps = 1e-3

        # Setting 10 decimal digits tensor display.
        torch.set_printoptions(precision=10)

        # Finding laplacian smoothing loss scaling factor according to input percentage.
        if with_laplacian_smooth:
            self.laplacian_smoothing = LaplacianSmoothing(device)
            if laplsmooth_loss_perc == -1:
                self.laplsmooth_scaling_factor = 1
            else:
                laplacian_smooth_0 = self.laplacian_smoothing(self.initial_mesh)
                self.laplsmooth_scaling_factor = laplsmooth_loss_perc * loss_0 / max(laplacian_smooth_0, eps)

        # Finding normal consistency loss scaling factor according to input percentage.
        if with_normal_consistency:
            self.normal_consistency = NormalConsistency(self.initial_mesh, device, boundary_reg)
            if normcons_loss_perc == -1:
                self.normcons_scaling_factor = 1
            else:
                normal_consistency_0 = self.normal_consistency(self.initial_mesh)
                self.normcons_scaling_factor = normcons_loss_perc * loss_0 / max(normal_consistency_0, eps)

        # Finding var face areas scaling factor according to input percentage.
        if with_var_face_areas:
            if varfaceareas_loss_perc == -1:
                self.varareas_scaling_factor = 1
            else:
                var_areas_0 = torch.var(self.initial_mesh.face_areas)
                self.varareas_scaling_factor = varfaceareas_loss_perc * loss_0 / max(var_areas_0, eps)

        self.device = torch.device(device)

        self.make_optimizer()

    def make_optimizer(self):
        # Initializing displacements.
        if self.init_mode == 'uniform':
            self.displacements = torch.distributions.Uniform(0,1e-4).sample((len(self.initial_mesh.vertices[self.Structural_calculus.non_constrained_vertices]), 3))
            self.displacements = self.displacements.to(self.device)
            self.displacements.requires_grad = True
        elif self.init_mode == 'normal':
            self.displacements = torch.distributions.Normal(0,1e-4).sample((len(self.initial_mesh.vertices[self.Structural_calculus.non_constrained_vertices]), 3))
            self.displacements = self.displacements.to(self.device)
            self.displacements.requires_grad = True
        elif self.init_mode == 'zeros':
            self.displacements = torch.zeros(len(self.initial_mesh.vertices[self.Structural_calculus.non_constrained_vertices]), 3, device=self.device, requires_grad=True)

        # Building optimizer.
        # self.optimizer = torch.optim.Adam([ self.displacements ], lr=lr)
        self.optimizer = torch.optim.SGD([ self.displacements ], lr=self.lr, momentum=self.momentum)

    def optimize(self, n_iter, save, save_interval, display_interval, save_label, take_times, with_remeshing, remeshing_interval, see_not_smoothed, save_prefix='', wandb_run=None):
        # Initializing best loss.
        best_loss = torch.tensor(float('inf'), device=self.device)

        # Saving initial mesh with structural data.
        if save:
            filename = save_prefix + save_label + '_start.ply'
            quality = torch.norm(self.Structural_calculus.vertex_deformations[:, :3], p=2, dim=1)
            save_mesh(self.initial_mesh, filename, v_quality=quality.unsqueeze(1))

        # Saving not smoothed point cloud, if requested
        if hasattr(self, 'normal_consistency') and see_not_smoothed and save:
            if self.normal_consistency.not_smoothed_points.shape[0] != 0:
                filename = save_prefix + '[cloud]' + save_label + '.ply'
                save_cloud(self.normal_consistency.not_smoothed_points, filename)

        for current_iteration in range(n_iter):
            # Executing remeshing if requested.
            if with_remeshing and current_iteration != 0 and current_iteration % remeshing_interval == 0:
                self.restart(iteration_mesh, current_iteration, save_label, save_prefix, display_interval)
                # Saving not smoothed point cloud, if requested. 
                if hasattr(self, 'normal_consistency') and see_not_smoothed and save:
                    if self.normal_consistency.not_smoothed_points.shape[0] != 0:
                        filename = save_prefix + '[remesh_cloud]' + save_label + '_' + str(current_iteration) + '.ply'
                        save_cloud(self.normal_consistency.not_smoothed_points, filename)

            iter_start = time.time()

            # Putting grads to None.
            self.optimizer.zero_grad(set_to_none=True)

            # Initializing wandb log dictionary.
            log_dict = {}

            # Generating current iteration displaced mesh.
            offset = torch.zeros(self.initial_mesh.vertices.shape, device=self.device)
            offset[self.Structural_calculus.non_constrained_vertices, :] = self.displacements
            iteration_mesh = self.initial_mesh.update_verts(offset)

            # Keeping max vertex displacement norm per iteration.
            max_displacement_norm = torch.max(torch.norm(offset, p=2, dim=1))
            log_dict['max_displacement_norm'] = max_displacement_norm

            # Computing loss by summing components.
            loss = 0

            # Structural loss.
            structural_loss = self.Structural_calculus(iteration_mesh, self.loss_type)
            loss += structural_loss
            log_dict['structural_loss'] = structural_loss

            # Keeping max stress deformation.
            max_deformation_norm = torch.max(torch.norm(self.Structural_calculus.vertex_deformations[:, :3], p=2, dim=1))
            log_dict['max_load_deformation_norm'] = max_deformation_norm

            # Laplacian smoothing.
            if hasattr(self, 'laplacian_smoothing'):
                ls = self.laplacian_smoothing(iteration_mesh)
                log_dict['laplacian_smoothing'] = ls
                loss += self.laplsmooth_scaling_factor * ls

            # Normal consistency.
            if hasattr(self, 'normal_consistency'):
                nc = self.normal_consistency(iteration_mesh)
                log_dict['normal_consistency'] = nc
                loss += self.normcons_scaling_factor * nc

            # Face area variance.
            if hasattr(self, 'varareas_scaling_factor'):
                var_areas = torch.var(iteration_mesh.face_areas)
                log_dict['var_face_areas'] = var_areas
                loss += self.varareas_scaling_factor * var_areas

            log_dict['loss'] = loss

            # Displaying loss if requested.
            if display_interval != -1 and current_iteration % display_interval == 0:
                print('*********** Iteration: ', current_iteration, ' Loss: ', loss, '***********')

            # Saving current iteration mesh if requested.
            if current_iteration % save_interval == 0:
                if save:
                    filename = save_prefix + save_label + '_' + str(current_iteration) + '.ply'
                    quality = torch.norm(self.Structural_calculus.vertex_deformations[:, :3], p=2, dim=1)
                    save_mesh(iteration_mesh, filename, v_quality=quality.unsqueeze(1))

            # Keeping data if loss is best.
            if loss < best_loss:
                best_loss = loss
                best_iteration = current_iteration

                # Saving losses at best iteration.
                structural_loss_at_best_iteration = structural_loss
                max_displacement_norm_at_best_iteration = max_displacement_norm
                max_deformation_norm_at_best_iteration = max_deformation_norm
                if hasattr(self, 'laplacian_smoothing'):
                    laplacian_smoothing_at_best_iteration = ls
                if hasattr(self, 'normal_consistency'):
                    normal_consistency_at_best_iteration = nc
                if hasattr(self, 'varareas_scaling_factor'):
                    var_face_areas_at_best_iteration = var_areas

                if save:
                    best_mesh = iteration_mesh
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
                wandb_run.summary['var_face_areas_at_best_iteration'] = var_face_areas_at_best_iteration

            # Computing gradients and updating optimizer
            back_start = time.time()
            loss.backward()
            back_end = time.time()
            self.optimizer.step()

            # Deleting grad history in involved tensors.
            self.Structural_calculus.clean_attributes()

            iter_end = time.time()

            # Displaying times if requested.
            if take_times:
                print('Iteration time: ' + str(iter_end - iter_start))
                print('Backward time: ' + str(back_end - back_start))

        # Saving best mesh, if mesh saving is enabled.
        if save and n_iter > 0:
            filename = save_prefix + '[BEST]' + save_label + '_' + str(best_iteration) + '.ply'
            save_mesh(best_mesh, filename, v_quality=best_quality.unsqueeze(1))

    def restart(self, iteration_mesh, current_iteration, save_label, save_prefix, display_interval):
        if display_interval != -1:
            print('Computing isotrophic remesh...')

        # Saving remesh.
        filename = save_prefix + '[remesh]' + save_label + '_' + str(current_iteration) + '.ply'
        target_length = float(torch.mean(iteration_mesh.edge_lengths))
        isotrophic_remesh(iteration_mesh, filename, target_length)

        # Reloading mesh.
        self.initial_mesh = Mesh(file=filename, device=self.device)
        self.Structural_calculus = StructuralCalculus(device=self.device, mesh=self.initial_mesh, beam_have_load=self.beam_have_load)
        if hasattr(self, 'normal_consistency'):
            self.normal_consistency = NormalConsistency(self.initial_mesh, self.device, self.boundary_reg)

        self.make_optimizer()
        

if __name__ == '__main__':
    parser = OptimizerOptions()
    options = parser.parse()
    lo = StructuralOptimizer(options.path, options.lr, options.momentum, options.device, options.init_mode, options.loss_type, options.with_laplacian_smooth, options.with_normal_consistency, options.with_var_face_areas, options.laplsmooth_loss_perc, options.normcons_loss_perc, options.varfaceareas_loss_perc, options.boundary_reg)
    lo.optimize(options.n_iter, options.save, options.save_interval, options.display_interval, options.save_label, options.take_times, options.with_remeshing, options.remeshing_interval, options.see_not_smoothed)
