import torch
from LacconianCalculus import LacconianCalculus
from models.layers.mesh import Mesh
from options.optimizer_options import OptimizerOptions
from utils import save_mesh


class LacconianOptimizer:

    def __init__(self, file, lr, device, init_mode, beam_have_load):
        self.mesh = Mesh(file=file, device=device)
        self.lacconian_calculus = LacconianCalculus(device=device, mesh=self.mesh, beam_have_load=beam_have_load)
        self.device = torch.device(device)

        # Initializing displacements.
        if init_mode == 'stress_aided':
            self.lc = LacconianCalculus(file=file, device=device, beam_have_load=beam_have_load)
            self.displacements = -self.lc.vertex_deformations[self.lc.non_constrained_vertices, :3]
            self.displacements.requires_grad_()
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

    def start(self, n_iter, plot, save, plot_save_interval, display_interval, save_label, loss_type):
        for iteration in range(n_iter):
            # Putting grads to None.
            self.optimizer.zero_grad(set_to_none=True)

            # Summing displacements to mesh vertices.
            self.mesh.vertices[self.lacconian_calculus.non_constrained_vertices, :] += self.displacements

            # Plotting/saving.
            if iteration % plot_save_interval == 0:
                if plot:
                    if not hasattr(self.lacconian_calculus, 'vertex_deformations'):
                        self.mesh.plot_mesh()
                    else:
                        colors = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
                        self.mesh.plot_mesh(colors=colors)

                if save:
                    filename = save_label + '_' + str(iteration) + '.ply'
                    if not hasattr(self.lacconian_calculus, 'vertex_deformations'):
                        save_mesh(self.mesh, filename)
                    else:
                        quality = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
                        save_mesh(self.mesh, filename, v_quality=quality.unsqueeze(1))

            loss = self.lacconian_calculus(loss_type)

            if iteration % display_interval == 0:
                print('Iteration: ', iteration, ' Loss: ', loss)

            # Computing gradients and updating optimizer
            loss.backward()
            self.optimizer.step()

            # Deleting grad history in all re-usable attributes.
            self.lacconian_calculus.clean_attributes()

parser = OptimizerOptions()
options = parser.parse()
lo = LacconianOptimizer(options.path, options.lr, options.device, options.init_mode, options.beam_have_load)
lo.start(options.n_iter, options.plot, options.save, options.plot_save_interval, options.display_interval, options.save_label, options.loss_type)
