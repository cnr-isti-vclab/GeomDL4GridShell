import torch
from LacconianCalculus import LacconianCalculus
from models.layers.mesh import Mesh
from options.optimizer_options import OptimizerOptions
from utils import save_mesh

class LacconianOptimizer:

    def __init__(self, file, lr, device):
        self.mesh = Mesh(file=file, device=device)
        self.lacconian_calculus = LacconianCalculus(device=device, mesh=self.mesh)
        self.device = torch.device(device)

        #Building optimizer.
        self.lc = LacconianCalculus(file=file)
        self.displacements = -self.lc.vertex_deformations[self.lc.non_constrained_vertices, :3]
        self.displacements.requires_grad_()
        # self.displacements = torch.distributions.Uniform(0,1e-2).sample((len(self.mesh.vertices[self.lacconian_calculus.non_constrained_vertices]), 3))
        # self.displacements = torch.distributions.Normal(0,1e-2).sample((len(self.mesh.vertices[self.lacconian_calculus.non_constrained_vertices]), 3))
        # self.displacements.to(device).requires_grad_()
        # self.displacements = torch.zeros(len(self.mesh.vertices[self.lacconian_calculus.non_constrained_vertices]), 3, device=self.device, requires_grad=True)
        self.optimizer = torch.optim.Adam([ self.displacements ], lr=lr)

    def start(self, n_iter, plot, save, interval, savelabel):
        for iteration in range(n_iter):
            #Putting grads to None.
            self.optimizer.zero_grad(set_to_none=True)

            #Summing displacements to mesh vertices.
            self.mesh.vertices[self.lacconian_calculus.non_constrained_vertices, :] += self.displacements

            # Plotting/saving.
            if iteration % interval == 0:
                if plot:
                    self.plot_grid_shell()
                if save:
                    filename = savelabel + '_' + str(iteration) + '.ply'
                    save_mesh(self.mesh, filename)

            loss = self.lacconian_calculus()
            print('Iteration: ', iteration, ' Loss: ', loss)

            #Computing gradients and updating optimizer
            loss.backward()
            self.optimizer.step()

            #Deleting grad history in all re-usable attributes.
            self.lacconian_calculus.clean_attributes()

    def plot_grid_shell(self):
        if not hasattr(self.lacconian_calculus, 'vertex_deformations'):
            self.mesh.plot_mesh()
        else:
            colors = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
            self.mesh.plot_mesh(colors=colors)


parser = OptimizerOptions()
options = parser.parse()
lo = LacconianOptimizer(options.path, options.lr, options.device)
lo.start(options.n_iter, options.plot, options.save, options.interval, options.savelabel)
