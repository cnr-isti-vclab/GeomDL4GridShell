import torch
from LacconianCalculus import LacconianCalculus
from models.layers.mesh import Mesh
from options.optimizer_options import OptimizerOptions
from utils import save_mesh

class LacconianOptimizer:

    def __init__(self, file, lr, device):
        self.mesh = Mesh(file=file, vertices_have_grad=True, device=device)
        self.lacconian_calculus = LacconianCalculus(device=device)
        self.device = torch.device(device)

        #Keeping non constrainted vertex mask.
        self.non_constraint_mask = self.mesh.vertex_is_constrainted.logical_not()

        #Building optimizer.
        self.displacements = torch.zeros(int(torch.sum(self.non_constraint_mask)), 3, requires_grad=True, device=self.device)
        self.optimizer = torch.optim.Adam([ self.displacements ], lr=lr)

    def start(self, n_iter, plot, save, interval, savelabel):
        for iteration in range(n_iter):
            #Putting grads to None.
            self.optimizer.zero_grad(set_to_none=True)

            #Summing displacements to mesh vertices.
            self.mesh.vertices[self.non_constraint_mask, :] += self.displacements

            # Plotting/saving.
            if iteration % interval == 0:
                if plot:
                    self.plot_grid_shell()
                if save:
                    filename = savelabel + '_' + str(iteration) + '.ply'
                    save_mesh(self.mesh, filename)

            loss = self.lacconian_calculus(self.mesh)
            print('Iteration: ', iteration, ' Loss: ', loss)

            #Computing gradients and updating optimizer
            loss.backward()
            self.optimizer.step()

            #Deleting grad history in mesh.vertices
            self.mesh.vertices.detach_()

    def plot_grid_shell(self):
        if self.lacconian_calculus.vertex_deformations is None:
            self.mesh.plot_mesh()
        else:
            colors = torch.norm(self.lacconian_calculus.vertex_deformations[:, :3], p=2, dim=1)
            self.mesh.plot_mesh(colors=colors)


parser = OptimizerOptions()
options = parser.parse()
lo = LacconianOptimizer(options.path, options.lr, options.device)
lo.start(options.n_iter, options.plot, options.save, options.interval, options.savelabel)
