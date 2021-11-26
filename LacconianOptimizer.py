import torch
from torch.optim import optimizer
from LacconianCalculus import LacconianCalculus
from models.layers.mesh import Mesh

class LacconianOptimizer:

    def __init__(self, file, lr=0.1, momentum=0.9, device='cpu'):
        self.mesh = Mesh(file=file, vertices_have_grad=True, device=device)
        self.device = device
        self.optimizer = torch.optim.SGD([self.mesh.vertices], lr=lr, momentum=momentum)
        self.lacconian_calculus = LacconianCalculus(device=device)

    def start(self, n_iter=1000):
        #Keeping non constrainted vertex mask.
        non_constraint_mask = self.mesh.vertex_is_constrainted.logical_not()

        for iteration in range(n_iter):
            #Initialize optimizer
            self.optimizer.zero_grad()

            loss = self.lacconian_calculus(self.mesh)
            print('Iteration: ', iteration, ' Loss: ', loss)

            loss.backward()
            #CONDITION: gradients belonging to non displaceable vertices are vanished.
            self.mesh.vertices.grad[non_constraint_mask, :] = 0
            self.optimizer.step()

lo = LacconianOptimizer('meshes/go.ply')
lo.start(n_iter=5)
