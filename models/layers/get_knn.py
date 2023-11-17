import torch
from utils.utils import save_mesh, save_cloud

def get_knn(x, mesh, target_idx, k):
    topk = torch.topk(torch.norm(x[target_idx] - x, dim=1), k=k, largest=False).indices
    color = torch.tensor([0., 0., 1., 1.]).repeat(k, 1)
    color[0, 0] = 1.
    color[0, 2] = 0.

    filename = 'knn' + str(k) + '.ply'
    save_cloud(mesh.vertices[topk], filename, color)

    filename = 'mesh.ply'
    save_mesh(mesh, filename)
