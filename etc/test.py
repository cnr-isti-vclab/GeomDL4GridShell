import torch
from models.layers.featured_mesh import FeaturedMesh
from utils import save_cloud, save_mesh

target_idx = 60
k = 120

mesh = FeaturedMesh(file='meshes/station.ply')
mesh.compute_mesh_input_features()
x = mesh.input_features[:, 8:12]

topk = torch.topk(torch.norm(x[target_idx] - x, dim=1), k=k, largest=False).indices
color = torch.tensor([0., 0., 1., 1.]).repeat(k, 1)
color[0, 0] = 1.
color[0, 2] = 0.

filename = 'knn' + str(k) + '.ply'
save_cloud(mesh.vertices[topk], filename, color)

filename = 'mesh.ply'
save_mesh(mesh, filename)