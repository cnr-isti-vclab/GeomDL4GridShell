import torch
from LacconianCalculus import LacconianCalculus

class NormalConsistency:

    def __init__(self, mesh, device):
        self.initial_mesh = mesh
        self.device = torch.device(device)
        self.make_faces_per_edge()

    def make_faces_per_edge(self):
        edge_list = []

        for edge in range(self.initial_mesh.edges.shape[0]):
            faces = torch.any(self.initial_mesh.edges_per_face == edge, dim=1).nonzero().flatten()

            # Boundary edges are excluded.
            if faces.shape[0] == 2:
                edge_list.append(faces)
            
        self.faces_per_edge = torch.stack(edge_list, dim=0).long()

    def __call__(self, mesh):
        n0 = mesh.face_normals[self.faces_per_edge[:, 0]]
        n1 = mesh.face_normals[self.faces_per_edge[:, 1]]
        consistency = 1 - torch.cosine_similarity(n0, n1, dim=1)
        
        loss = torch.mean(consistency)

        return loss
