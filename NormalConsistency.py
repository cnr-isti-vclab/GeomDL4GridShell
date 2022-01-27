'''
Adapted from pytorch3d.loss.mesh_normal_consistency.py
Copyright (c) Facebook, Inc. and its affiliates.
'''
import torch
from LacconianCalculus import LacconianCalculus

class NormalConsistency:
    # IMPORTANT: to be reviewed in case of remeshing.

    def __init__(self, mesh, device, relative=False):
        self.mesh = mesh
        self.relative = relative                    # If True, we compute a relative version of normal consistency reg.
        self.device = torch.device(device)
        self.make_faces_per_edge()

        # Computing initial consistency.
        if self.relative:
            n0 = self.mesh.face_normals[self.faces_per_edge[:, 0]]
            n1 = self.mesh.face_normals[self.faces_per_edge[:, 1]]
            self.consistency_0 = 1 - torch.cosine_similarity(n0, n1, dim=1)

    def make_faces_per_edge(self):
        edge_list = []

        for edge in range(self.mesh.edges.shape[0]):
            faces = torch.any(self.mesh.edges_per_face == edge, dim=1).nonzero().flatten()

            if faces.shape[0] == 2:
                # Boundary edges are excluded.
                edge_list.append(faces)
            
        self.faces_per_edge = torch.stack(edge_list, dim=0).long()

    def __call__(self):
        n0 = self.mesh.face_normals[self.faces_per_edge[:, 0]]
        n1 = self.mesh.face_normals[self.faces_per_edge[:, 1]]
        consistency = 1 - torch.cosine_similarity(n0, n1, dim=1)

        if self.relative:
            loss = torch.mean((consistency - self.consistency_0) / self.consistency_0)
            print(loss)
        else:
            loss = torch.mean(consistency)

        return loss
