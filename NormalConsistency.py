import torch
from LacconianCalculus import LacconianCalculus

class NormalConsistency:

    def __init__(self, mesh, device, boundary_reg):
        self.initial_mesh = mesh
        self.device = torch.device(device)
        self.boundary_reg = boundary_reg
        self.make_adjacency_matrices()

        # Initial >=30° angles are not smoothed.
        consistency = self.compute_consistency(mesh, 1.)
        self.consistency_mask = (consistency >=  1 - torch.cos(torch.tensor(torch.pi / 6), device=self.device))

    def make_adjacency_matrices(self):
        edge_list = []
        boundary_free_vertex_list = []

        edge_is_on_boundary = torch.zeros(self.initial_mesh.edges.shape[0], dtype=torch.bool, device=self.device)

        # Building edge flaps face matrix.
        for edge in range(self.initial_mesh.edges.shape[0]):
            faces = torch.any(self.initial_mesh.edges_per_face == edge, dim=1).nonzero().flatten()
            # Boundary edges are excluded.
            if faces.shape[0] == 2:
                edge_list.append(faces)
            else:
                edge_is_on_boundary[edge] = True
            
        self.faces_per_edge = torch.stack(edge_list, dim=0).long()

        # Building boundary vertex edge matrix.
        if self.boundary_reg:
            for vertex in range(self.initial_mesh.vertices.shape[0]):
                if self.initial_mesh.vertex_is_on_boundary[vertex] and not self.initial_mesh.vertex_is_red[vertex] and not self.initial_mesh.vertex_is_blue[vertex]:
                    edges = (torch.any(self.initial_mesh.edges == vertex, dim=1) * edge_is_on_boundary).nonzero().flatten()
                    boundary_free_vertex_list.append(edges)
            
            if len(boundary_free_vertex_list) != 0:
                self.edges_per_free_boundary_vertex = torch.stack(boundary_free_vertex_list, dim=0).long()
            else:
                self.boundary_reg = False

    def __call__(self, mesh):
        consistency = self.compute_consistency(mesh, 0.5)
        loss = torch.mean(consistency[self.consistency_mask])
        return loss

    def compute_consistency(self, mesh, weight):
        if self.boundary_reg:
            n0 = torch.cat((mesh.face_normals[self.faces_per_edge[:, 0]], weight * mesh.cross_dirs[self.edges_per_free_boundary_vertex[:, 0]]), dim=0)
            n1 = torch.cat((mesh.face_normals[self.faces_per_edge[:, 1]], weight * mesh.cross_dirs[self.edges_per_free_boundary_vertex[:, 1]]), dim=0)
        else:
            n0 = mesh.face_normals[self.faces_per_edge[:, 0]]
            n1 = mesh.face_normals[self.faces_per_edge[:, 1]]

        consistency = 1 - torch.cosine_similarity(n0, n1, dim=1)
        return consistency
