import torch
import numpy as np
from utils import load_mesh, edge_connectivity
from torch.nn.functional import normalize
import copy

class Mesh:

    def __init__(self, file=None, vertices=None, faces=None, device='cpu'):
        self.device = torch.device(device)
        if file is not None and vertices is None and faces is None:
            self.vertices, self.faces, self.vertex_is_red, self.vertex_is_on_boundary, self.bb_diagonal = load_mesh(file)
        elif vertices is not None and faces is not None:
            self.vertices, self.faces = vertices.cpu().numpy(), faces.cpu().numpy()
            self.vertex_is_red = self.vertex_is_on_boundary = None
        else:
            raise ValueError('Wrong mesh data provided: file xor vertices + faces.')
        edges, edges_per_face = edge_connectivity(self.faces)
        self.edges = torch.from_numpy(edges)
        self.edges_per_face = torch.from_numpy(edges_per_face)
        if type(self.vertices) is np.ndarray:
            self.vertices = torch.from_numpy(self.vertices)
        if type(self.faces) is np.ndarray:
            self.faces = torch.from_numpy(self.faces)
        if type(self.vertex_is_red) is np.ndarray:
            self.vertex_is_red = torch.from_numpy(self.vertex_is_red)
        if type(self.vertex_is_on_boundary) is np.ndarray:
            self.vertex_is_on_boundary = torch.from_numpy(self.vertex_is_on_boundary)
        self.vertices = self.vertices.to(device)
        self.edges = self.edges.long().to(device)
        self.faces = self.faces.long().to(device)
        self.edges_per_face = self.edges_per_face.long().to(device)
        if self.vertex_is_red is not None:
            self.vertex_is_red = self.vertex_is_red.to(device)
        if self.vertex_is_on_boundary is not None:
            self.vertex_is_on_boundary = self.vertex_is_on_boundary.to(device)
        self.make_on_mesh_shared_computations()

    def compute_edge_lengths_and_directions(self):
        # Edge directions are computed by endpoints difference.
        # Please note: self.vertices[self.edges] is a (#edges, 2, 3) tensor aggregating per edge endpoint coordinates.
        edge_verts = self.vertices[self.edges]
        edge_directions = edge_verts[:, 1, :] - edge_verts[:, 0, :]

        # Computing edge lengths.
        self.edge_lengths = torch.norm(edge_directions, p=2, dim=1)

        # Normalizing edge directions.
        self.edge_directions = edge_directions / self.edge_lengths.unsqueeze(1)

    def compute_edge_normals(self):
        # Getting face normals and areas.
        self.face_areas, self.face_normals = self.face_areas_normals(self.vertices, self.faces, normalize=False)

        ############################################################################################################
        # Computing edge normals by weighting normals from (at most 2) incident faces.
        # Some details: vec.scatter_add(0, idx, src) with vec, idx, src 1d tensors, add at vec positions specified
        # by idx corresponding src values.
        self.edge_normals = torch.zeros(self.edges.shape[0], 3, device=self.device)

        self.edge_normals[:, 0].scatter_add_(0, self.edges_per_face.flatten(), torch.stack([self.face_normals[:, 0]] * 3, dim=1).flatten())
        self.edge_normals[:, 1].scatter_add_(0, self.edges_per_face.flatten(), torch.stack([self.face_normals[:, 1]] * 3, dim=1).flatten())
        self.edge_normals[:, 2].scatter_add_(0, self.edges_per_face.flatten(), torch.stack([self.face_normals[:, 2]] * 3, dim=1).flatten())

        # Applying final l2-normalization.
        self.edge_normals = normalize(self.edge_normals, p=2, dim=1)

    # Makes all mesh computations needed and shared between loss classes: face areas and normals, edge lengths and
    # directions, edge normals. 
    def make_on_mesh_shared_computations(self):
        self.compute_edge_lengths_and_directions()
        self.compute_edge_normals()
        self.cross_dirs = torch.cross(self.edge_directions, self.edge_normals)

    @staticmethod
    def face_areas_normals(vs, faces, normalize=True):
        if type(vs) is not torch.Tensor:
            vs = torch.from_numpy(vs)
        if type(faces) is not torch.Tensor:
            faces = torch.from_numpy(faces).long()

        # Again, vs[faces] aggregates face vertices coordinates in a (#faces, 3, 3) tensor.
        face_verts = vs[faces]

        face_normals = torch.cross(face_verts[:, 1, :] - face_verts[:, 0, :],
                                   face_verts[:, 2, :] - face_verts[:, 1, :])
        face_areas = torch.norm(face_normals, dim=1)

        if normalize:
            face_normals = face_normals / face_areas.unsqueeze(1)
        face_areas = torch.mul(0.5, face_areas)

        return face_areas, face_normals

    def update_verts(self, verts):
        """
        update verts positions only, same connectivity
        """
        new_mesh = copy.deepcopy(self)
        new_mesh.vertices += verts
        new_mesh.make_on_mesh_shared_computations()
        
        return new_mesh