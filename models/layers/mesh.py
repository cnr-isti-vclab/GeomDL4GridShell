##################################################################################################
# CHANGES:
# -- new parameters Mesh.vertex_is_constrained, Mesh.edges.
# -- utils.load_obj is replaced by utils.load_mesh;
# -- optional parameter normalize added in face_areas_normals method;
# -- Mesh.vs renamed to Mesh.vertices;
# -- Mesh.normals renamed to Mesh.face_normals;
# -- Mesh.area renamed to Mesh.face_areas.

import torch
import numpy as np
# from queue import Queue
from utils import load_mesh, plot_mesh, edge_connectivity
from torch.nn.functional import normalize
# import copy
# from pathlib import Path
# import pickle
# from pytorch3d.ops.knn import knn_gather, knn_points


class Mesh:

    def __init__(self, file, hold_history=False, vertices=None, faces=None, device='cpu', gfmm=True):
        if file is None:
            return
        # self.filename = Path(file)
        self.vertices = self.v_mask = self.edge_areas = None
        self.edges = self.gemm_edges = self.sides = None
        self.device = torch.device(device)
        if vertices is not None and faces is not None:
            self.vertices, self.faces = vertices.cpu().numpy(), faces.cpu().numpy()
            # self.scale, self.translations = 1.0, np.zeros(3,)
        else:
            self.vertices, self.faces, self.vertex_is_red, self.vertex_is_blue = load_mesh(file)
            # self.normalize_unit_bb()
        # self.vs_in = copy.deepcopy(self.vertices)
        # self.v_mask = np.ones(len(self.vertices), dtype=bool)
        # self.build_gemm()
        # self.history_data = None
        # if hold_history:
        #     self.init_history()
        # if gfmm:
        #     self.gfmm = self.build_gfmm() TODO get rid of this DS
        # else:
        #     self.gfmm = None
        edges, edges_per_face = edge_connectivity(self.faces)
        self.edges = torch.from_numpy(edges)
        self.edges_per_face = torch.from_numpy(edges_per_face)
        if type(self.vertices) is np.ndarray:
            self.vertices = torch.from_numpy(self.vertices)
        if type(self.faces) is np.ndarray:
            self.faces = torch.from_numpy(self.faces)
        if type(self.vertex_is_red) is np.ndarray:
            self.vertex_is_red = torch.from_numpy(self.vertex_is_red)
        if type(self.vertex_is_blue) is np.ndarray:
            self.vertex_is_blue = torch.from_numpy(self.vertex_is_blue)
        self.vertices = self.vertices.to(device)
        self.edges = self.edges.long().to(device)
        self.faces = self.faces.long().to(device)
        self.edges_per_face = self.edges_per_face.long().to(device)
        self.vertex_is_red = self.vertex_is_red.to(device)
        self.vertex_is_blue = self.vertex_is_blue.to(device)
        # self.face_areas, self.face_normals = self.face_areas_normals(self.vertices, self.faces)

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

        # Getting face incidence mask per vertex.
        # CAUTION: we are re-using same mask as mesh topology don't change.
        if not hasattr(self, 'vertex_face_mask'):
            self.vertex_face_mask, self.vertex_edge_mask, self.edge_face_mask = self.make_incidence_masks()

        ############################################################################################################
        # Computing edge normals by weighting normals from incident faces.
        # Some details: we work on (#axes, #edges, #faces) tensor. Along dim 0 there are 3 directional batches,
        # along dim 1 we move towards all possible edges and dim 2 is related to faces.
        reshaped_face_normals = self.face_normals.T.view(3, 1, self.faces.shape[0])
        
        # expand(shape) repeat tensor elements to reach target shape: -1 means that dim is preserved.
        expanded_face_normals = reshaped_face_normals.expand(-1, self.edges.shape[0], -1)

        # Vertex-face incidence mask is applied to each one of three batches.
        masked_face_normals = torch.mul(expanded_face_normals, self.edge_face_mask)

        # Edge normals are computed by summing masked (#axes, #edges, #faces) tensor along face dimension (dim=2).
        edge_normals = torch.sum(masked_face_normals, dim=2)
        self.edge_normals = normalize(edge_normals.T, p=2, dim=1)
    
    # This method builds boolean tensors giving incident faces/edges per vertex, faces per edge.
    def make_incidence_masks(self):
        vf_mask = torch.zeros(self.vertices.shape[0], self.faces.shape[0], dtype=torch.bool, device=self.device)
        ve_mask = torch.zeros(self.vertices.shape[0], self.edges.shape[0], dtype=torch.bool, device=self.device)
        ef_mask = torch.zeros(self.edges.shape[0], self.faces.shape[0], dtype=torch.bool, device=self.device)

        # Building per vertex masks.
        for idx, _ in enumerate(self.vertices):
            vf_mask[idx, :] = torch.any(torch.where(self.faces == idx, True, False), axis=1)
            ve_mask[idx, :] = torch.any(torch.where(self.edges == idx, True, False), axis=1)

        # Building per edge mask.
        for idx, edge in enumerate(self.edges):
            face_has_endpt0 = torch.any(torch.where(self.faces == edge[0], True, False), axis=1)
            face_has_endpt1 = torch.any(torch.where(self.faces == edge[1], True, False), axis=1)
            ef_mask[idx, :] = torch.logical_and(face_has_endpt0, face_has_endpt1)

        return vf_mask, ve_mask, ef_mask

    # Makes all the mesh computations needed and shared with loss classes: face areas and normals, edge lengths and
    # directions, edge normals. 
    def make_on_mesh_shared_computations(self):
        self.compute_edge_lengths_and_directions()
        self.compute_edge_normals()

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
        :param verts: new verts
        """
        self.vertices = verts

    def plot_mesh(self, colors=None):
        vertices = self.vertices.detach()
        faces = self.faces.detach()
        if colors is not None:
            colors = colors.detach()
        plot_mesh(vertices, faces, colors)