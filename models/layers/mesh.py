##################################################################################################
#CHANGES:
# -- new parameters Mesh.vertex_is_constrained, Mesh.edges.
# -- utils.load_obj is replaced by utils.load_mesh;
# -- optional parameter normalize added in face_areas_normals method;
# -- Mesh.vs renamed to Mesh.vertices;
# -- Mesh.normals renamed to Mesh.face_normals;
# -- Mesh.area renamed to Mesh.face_areas.

import torch
import numpy as np
#from queue import Queue
from utils import load_mesh, plot_mesh, get_edge_matrix
from torch.nn.functional import normalize
#import copy
#from pathlib import Path
#import pickle
#from pytorch3d.ops.knn import knn_gather, knn_points


class Mesh:

    def __init__(self, file, hold_history=False, vertices=None, faces=None, device='cpu', gfmm=True, vertices_have_grad=False):
        if file is None:
            return
        #self.filename = Path(file)
        self.vertices = self.v_mask = self.edge_areas = None
        self.edges = self.gemm_edges = self.sides = None
        self.device = torch.device(device)
        if vertices is not None and faces is not None:
            self.vertices, self.faces = vertices.cpu().numpy(), faces.cpu().numpy()
            #self.scale, self.translations = 1.0, np.zeros(3,)
        else:
            self.vertices, self.faces, self.vertex_is_constrainted = load_mesh(file)
            #self.normalize_unit_bb()
        #self.vs_in = copy.deepcopy(self.vertices)
        #self.v_mask = np.ones(len(self.vertices), dtype=bool)
        #self.build_gemm()
        #self.history_data = None
        #if hold_history:
        #    self.init_history()
        #if gfmm:
        #    self.gfmm = self.build_gfmm() #TODO get rid of this DS
        #else:
        #    self.gfmm = None
        self.edges = get_edge_matrix(self.faces)
        if type(self.vertices) is np.ndarray:
            self.vertices = torch.from_numpy(self.vertices)
        if type(self.faces) is np.ndarray:
            self.faces = torch.from_numpy(self.faces)
        if type(self.vertex_is_constrainted) is np.ndarray:
            self.vertex_is_constrainted = torch.from_numpy(self.vertex_is_constrainted)
        self.vertices = self.vertices.to(device)
        self.faces = self.faces.long()
        #self.face_areas, self.face_normals = self.face_areas_normals(self.vertices, self.faces)

    def compute_edge_lengths_and_directions(self):
        #Edge directions are computed by endpoints difference.
        ix_grid_1 = np.ix_(self.edges[:, 0], 3*[True])
        ix_grid_2 = np.ix_(self.edges[:, 1], 3*[True])
        edge_directions = self.vertices[ix_grid_2] - self.vertices[ix_grid_1]

        #Computing edge lengths.
        edge_lengths = torch.norm(edge_directions, p=2, dim=1)

        #Normalizing edge directions.
        edge_directions = edge_directions / edge_lengths[:, None]

        return edge_directions, edge_lengths

    def compute_edge_normals(self):
        #Getting face normals and areas.
        self.face_areas, self.face_normals = self.face_areas_normals(self.vertices, self.faces, normalize=False)

        #Getting face incidence mask per vertex.
        self.incidence_mask = self.make_per_vertex_face_incidence_mask()

        #Computing vertex normals from incident faces.
        vertex_normals = torch.zeros(len(self.vertices), 3, device=self.device)
        for vertex_id in range(len(self.vertices)):
            #Selecting incident faces in the current vertex.
            ix_mask = self.incidence_mask[vertex_id, :]
            incident_faces_normals = self.faces[ix_mask, :]

            #Vertex normal is deduced just by summing.
            vertex_normals[vertex_id, :] = torch.sum(incident_faces_normals, dim=0)
        vertex_normals = normalize(vertex_normals, p=2, dim=1)

        #Computing edge normals by endpoint means.
        edge_normals = 0.5 * vertex_normals[self.edges[:, 0], :] + 0.5 * vertex_normals[self.edges[:, 1], :]
        edge_normals = normalize(edge_normals, p=2, dim=1)

        return edge_normals

    def make_per_vertex_face_incidence_mask(self):
        mask = []
        for vertex_id in range(len(self.vertices)):
            mask.append(np.any(np.where(self.faces == vertex_id, True, False), axis=1))
        return np.array(mask)

    @staticmethod
    def face_areas_normals(vs, faces, normalize=True):
        if type(vs) is not torch.Tensor:
            vs = torch.from_numpy(vs)
        if type(faces) is not torch.Tensor:
            faces = torch.from_numpy(faces)

        face_normals = torch.cross(vs[faces[:, 1]] - vs[faces[:, 0]],
                                   vs[faces[:, 2]] - vs[faces[:, 1]])
        face_areas = torch.norm(face_normals, dim=1)

        if normalize:
            face_normals = face_normals / face_areas[:, None]
        face_areas = 0.5 * face_areas

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





