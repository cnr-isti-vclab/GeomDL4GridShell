import torch
import warnings
from models.layers.mesh import Mesh
from torch.nn.functional import normalize
from utils.utils import extract_apss_principal_curvatures, extract_geodesic_distances

# Class that extends Mesh for admitting vertex input feature vectors
class FeaturedMesh(Mesh):

    def __init__(self, file, vertices=None, faces=None, device='cpu'):
        super(FeaturedMesh, self).__init__(file, vertices, faces, device)
        self.file = file

    def compute_mesh_input_features(self):
        feature_list = []
        feature_mask = []

        # input_features[:, 0:3]: vertex coordinates.
        feature_list.append(self.vertices)
        feature_mask.append([0, 1, 2])

        # input_features[:, 3:6]: vertex normals.
        self.compute_vertex_normals()
        feature_list.append(self.vertex_normals)
        feature_mask.append([3, 4, 5])

        # input_features[:, 6:8]: principal curvatures.
        k1, k2 = extract_apss_principal_curvatures(self.file)
        k1 = torch.from_numpy(k1).to(self.device)
        k2 = torch.from_numpy(k2).to(self.device)
        feature_list.append(k1)
        feature_list.append(k2)
        feature_mask.append([6, 7])

        # input_features[:, 8]: geodesic distance (i.e min geodetic distance) from firm vertices;
        # input_features[:, 9]: geodesic centrality (i.e mean of geodetic distances) from firm vertices;
        # input_features[:, 10]: geodesic distance (i.e min geodetic distance) from mesh boundary;
        # input_features[:, 11]: geodesic centrality (i.e mean of geodetic distances) from mesh boundary.
        geodesic_distance_firm, geodesic_centrality_firm, geodesic_distance_bound, geodesic_centrality_bound = extract_geodesic_distances(self.file)
        geodesic_distance_firm = torch.from_numpy(geodesic_distance_firm).to(self.device).unsqueeze(1)
        geodesic_centrality_firm = torch.from_numpy(geodesic_centrality_firm).to(self.device).unsqueeze(1)
        geodesic_distance_bound = torch.from_numpy(geodesic_distance_bound).to(self.device).unsqueeze(1)
        geodesic_centrality_bound = torch.from_numpy(geodesic_centrality_bound).to(self.device).unsqueeze(1)
        feature_list.append(geodesic_distance_firm)
        feature_list.append(geodesic_centrality_firm)
        feature_list.append(geodesic_distance_bound)
        feature_list.append(geodesic_centrality_bound)
        feature_mask.append([8, 9, 10, 11])

        self.input_features = torch.cat(feature_list, dim=1)
        self.feature_mask = feature_mask

    def compute_vertex_normals(self):
        ############################################################################################################
        # Computing vertex normals by weighting normals from incident faces.
        # Some details: vec.scatter_add(0, idx, src) with vec, idx, src 1d tensors, add at vec positions specified
        # by idx corresponding src values.
        vertex_normals = torch.zeros(self.vertices.shape[0], 3, device=self.device)

        vertex_normals[:, 0].scatter_add_(0, self.faces.flatten(), torch.stack([self.face_normals[:, 0]] * 3, dim=1).flatten())
        vertex_normals[:, 1].scatter_add_(0, self.faces.flatten(), torch.stack([self.face_normals[:, 1]] * 3, dim=1).flatten())
        vertex_normals[:, 2].scatter_add_(0, self.faces.flatten(), torch.stack([self.face_normals[:, 2]] * 3, dim=1).flatten())

        # Applying final l2-normalization.
        self.vertex_normals = normalize(vertex_normals, p=2, dim=1)
        