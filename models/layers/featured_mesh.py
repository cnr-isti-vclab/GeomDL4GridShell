import torch
import warnings
from models.layers.mesh import Mesh
from torch.nn.functional import normalize
from utils import extract_apss_principal_curvatures, extract_geodesic_distances, get_cotan_matrix

class FeaturedMesh(Mesh):

    def __init__(self, file, vertices=None, faces=None, device='cpu'):
        super(FeaturedMesh, self).__init__(file, vertices, faces, device)
        self.file = file

    def compute_mesh_input_features(self):
        feature_list = []

        # input_features[:, 0:3]: vertex coordinates.
        feature_list.append(self.vertices)

        # input_features[:, 3:6]: vertex normals.
        self.compute_vertex_normals()
        feature_list.append(self.vertex_normals)

        # input_features[:, 6:8]: principal curvatures.
        k1, k2  = extract_apss_principal_curvatures(self.file)
        k1 = torch.from_numpy(k1).to(self.device).unsqueeze(1)
        k2 = torch.from_numpy(k2).to(self.device).unsqueeze(1)
        feature_list.append(k1)
        feature_list.append(k2)

        # input_features[:, 8]: geodesic distance (i.e min geodetic distance) from firm vertices;
        # input_features[:, 9]: geodesic centrality (i.e mean of geodetic distances) from firm vertices;
        # input_features[:, 10]: geodesic distance (i.e min geodetic distance) from mesh boundary;
        # input_features[:, 11]: geodesic centrality (i.e mean of geodetic distances) from mesh boundary.
        geodesic_distance_firm, geodesic_centrality_firm, geodesic_distance_bound, geodesic_centrality_bound = extract_geodesic_distances(self.file)
        geodesic_distance_firm = torch.from_numpy(geodetic_distance_firm).to(self.device).unsqueeze(1)
        geodesic_centrality_firm = torch.from_numpy(geodetic_centrality_firm).to(self.device).unsqueeze(1)
        geodesic_distance_bound = torch.from_numpy(geodetic_distance_bound).to(self.device).unsqueeze(1)
        geodesic_centrality_bound = torch.from_numpy(geodetic_centrality_bound).to(self.device).unsqueeze(1)
        # feature_list.append(geodesic_distance_firm)
        # feature_list.append(geodesic_centrality_firm)
        # feature_list.append(geodesic_distance_bound)
        # feature_list.append(geodesic_centrality_bound)

        # input_features[:, 12:16]: first 4 laplacian eigenvectors.
        eigenvectors = self.compute_laplacian_eigs(4)
        feature_list.append(eigenvectors)

        self.input_features = torch.cat(feature_list, dim=1) 

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

    def compute_laplacian_eigs(self, n):
        # Getting mesh cotan-laplacian matrix.
        mass, cot = get_cotan_matrix(self)
        mass = torch.from_numpy(mass).to(self.device).unsqueeze(1)
        cot = torch.from_numpy(cot).to(self.device)
        L = cot / (1 / mass)

        # Computing cotan-laplacian eigenpairs.
        eigenpairs = torch.linalg.eig(L)
        warnings.simplefilter('ignore')
        eigenvalues = eigenpairs.eigenvalues.to(torch.float32)
        eigenvectors = eigenpairs.eigenvectors.to(torch.float32)
        warnings.simplefilter('default')

        # Taking the n first module-bigger eigenvectors.
        sorted_idx = torch.sort(torch.abs(eigenvalues), descending=False).indices[ :n]
        return eigenvectors[:, sorted_idx]

