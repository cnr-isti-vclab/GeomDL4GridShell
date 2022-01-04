'''
Adapted from pytorch3d.loss.mesh_laplacian_smoothing.py; pytorch3d.ops.laplacian_matrices.py
https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/laplacian_matrices.py
https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/laplacian_matrices.py
Copyright (c) Facebook, Inc. and its affiliates.
'''
import torch

class LaplacianSmoothing:

    def __init__(self, device):
        self.device = torch.device(device)

    def __call__(self, mesh):
        '''
        LaplacianSmoothing loss is related to mean curvature averaged over vertices.
        Recalling that ||LV[i]|| = 2 * H(V[i]), H mean curvature, we use as loss
        1/|V| * sum_V[i] ||LV[i]||.
        '''
        with torch.no_grad():
            L = self.cotan_matrix(mesh)
            norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            idx = norm_w > 0
            norm_w[idx] = 1.0 / norm_w[idx]

        loss = L.mm(mesh.vertices) * norm_w - mesh.vertices
        loss = torch.norm(loss, dim=1)
        loss = torch.mean(loss)
        return loss

    def cotan_matrix(self, mesh):
        '''
               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij
        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        We use the cotangent laplacian variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        There is a nice trigonometry identity to compute cotangents. Consider a triangle
        with side lengths A, B, C and angles a, b, c.
               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C
        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have
        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a
        Putting these together, we get:
        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH

        Since sum_j w_ij = 1, we have
        LV[i] = sum_j w_ij * (v_j - v_i) = sum_j w_ij * v_j - (sum_j w_ij) * v_i =
              = (sum_j w_ij * v_j) - v_i.
        '''

        # Given a generic face F = <v0, v1, v2>, A contains lengths of v0-opposite edges,
        # B lenghts of v1-opposite edges, C of v2-opposite edges.
        A = mesh.edge_lengths[mesh.edges_per_face[:, 1]]
        B = mesh.edge_lengths[mesh.edges_per_face[:, 2]]
        C = mesh.edge_lengths[mesh.edges_per_face[:, 0]]

        # Compute cotangents of angles
        A2, B2, C2 = A * A, B * B, C * C
        cota = (B2 + C2 - A2) / mesh.face_areas
        cotb = (A2 + C2 - B2) / mesh.face_areas
        cotc = (A2 + B2 - C2) / mesh.face_areas
        cot = torch.stack([cota, cotb, cotc], dim=1)
        cot /= 4.0

        # Construct a sparse matrix by basically doing:
        # L[v1, v2] = cota
        # L[v2, v0] = cotb
        # L[v0, v1] = cotc
        ii = mesh.faces[:, [1, 2, 0]]
        jj = mesh.faces[:, [2, 0, 1]]
        idx = torch.stack([ii, jj], dim=0).view(2, mesh.faces.shape[0] * 3)
        L = torch.sparse.FloatTensor(idx, cot.view(-1), (mesh.vertices.shape[0], mesh.vertices.shape[0]), device=self.device)

        # Make it symmetric; this means we are also setting
        # L[v2, v1] = cota
        # L[v0, v2] = cotb
        # L[v1, v0] = cotc
        L += L.t()

        return L