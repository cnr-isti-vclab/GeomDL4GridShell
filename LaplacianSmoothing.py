'''
Adapted from pytorch3d.loss.mesh_laplacian_smoothing.py; pytorch3d.ops.laplacian_matrices.py
Copyright (c) Facebook, Inc. and its affiliates.
'''
import torch

class LaplacianSmoothing:

    def __init__(self, mesh, device, relative=False):
        self.device = torch.device(device)
        self.relative = relative                            # If True, we compute a relative version of laplacian smoothing reg.

        # Computing initial laplacian: computation rules are explained in __call__ method.
        if self.relative:
            L, inv_areas = self.cotan_matrix(mesh)
            L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1,1)
            norm_w = 0.25 * inv_areas
        
            self.mean_curvatures_0 = torch.norm((L.mm(mesh.vertices) - L_sum * mesh.vertices) * norm_w, p=2, dim=1)

    def __call__(self, mesh):
        '''
        LaplacianSmoothing loss is related to mean curvature averaged over vertices.
        Recalling that ||LV[i]|| = H(V[i]), H mean curvature, we use as loss
        1/|V| * sum_V[i] ||LV[i]||.
        '''
        with torch.no_grad():
            L, inv_areas = self.cotan_matrix(mesh)
            L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1,1)
            norm_w = 0.25 * inv_areas

        '''
        We have
        LV[i] = sum_j w_ij * (v_j - v_i) = (sum_j w_ij * v_j) - (sum_j w_ij * v_i) 
                                                                L_sum[i] * v_i
        1/(4*A[i]) are multiplied at the end (as norm_w).
        '''
        mean_curvatures = torch.norm((L.mm(mesh.vertices) - L_sum * mesh.vertices) * norm_w, p=2, dim=1)

        if self.relative == False:
            loss = torch.mean(mean_curvatures)
        else:
            loss = torch.mean(torch.abs((mean_curvatures - self.mean_curvatures_0) / self.mean_curvatures_0))

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
        We use the cotangent laplacian curvature variant,
            w_ij = (cot a_ij + cot b_ij) / (4 * A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.
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
        L = torch.sparse_coo_tensor(idx, cot.view(-1), (mesh.vertices.shape[0], mesh.vertices.shape[0]), device=self.device)

        # Make it symmetric; this means we are also setting
        # L[v2, v1] = cota
        # L[v0, v2] = cotb
        # L[v1, v0] = cotc
        L += L.t()

        # For each vertex, compute the sum of areas for triangles containing it.
        idx = mesh.faces.flatten()
        inv_areas = torch.zeros(mesh.vertices.shape[0], device=self.device)
        val = torch.stack([mesh.face_areas] * 3, dim=1).flatten()
        inv_areas.scatter_add_(0, idx, val)
        idx = inv_areas > 0
        inv_areas[idx] = 1.0 / inv_areas[idx]
        inv_areas = inv_areas.view(-1,1)

        return L, inv_areas