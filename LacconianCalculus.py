import torch
from models.layers.mesh import Mesh

DOF = 6     # degrees of freedom per vertex

class LacconianCalculus:
    
    def __init__(self, file=None, mesh=None, beam_properties=None, beam_have_load=False, device='cpu'):
        self.device = torch.device(device)
        self.set_beam_properties(beam_properties)
        self.beam_have_load = beam_have_load

        if file is not None:
            self.mesh = Mesh(file, device=device)
        elif mesh is not None:
            self.mesh = mesh
        else:
            raise ValueError('No reference mesh specified.')
            
        self.initialize_containers()
        self.mesh.make_on_mesh_shared_computations()
        self.set_loads_and_beam_frames()
        self.beam_model_solve()

    def __call__(self, loss_type):
        self.set_loads_and_beam_frames()
        self.beam_model_solve()

        # Giving right loss corresponding with loss_type.
        if loss_type == 'norm_vertex_deformations':
            return torch.sum(torch.norm(self.vertex_deformations[:, :3], p=2, dim=1))
        elif loss_type == 'mean_beam_energy':
            return torch.mean(self.beam_energy)

    # Store beam properties involved in the task.
    # Custom properties are passed through a list whose elements follow this order:
    #
    # -- [0] Poisson's ratio, default is 0.3;
    # -- [1] Young's modulus, default is 2.1*10^8 kN/m^2;
    # -- [2] beam section area, default is 1*10^-3 m^2; 
    # -- [3] moment of inertia2 Ixx = Iyy, default is 4.189828*10^-8, round section;
    # -- [4] moment of inertia3 Ixx = Iyy, default is 4.189828*10^-8, round section;
    # -- [5] polar moment, default is 8.379656e-8;
    # -- [6] shear section factor, default is 1.2;
    # -- [7] weight per surface unit, default is 3 kN/m^2;
    # -- [8] beam density, default is 78.5 kN/m^3.
    #
    def set_beam_properties(self, beam_properties):
        if beam_properties is not None:
            self.properties = beam_properties
        else:
            self.properties = [0.3, 21e7, 1e-3, 4.189828e-8, 4.189828e-8, 8.379656e-8, 1.2, 3, 78.5]

        # Computing G := young/(2 * (1 + poisson)).
        self.properties.append(self.properties[1] / (2 * (1+self.properties[0])))

        # Converting to torch.tensor on current device.
        self.properties = torch.tensor(self.properties, device=self.device)

    # Re-usable tensors are initialized just once at the beginning.
    # CAUTION: we are exploiting the fact our iterations preserve mesh connectivity.
    def initialize_containers(self):
        # Beam local frames container: (#edges, 3, 3) torch.tensor.
        self.beam_frames = torch.zeros(self.mesh.edges.shape[0], 3, 3, device=self.device)

        # Beam stiff matrices container: (#edges, 2*DOF, 2*DOF) torch.tensor.
        self.beam_stiff_matrices = torch.zeros(self.mesh.edges.shape[0], 2*DOF, 2*DOF, device=self.device)

        # Per-vertex loads vector.
        self.load = torch.zeros(self.mesh.vertices.shape[0] * DOF, device=self.device)

        ###########################################################################################################
        # Storing augmented stiffmatrix non-zero indices.
        self.augmented_stiff_idx = self.make_augmented_stiffmatrix_nnz_indices()

        ###########################################################################################################
        # Bulding bool tensor masks related to constraints.
        # Non-constrained vertex mask:
        self.non_constrained_vertices = torch.logical_or(self.mesh.vertex_is_red, self.mesh.vertex_is_blue).logical_not()

        # Red vertices refer to all dofs constrainess, blue vertices to just translation constrainess.
        blue_dofs = torch.kron(self.mesh.vertex_is_blue, torch.tensor([True] * int(DOF/2) + [False] * int(DOF/2), device=self.device))
        red_dofs = torch.kron(self.mesh.vertex_is_red, torch.tensor([True] * DOF, device=self.device))

        # Non-constrained vertex dofs mask.
        self.dofs_non_constrained_mask = torch.logical_and(blue_dofs.logical_not(), red_dofs.logical_not())

    # Please note, + operator in this case makes row-wise arranged torch.arange(DOF) copies each time shiftedy by
    # corresponding 6 * self.mesh.edges[:, 0].unsqueeze(-1) row element.
    # (...).unsqueeze(-1) expand (#edges, ) tensor to (#edges, 1) column tensor.
    def make_edge_endpoints_dofs_matrix(self):
        endpts_0 = 6 * self.mesh.edges[:, 0].unsqueeze(-1) + torch.arange(DOF, device=self.device)
        endpts_1 = 6 * self.mesh.edges[:, 1].unsqueeze(-1) + torch.arange(DOF, device=self.device)
        return torch.cat((endpts_0, endpts_1), dim=1) 

    # Builds augmented stiffmatrix non-zero element tensor according to vertex dofs.
    def make_augmented_stiffmatrix_nnz_indices(self):
        self.endpoints_dofs_matrix = self.make_edge_endpoints_dofs_matrix()

        dim1_idx = self.endpoints_dofs_matrix.view(-1, 1).expand(-1, 2*DOF).flatten()
        dim2_idx = self.endpoints_dofs_matrix.expand(2*DOF, -1, -1).transpose(0, 1).flatten()

        return torch.stack([dim1_idx, dim2_idx], dim=0)

    # Stores load matrix and builds beam frames.
    # Load matrix (no. vertices x DOF) has row-structure (Fx, Fy, Fz, Mx, My, Mz) whose values are referred to global ref system.
    def set_loads_and_beam_frames(self):
        #####################################################################################################################################
        # Load computation.
        # Computing beam loads on each vertex: -1/2 * <beam volume> * <beam density> (beam load is equally parted to endpoints) along z-axis.
        # Some details: vec.scatter_add(0, idx, src) with vec, idx, src 1d tensors, add at vec positions specified
        # by idx corresponding src values.
        if self.beam_have_load:
            beam_loads = torch.mul(self.mesh.edge_lengths, -1/2 * self.properties[2] * self.properties[8])
            on_vertices_beam_loads = torch.zeros(self.mesh.vertices.shape[0], device=self.device)
            on_vertices_beam_loads.scatter_add_(0, self.mesh.edges.flatten(), torch.stack([beam_loads] * 2, dim=1).flatten())

        # Computing face loads on each vertex: -1/3 * <face areas> * <weight per surface unit> (face load is equally parted to vertices) along z.
        # Some details: vec.scatter_add(0, idx, src) with vec, idx, src 1d tensors, add at vec positions specified
        # by idx corresponding src values.
        face_loads = torch.mul(self.mesh.face_areas, -1/3 * self.properties[7])
        on_vertices_face_loads = torch.zeros(self.mesh.vertices.shape[0], device=self.device)
        on_vertices_face_loads.scatter_add_(0, self.mesh.faces.flatten(), torch.stack([face_loads] * 3, dim=1).flatten())

        # Summing beam and face components to compute per-vertex loads.
        if self.beam_have_load:
            self.load[DOF*torch.arange(self.mesh.vertices.shape[0], device=self.device) + 2] = on_vertices_beam_loads + on_vertices_face_loads
        else:
            self.load[DOF*torch.arange(self.mesh.vertices.shape[0], device=self.device) + 2] = on_vertices_face_loads

        #######################################################################################################################################
        # Beam frames computation.
        self.beam_frames[:, 0, :] = self.mesh.edge_directions
        self.beam_frames[:, 1, :] = self.mesh.edge_normals
        self.beam_frames[:, 2, :] = torch.cross(self.mesh.edge_directions, self.mesh.edge_normals)

    # Execute all stiffness and resistence computations.
    def beam_model_solve(self):
        self.build_stiff_matrix()
        self.compute_stiff_deformation()

    # Stiffness matrices in beam reference systems are computed and then aggregated to compound a global stiff matrix.
    def build_stiff_matrix(self):
        ###########################################################################################################
        # Assembling beam-local stiff matrices whose have the following structure:
        #     k1 = properties.young * properties.cross_area / self.mesh.edge_lengths[edge_id]
        #     k2 = 12 * properties.young * properties.inertia2 / (self.mesh.edge_lengths[edge_id]**3)
        #     k3 = 12 * properties.young * properties.inertia3 / (self.mesh.edge_lengths[edge_id]**3)
        #     k4 = 6 * properties.young * properties.inertia3 / (self.mesh.edge_lengths[edge_id]**2)
        #     k5 = 6 * properties.young * properties.inertia2 /(self.mesh.edge_lengths[edge_id]**2)
        #     k6 = self.G * properties.polar / self.mesh.edge_lengths[edge_id]
        #     k7 = properties.young * properties.inertia2 / self.mesh.edge_lengths[edge_id]
        #     k8 = properties.young * properties.inertia3 / self.mesh.edge_lengths[edge_id]
        #     k_e = [[k1,     0,	    0,      0,	    0,      0,     -k1,	    0,       0, 	0,  	 0,      0],
        #             [0,     k3,	    0,      0,	    0,      k4,	     0,   -k3,       0, 	0,  	 0,     k4],
        #             [0,      0,	   k2,      0,	  -k5,       0, 	 0,	    0,     -k2, 	0,     -k5,      0],
        #             [0,      0,	    0,     k6,	    0,       0,	     0,	    0,       0,   -k6,  	 0,      0],
        #             [0,      0,     -k5,      0,	 4*k7,       0, 	 0,	    0,      k5, 	0,	  2*k7, 	 0],         
        #             [0,     k4,       0,      0,	    0,    4*k8,	     0,   -k4,       0, 	0,	     0,   2*k8],         
        #             [-k1,	   0,	    0,      0,	    0,       0, 	k1,	    0,       0, 	0,	     0,      0],
        #             [0,    -k3,	    0,      0,	    0,     -k4, 	 0,	   k3,       0, 	0,	     0,    -k4],
        #             [0,      0,	  -k2,      0,     k5,       0,	     0,	    0,      k2,	    0,  	k4,      0],
        #             [0,      0,	    0,    -k6,	    0,       0,	     0,	    0,       0,	   k6,	     0,      0],
        #             [0,      0,	  -k5,      0,	 2*k7,	     0,	     0,	    0,      k5,	    0,    4*k7,	     0],
        #             [0,     k4,	    0,      0,	    0,    2*k8,	     0,	  -k4,       0,     0,  	 0,	  4*k8]]
        #

        # Computing squares and cubes of beam_lenghts tensor.
        squared_beam_lenghts = torch.pow(self.mesh.edge_lengths, 2)
        cubed_beam_lenghts = torch.pow(self.mesh.edge_lengths, 3)

        # Filling non empty entries in self.beam_stiff_matrices.
        # k1 and -k1
        self.beam_stiff_matrices[:, 0, 0] = self.properties[1] * self.properties[2] / self.mesh.edge_lengths
        self.beam_stiff_matrices[:, 6, 6] = self.beam_stiff_matrices[:, 0, 0]
        self.beam_stiff_matrices[:, 6, 0] = self.beam_stiff_matrices[:, 0, 6] = -self.beam_stiff_matrices[:, 0, 0]
        # k2 and -k2
        self.beam_stiff_matrices[:, 2, 2] = 12 * self.properties[1] * self.properties[3] / cubed_beam_lenghts
        self.beam_stiff_matrices[:, 8, 8] = self.beam_stiff_matrices[:, 2, 2]
        self.beam_stiff_matrices[:, 8, 2] = self.beam_stiff_matrices[:, 2, 8] = -self.beam_stiff_matrices[:, 2, 2]
        # k3 and -k3
        self.beam_stiff_matrices[:, 1, 1] = 12 * self.properties[1] * self.properties[4] / cubed_beam_lenghts
        self.beam_stiff_matrices[:, 7, 7] = self.beam_stiff_matrices[:, 1, 1]
        self.beam_stiff_matrices[:, 7, 1] = self.beam_stiff_matrices[:, 1, 7] = -self.beam_stiff_matrices[:, 1, 1]
        # k4 and -k4
        self.beam_stiff_matrices[:, 5, 1] = 6 * self.properties[1] * self.properties[4] / squared_beam_lenghts
        self.beam_stiff_matrices[:, 1, 5] = self.beam_stiff_matrices[:, 11, 1]  = self.beam_stiff_matrices[:, 5, 1]
        self.beam_stiff_matrices[:, 1, 11] = self.beam_stiff_matrices[:, 8, 10] = self.beam_stiff_matrices[:, 5, 1]
        self.beam_stiff_matrices[:, 7, 5] = self.beam_stiff_matrices[:, 11, 7] = - self.beam_stiff_matrices[:, 5, 1]
        self.beam_stiff_matrices[:, 5, 7] = self.beam_stiff_matrices[:, 7, 11] = - self.beam_stiff_matrices[:, 5, 1]
        # k5 and -k5
        self.beam_stiff_matrices[:, 8, 4] =  6 * self.properties[1] * self.properties[3] / squared_beam_lenghts
        self.beam_stiff_matrices[:, 4, 8] = self.beam_stiff_matrices[:, 10, 8] = self.beam_stiff_matrices[:, 8, 4]
        self.beam_stiff_matrices[:, 4, 2] = self.beam_stiff_matrices[:, 10, 2] = -self.beam_stiff_matrices[:, 8, 4]
        self.beam_stiff_matrices[:, 2, 4] = self.beam_stiff_matrices[:, 2, 10] = -self.beam_stiff_matrices[:, 8, 4]
        # k6 and -k6
        self.beam_stiff_matrices[:, 3, 3] = self.properties[9] * self.properties[5] / self.mesh.edge_lengths
        self.beam_stiff_matrices[:, 9, 9] = self.beam_stiff_matrices[:, 3, 3]
        self.beam_stiff_matrices[:, 9, 3] = self.beam_stiff_matrices[:, 3, 9] = -self.beam_stiff_matrices[:, 3, 3]
        # k7 and multiples
        k7 = self.properties[1] * self.properties[3] / self.mesh.edge_lengths
        self.beam_stiff_matrices[:, 4, 4] = self.beam_stiff_matrices[:, 10, 10] = torch.mul(4, k7)
        self.beam_stiff_matrices[:, 10, 4] = self.beam_stiff_matrices[:, 4, 10] = torch.mul(2, k7)
        # k8 and multiples
        k8 = self.properties[1] * self.properties[4] / self.mesh.edge_lengths
        self.beam_stiff_matrices[:, 5, 5] = self.beam_stiff_matrices[:, 11, 11] = torch.mul(4, k8)
        self.beam_stiff_matrices[:, 11, 5] = self.beam_stiff_matrices[:, 5, 11] = torch.mul(2, k8)

        ###########################################################################################################
        # Assembling beam-local to global transition matrices via Kronecker product: container is again a
        # 3-dimensional torch.tensor.
        transition_matrices = torch.kron(torch.eye(4, 4, device=self.device), self.beam_frames)

        ###########################################################################################################
        # Building beam contributions to global stiff matrix: container (beam_contributions) is a 3d torch.tensor,
        # another contanier (self.beam_forces_contributions) for products beam_stiff_matrix @ transition_matrix
        # is saved in order not to repeat steps in node forces computation phase.
        # Please note: @ operator for 3d tensors produces a 'batched' 2d matrix multiplication along sections.
        self.beam_forces_contributions = self.beam_stiff_matrices @ transition_matrices
        beam_contributions = torch.transpose(transition_matrices, 1, 2) @ self.beam_forces_contributions
                                                #torch.transpose(_, 1, 2) transpose along dimensions 1 and 2

        ###########################################################################################################
        # Building global stiff matrix by adding all beam contributions.
        # Global stiff matrix: (DOF*#vertices, DOF*#vertices), computed by densing an augmented sparse
        # (DOF*#vertices, DOF*#vertices), admitting several entries for the same ij coordinate couple.
        size = (DOF * self.mesh.vertices.shape[0], DOF * self.mesh.vertices.shape[0])
        augmented_stiff_matrix = torch.sparse_coo_tensor(self.augmented_stiff_idx, beam_contributions.flatten(), size=size, device=self.device)
        self.stiff_matrix = augmented_stiff_matrix.to_dense()

        # Freeing memory space.
        del transition_matrices, beam_contributions, augmented_stiff_matrix, squared_beam_lenghts, cubed_beam_lenghts

    # Compute vertex deformations by solving a stiff-matrix-based linear system.
    def compute_stiff_deformation(self):
        # Solving reduced linear system and adding zeros in constrained dofs.
        self.vertex_deformations = torch.zeros(len(self.mesh.vertices) * DOF, device=self.device)
        sys_sol = torch.linalg.solve(self.stiff_matrix[self.dofs_non_constrained_mask][:, self.dofs_non_constrained_mask], self.load[self.dofs_non_constrained_mask])
        self.vertex_deformations[self.dofs_non_constrained_mask] = sys_sol

        # Freeing memory space.
        del self.stiff_matrix, sys_sol

        # Computing beam resulting forces and energies.
        self.compute_beam_force_and_energy()

        # Making deformation tensor by reshaping self.vertex_deformations.
        self.vertex_deformations = self.vertex_deformations.view(self.mesh.vertices.shape[0], DOF)

    # Computing beam resulting forces and energies.
    def compute_beam_force_and_energy(self):
        # edge_dofs_deformations aggregates vertex_deformation in a edge-endpoints-wise manner: (#edges, 2*DOF) torch.tensor
        edge_dofs_deformations = self.vertex_deformations[self.endpoints_dofs_matrix]

        #############################################################################################################################
        # Computing resulting forces at nodes via 'batched' matrix multiplication @.
        # Some details:
        # edge_dofs_deformations.unsqueeze(2) expands (#edges, 2*DOF) to #edge batches of (2*DOF, 1), i.e. (#edges, 2*DOF, 1) tensor;
        # vice-versa, (...).squeeze(2) contracts (#edges, 2*DOF, 1) -> (#edges, 2*DOF)
        node_forces = (self.beam_forces_contributions @ edge_dofs_deformations.unsqueeze(2)).squeeze(2)

        # Averaging force components 0:DOF with D0F:2*DOF of opposite sign.
        mean_forces = torch.mul(0.5, node_forces[:, :DOF] - node_forces[:, DOF:2*DOF])

        # Computing beam energy according this coordinate system:
        # axes: 1=elementAxis; 2=EdgeNormal; 3=InPlaneAxis
        # output rows: [Axial_startNode; Shear2_startNode; Shear3_startNode; Torque_startNode; Bending3_startNode; Bending2_startNode;
        #                ...Axial_endNode; Shear2_endNode; Shear3_endNode; Torque_endNode; Bending3_endNode; Bending2_endNode]
        self.beam_energy = self.mesh.edge_lengths/2 * ( mean_forces[:, 0]**2/(self.properties[1]*self.properties[2]) +
                            self.properties[6] * mean_forces[:, 1]**2 / (self.properties[9] * self.properties[2]) +
                            self.properties[6] * mean_forces[:, 2]**2 / (self.properties[9] * self.properties[2]) +
                            mean_forces[:, 3]**2 / (self.properties[9] * self.properties[5]) +
                            mean_forces[:, 4]**2 / (self.properties[1] * self.properties[4]) +
                            mean_forces[:, 5]**2 / (self.properties[1] * self.properties[3]) )

        # Freeing memory space.
        del edge_dofs_deformations, self.beam_forces_contributions

    def clean_attributes(self):
        self.mesh.vertices.detach_()
        self.load.detach_()
        self.beam_frames.detach_()
        self.beam_stiff_matrices.detach_()
        
    # Displace initial mesh with self.beam_model_solve() computed translations.
    def displace_mesh(self):
        # REQUIRES: self.beam_model_solve() has to be called before executing this.
        if not hasattr(self, 'vertex_deformations'):
            raise RuntimeError("self.beam_model_solve() method not called yet.")

        # Updating mesh vertices.
        self.mesh.update_verts(self.mesh.vertices + self.vertex_deformations[:, :int(DOF/2)])

    # Show displaced mesh via meshplot.
    def plot_grid_shell(self):
        colors = torch.norm(self.vertex_deformations[:, :3], p=2, dim=1)
        self.mesh.plot_mesh(colors)

# lc = LacconianCalculus(file='meshes/Shell.ply', device='cpu')
# lc.displace_mesh()
# lc.plot_grid_shell()
