import torch
from models.layers.mesh import Mesh

DOF = 6     #degrees of freedom per vertex

class LacconianCalculus:
    
    def __init__(self, file=None, mesh=None, beam_properties=None, device='cpu'):
        self.device = torch.device(device)
        self.set_beam_properties(beam_properties)

        if file is not None:
            self.mesh = Mesh(file, device=device)
            self.initialize_containers()
            self.set_beam_model_data()
            self.beam_model_solve()

        if mesh is not None:
            self.mesh = mesh
            self.initialize_containers()

    def __call__(self):
        self.set_beam_model_data()
        self.beam_model_solve()
        return torch.sum(torch.norm(self.vertex_deformations[:, :3], p=2, dim=1))
        #return torch.sum(self.beam_energy)

    #Store beam properties involved in the task.
    #Custom properties are passed through a list whose elements follow this order:
    #
    # -- Poisson's ratio, default is 0.3;
    # -- Young's modulus, default is 2.1*10^8 kN/m^2;
    # -- beam section area, default is 1*10^-3 m^2; 
    # -- moment of inertia2 Ixx = Iyy, default is 4.189828*10^-8, round section;
    # -- moment of interia3 Ixx = Iyy, default is 4.189828*10^-8, round section;
    # -- polar moment, default is 8.379656e-8;
    # -- shear section factor, default is 1.2;
    # -- weight per surface unit, default is 3 kN/m^2.
    #
    def set_beam_properties(self, beam_properties):
        if beam_properties is not None:
            self.properties = beam_properties
        else:
            self.properties = [0.3, 21e7, 1e-3, 4.189828e-8, 4.189828e-8, 8.379656e-8, 1.2, -3]

        #Computing G := young/(2 * (1 + poisson)).
        self.properties.append(self.properties[1] / (2 * (1+self.properties[0])))

        #Converting to torch.tensor on current device.
        self.properties = torch.tensor(self.properties, device=self.device)

    #Re-usable tensors are initialized just once at the beginning.
    #CAUTION: we are exploiting the fact our iterations preserve mesh connectivity.
    def initialize_containers(self):
        #Beam local frames container: (#edges, 3, 3) torch.tensor.
        self.beam_frames = torch.zeros(len(self.mesh.edges), 3, 3, device=self.device)

        #Beam stiff matrices container: (#edges, 2*DOF, 2*DOF) torch.tensor.
        self.beam_stiff_matrices = torch.zeros(len(self.mesh.edges), 2*DOF, 2*DOF, device=self.device)

        #Global stiff matrix: (DOF*#vertices, DOF*#vertices)
        self.stiff_matrix = torch.zeros(DOF*len(self.mesh.vertices), DOF*len(self.mesh.vertices), device=self.device)

        ###########################################################################################################
        #Building endpoints-related dofs per edge matrix.
        self.endpoints_dofs_matrix = self.make_edge_endpoints_dofs_matrix()

        ###########################################################################################################
        #Building non-constrained vertex dofs mask, using Kronecker product.
        self.non_constrained_vertices = self.mesh.vertex_is_constrained.logical_not()
        self.dofs_non_constrained_mask = torch.kron(self.non_constrained_vertices, torch.tensor([True] * DOF, device=self.device))

    #Please note, + operator in this case makes row-wise arranged torch.arange(DOF) copies each time shiftedy by
    #corresponding 6 * self.mesh.edges[:, 0].unsqueeze(-1) row element.
    # (...).unsqueeze(-1) expand (#edges, ) tensor to (#edges, 1) column tensor.
    def make_edge_endpoints_dofs_matrix(self):
        endpts_0 = 6 * self.mesh.edges[:, 0].unsqueeze(-1) + torch.arange(DOF, device=self.device)
        endpts_1 = 6 * self.mesh.edges[:, 1].unsqueeze(-1) + torch.arange(DOF, device=self.device)
        return torch.cat((endpts_0, endpts_1), dim=1)

    #Stores load matrix, beam lengths, beam local frames.
    #Load matrix (no. vertices x DOF) has row-structure (Fx, Fy, Fz, Mx, My, Mz) whose values are referred to global ref system.
    def set_beam_model_data(self):
        #Computation of all mesh-based prerequisites.
        beam_directions, self.beam_lengths = self.mesh.compute_edge_lengths_and_directions()
        beam_normals = self.mesh.compute_edge_normals()
        
        #Saving load matrix: all entries are zero except Fz who is set -1/3 * weight_per_surface * <sum of all incident face areas>
        self.load = torch.zeros(len(self.mesh.vertices) * DOF, device=self.device)
        for idx, face_vertex_mask in enumerate(self.mesh.incidence_mask):
            self.load[DOF*idx + 2] = 1/3 * self.properties[7] * torch.sum(self.mesh.face_areas[face_vertex_mask])

        self.beam_frames[:, 0, :] = beam_directions
        self.beam_frames[:, 1, :] = beam_normals
        self.beam_frames[:, 2, :] = torch.cross(beam_directions, beam_normals)

    #Execute all stiffness and resistence computations.
    def beam_model_solve(self):
        self.build_stiff_matrix()
        self.compute_stiff_deformation()

    #Stiffness matrices in beam reference systems are computed and then aggregated to compound a global stiff matrix.
    def build_stiff_matrix(self):
        ###########################################################################################################
        #Assembling beam-local stiff matrices whose have the following structure:
        #     k1 = properties.young * properties.cross_area / self.beam_lengths[edge_id]
        #     k2 = 12 * properties.young * properties.inertia2 / (self.beam_lengths[edge_id]**3)
        #     k3 = 12 * properties.young * properties.inertia3 / (self.beam_lengths[edge_id]**3)
        #     k4 = 6 * properties.young * properties.inertia3 / (self.beam_lengths[edge_id]**2)
        #     k5 = 6 * properties.young * properties.inertia2 /(self.beam_lengths[edge_id]**2)
        #     k6 = self.G * properties.polar / self.beam_lengths[edge_id]
        #     k7 = properties.young * properties.inertia2 / self.beam_lengths[edge_id]
        #     k8 = properties.young * properties.inertia3 / self.beam_lengths[edge_id]
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

        #Computing squares and cubes of beam_lenghts tensor.
        squared_beam_lenghts = torch.pow(self.beam_lengths, 2)
        cubed_beam_lenghts = torch.pow(self.beam_lengths, 3)

        #Filling non empty entries in self.beam_stiff_matrices.
        # k1 and -k1
        self.beam_stiff_matrices[:, 0, 0] = self.properties[1] * self.properties[2] / self.beam_lengths
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
        self.beam_stiff_matrices[:, 3, 3] = self.properties[8] * self.properties[5] / self.beam_lengths
        self.beam_stiff_matrices[:, 9, 9] = self.beam_stiff_matrices[:, 3, 3]
        self.beam_stiff_matrices[:, 9, 3] = self.beam_stiff_matrices[:, 3, 9] = -self.beam_stiff_matrices[:, 3, 3]
        # k7 and multiples
        k7 = self.properties[1] * self.properties[3] / self.beam_lengths
        self.beam_stiff_matrices[:, 4, 4] = self.beam_stiff_matrices[:, 10, 10] = torch.mul(4, k7)
        self.beam_stiff_matrices[:, 10, 4] = self.beam_stiff_matrices[:, 4, 10] = torch.mul(2, k7)
        # k8 and multiples
        k8 = self.properties[1] * self.properties[4] / self.beam_lengths
        self.beam_stiff_matrices[:, 5, 5] = self.beam_stiff_matrices[:, 11, 11] = torch.mul(4, k8)
        self.beam_stiff_matrices[:, 11, 5] = self.beam_stiff_matrices[:, 5, 11] = torch.mul(2, k8)

        ###########################################################################################################
        #Assembling beam-local to global transition matrices via Kronecker product: container is again a 
        # 3-dimensional torch.tensor.
        self.transition_matrices = torch.kron(torch.eye(4, 4, device=self.device), self.beam_frames)

        ###########################################################################################################
        #Building beam contributions to global stiff matrix: container (beam_contributions) is a 3d torch.tensor, 
        #another contanier (self.beam_forces_contributions) for products beam_stiff_matrix @ transition_matrix
        #is saved in order not to repeat steps in node forces computation phase.
        #Please note: @ operator for 3d tensors produces a 'batched' 2d matrix multiplication along sections.
        self.beam_forces_contributions = self.beam_stiff_matrices @ self.transition_matrices
        beam_contributions = torch.transpose(self.transition_matrices, 1, 2) @ self.beam_forces_contributions
                                                #torch.transpose(_, 1, 2) transpose along dimensions 1 and 2

        ###########################################################################################################
        #Building global stiff matrix by adding all beam contributions.
        for idx, dofs in enumerate(self.endpoints_dofs_matrix):
            rows, columns = torch.meshgrid(dofs, dofs, indexing='ij')
            self.stiff_matrix[rows, columns] += beam_contributions[idx, :, :]

    #Compute vertex deformations by solving a stiff-matrix-based linear system.
    def compute_stiff_deformation(self):
        #Solving reduced linear system and adding zeros in constrained dofs.
        self.vertex_deformations = torch.zeros(len(self.mesh.vertices) * DOF, device=self.device)
        sys_sol = torch.linalg.solve(self.stiff_matrix[self.dofs_non_constrained_mask][:, self.dofs_non_constrained_mask], self.load[self.dofs_non_constrained_mask])
        self.vertex_deformations[self.dofs_non_constrained_mask] = sys_sol

        #Computing beam resulting forces and energies.
        self.compute_beam_force_and_energy()

        #Making deformation tensor by reshaping self.vertex_deformations.
        self.vertex_deformations = self.vertex_deformations.view(len(self.mesh.vertices), DOF)

    #Computing beam resulting forces and energies.
    def compute_beam_force_and_energy(self):
        #edge_dofs_deformations aggregates vertex_deformation in a edge-endpoints-wise manner: (#edges, 2*DOF) torch.tensor 
        edge_dofs_deformations = self.vertex_deformations[self.endpoints_dofs_matrix]

        #############################################################################################################################
        #Computing resulting forces at nodes via 'batched' matrix multiplication @.
        #Some details: 
        # edge_dofs_deformations.unsqueeze(2) expands (#edges, 2*DOF) to #edge batches of (2*DOF, 1), i.e. (#edges, 2*DOF, 1) tensor;
        # vice-versa, (...).squeeze(2) contracts (#edges, 2*DOF, 1) -> (#edges, 2*DOF)
        node_forces = (self.beam_forces_contributions @ edge_dofs_deformations.unsqueeze(2)).squeeze(2)

        #Averaging force components 0:DOF with D0F:2*DOF of opposite sign.
        mean_forces = torch.mul(0.5, node_forces[:, :DOF] - node_forces[:, DOF:2*DOF])

        #Computing beam energy according this coordinate system:
        #axes: 1=elementAxis; 2=EdgeNormal; 3=InPlaneAxis
        #output rows: [Axial_startNode; Shear2_startNode; Shear3_startNode; Torque_startNode; Bending3_startNode; Bending2_startNode;
        #                ...Axial_endNode; Shear2_endNode; Shear3_endNode; Torque_endNode; Bending3_endNode; Bending2_endNode]
        self.beam_energy = self.beam_lengths/2 * ( mean_forces[:, 0]**2/(self.properties[1]*self.properties[2]) + 
                            self.properties[6] * mean_forces[:, 1]**2 / (self.properties[8] * self.properties[2]) + 
                            self.properties[6] * mean_forces[:, 2]**2 / (self.properties[8] * self.properties[2]) + 
                            mean_forces[:, 3]**2 / (self.properties[8] * self.properties[5]) + 
                            mean_forces[:, 4]**2 / (self.properties[1] * self.properties[4]) + 
                            mean_forces[:, 5]**2 / (self.properties[1] * self.properties[3]) )

    def clean_attributes(self):
        self.mesh.vertices.detach_()
        self.beam_frames.detach_()
        self.beam_stiff_matrices.detach_()
        self.stiff_matrix.detach_()
        
    #Displace initial mesh with self.beam_model_solve() computed translations.
    def displace_mesh(self):
        #REQUIRES: self.beam_model_solve() has to be called before executing this.
        if not hasattr(self, 'vertex_deformations'):
            raise RuntimeError("self.beam_model_solve() method not called yet.")

        #Updating mesh vertices.
        self.mesh.update_verts(self.mesh.vertices + self.vertex_deformations[:, :int(DOF/2)])

    #Show displaced mesh via meshplot.
    def plot_grid_shell(self):
        colors = torch.norm(self.vertex_deformations[:, :3], p=2, dim=1)
        self.mesh.plot_mesh(colors)

# lc = LacconianCalculus(file='meshes/casestudy_compr.ply', device='cpu')
# lc.displace_mesh()
# lc.plot_grid_shell()
