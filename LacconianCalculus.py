import numpy as np
import torch
from torch.nn.functional import normalize
from models.layers.mesh import Mesh
from collections import namedtuple
from collections.abc import Iterable
from utils import plot_mesh

DOF = 6     #degrees of freedom per vertex

class LacconianCalculus:
    
    def __init__(self, file=None, beam_properties=None, device='cpu'):
        self.device = torch.device(device)
        self.set_beam_properties(beam_properties)

        if file is not None:
            self.mesh = Mesh(file, device=device)
            self.set_beam_model_data()
            self.beam_model_solve()
            self.displace_mesh()
            self.plot_grid_shell()

    def __call__(self, mesh):
        self.mesh = mesh
        self.set_beam_model_data()
        self.beam_model_solve()
        return torch.sum(torch.norm(self.vertex_deformations, p=2, dim=1))

    #Store beam properties involved in the task.
    #Custom properties are passed through an iterable whose elements follow this order:
    #
    # -- Poisson's ratio, default is 0.3 (poisson);
    # -- Young's modulus, default is 2.1*10^8 kN/m^2 (young);
    # -- beam section area, default is 1*10^-3 m^2 (cross_area); 
    # -- moment of inertia2 Ixx = Iyy, default is 4.189828*10^-8, round section (inertia2);
    # -- moment of interia3 Ixx = Iyy, default is 4.189828*10^-8, round section (inertia3);
    # -- polar moment, default is 8.379656e-8 (polar);
    # -- shear section factor, default is 1.2 (shear);
    # -- weight per surface unit, default is 3 kN/m^2 (weight_per_surface).
    #
    def set_beam_properties(self, beam_properties):
        Properties = namedtuple('Properties', ['poisson', 'young', 'cross_area', 'inertia2', 'inertia3', 'polar', 'shear', 'weight_per_surface'])
        if beam_properties is None:
            self.properties = Properties(0.3, 21e7, 1e-2, 4.189828e-8, 4.189828e-8, 8.379656e-8, 1.2, -3)
        elif isinstance(beam_properties, Iterable):
            self.properties = Properties._make(beam_properties)
        else:
            raise ValueError('Passed a non-iterable beam properties container.')
        self.G = self.properties.young / (2 * (1+self.properties.poisson))

    #Stores load matrix, beam lengths, beam local frames.
    #Load matrix (no. vertices x DOF) has row-structure (Fx, Fy, Fz, Mx, My, Mz) whose values are referred to global ref system.
    def set_beam_model_data(self):
        #Computation of all mesh-based prerequisites.
        beam_directions, beam_lengths = self.mesh.compute_edge_lengths_and_directions()
        beam_normals = self.mesh.compute_edge_normals()
        
        #Saving load matrix: all entries are zero except Fz who is set -1/3 * weight_per_surface * <sum of all incident face areas>
        self.load = torch.zeros(len(self.mesh.vertices), DOF, device=self.device)
        for vertex_id in range(len(self.mesh.vertices)):
            ix_mask = self.mesh.incidence_mask[vertex_id, :]
            self.load[vertex_id, 2] = 1/3 * self.properties.weight_per_surface * torch.sum(self.mesh.face_areas[ix_mask], dim=0)

        #Saving beam lenghts.
        self.beam_lengths = beam_lengths

        #Saving beam local frames, i.e. bases of beam-fixed reference systems stored in a 3-dimensional tensor.
        self.beam_frames = torch.zeros(len(self.mesh.edges), 3, 3, device=self.device)

        self.beam_frames[:, 0, :] = beam_directions
        self.beam_frames[:, 1, :] = beam_normals
        self.beam_frames[:, 2, :] = normalize(torch.cross(beam_directions, beam_normals, dim=1), dim=1)
                                    # renormalization is to prevent numerical orthogonality loss

        #Deformation tensor initialization.
        self.vertex_deformations = None

    #Execute all stiffness and resistence computations.
    def beam_model_solve(self):
        self.build_stiff_matrix()
        self.compute_stiff_deformation()

    #Stiffness matrices in beam reference systems are computed and then aggregated to compound a global stiff matrix.
    def build_stiff_matrix(self):
        #Global stiff matrix container: 2-dimensional torch.tensor.
        self.stiff_matrix = torch.zeros(DOF*len(self.mesh.vertices), DOF*len(self.mesh.vertices), device=self.device)
        #Beam stiff matrices container: 3-dimensional torch.tensor.
        self.beam_stiff_matrices = torch.zeros(2*DOF, 2*DOF, len(self.mesh.edges), device=self.device)

        #Transition matrices from beam to general reference system container: 3-dimensional torch.tensor.
        self.transition_matrices = torch.zeros(2*DOF, 2*DOF, len(self.mesh.edges), device=self.device)

        for edge_id, edge in enumerate(self.mesh.edges):
            #Assembling local reference matrices.
            k1 = self.properties.young * self.properties.cross_area / self.beam_lengths[edge_id]
            k2 = 12 * self.properties.young * self.properties.inertia2 / (self.beam_lengths[edge_id]**3)
            k3 = 12 * self.properties.young * self.properties.inertia3 / (self.beam_lengths[edge_id]**3)
            k4 = 6 * self.properties.young * self.properties.inertia3 / (self.beam_lengths[edge_id]**2)
            k5 = 6 * self.properties.young * self.properties.inertia2 /(self.beam_lengths[edge_id]**2)
            k6 = self.G * self.properties.polar / self.beam_lengths[edge_id]
            k7 = self.properties.young * self.properties.inertia2 / self.beam_lengths[edge_id]
            k8 = self.properties.young * self.properties.inertia3 / self.beam_lengths[edge_id]
            k_e = [[k1,     0,	    0,      0,	    0,      0,     -k1,	    0,       0, 	0,  	 0,      0],
                    [0,    k3,	    0,      0,	    0,      k4,	     0,   -k3,       0, 	0,  	 0,     k4],
                    [0,      0,	   k2,      0,	  -k5,       0, 	 0,	    0,     -k2, 	0,     -k5,      0],
                    [0,      0,	    0,     k6,	    0,       0,	     0,	    0,       0,   -k6,  	 0,      0],
                    [0,      0,    -k5,      0,	 4*k7,       0, 	 0,	    0,      k5, 	0,	  2*k7, 	 0],         
                    [0,     k4,      0,      0,	    0,    4*k8,	     0,   -k4,       0, 	0,	     0,   2*k8],         
                    [-k1,	0,	    0,      0,	    0,       0, 	k1,	    0,       0, 	0,	     0,      0],
                    [0,    -k3,	    0,      0,	    0,     -k4, 	 0,	   k3,       0, 	0,	     0,    -k4],
                    [0,      0,	  -k2,      0,     k5,       0,	     0,	    0,      k2,	    0,  	k4,      0],
                    [0,      0,	    0,    -k6,	    0,       0,	     0,	    0,       0,	   k6,	     0,      0],
                    [0,      0,	  -k5,      0,	 2*k7,	     0,	     0,	    0,      k5,	    0,    4*k7,	     0],
                    [0,     k4,	    0,      0,	    0,    2*k8,	     0,	  -k4,       0,     0,  	 0,	  4*k8]]
            k_e = torch.tensor(k_e, device=self.device)
            self.beam_stiff_matrices[:, :, edge_id] = k_e

            #Assembling local to global reference transition matrices (via Kronecker product).
            edge_trans_matrix = torch.kron(torch.eye(4, 4, device=self.device), self.beam_frames[edge_id, :, :])
            self.transition_matrices[:, :, edge_id] = edge_trans_matrix

            #Adding edge contribution to global stiff matrix.
            edge_dofs = torch.cat((6 * edge[0] + torch.arange(6), 6 * edge[1] + torch.arange(6)), dim=0)
            ix_grid = np.ix_(edge_dofs, edge_dofs)
            self.stiff_matrix[ix_grid] += edge_trans_matrix.T @ k_e @ edge_trans_matrix

    #Compute vertex deformations by solving a stiff-matrix-based linear system.
    def compute_stiff_deformation(self):
        #Vectorializing load matrix
        loads = torch.reshape(self.load, shape=(DOF * len(self.mesh.vertices), ))

        #Building non-constrainted vertex dofs mask, again, using Kronecker product.
        non_constrained_vertices = self.mesh.vertex_is_constrainted.logical_not()
        ix_mask = torch.kron(non_constrained_vertices, torch.tensor([True] * DOF))
        ix_grid = np.ix_(ix_mask, ix_mask)

        #Solving reduced linear system and adding zeros in constrained dofs.
        self.vertex_deformations = torch.zeros(len(self.mesh.vertices) * DOF, device=self.device)
        sys_sol = torch.linalg.solve(self.stiff_matrix[ix_grid], loads[ix_mask])
        self.vertex_deformations[ix_mask] = sys_sol

        #Computing beam resulting forces and energies.
        self.compute_beam_force_and_energy()

        #Making deformation tensor by reshaping self.vertex_deformations.
        self.vertex_deformations = torch.reshape(self.vertex_deformations, shape=(len(self.mesh.vertices), DOF))

    #Computing beam resulting forces and energies.
    def compute_beam_force_and_energy(self):
        #axes: 1=elementAxis; 2=EdgeNormal; 3=InPlaneAxis
        #output rows: [Axial_startNode; Shear2_startNode; Shear3_startNode; Torque_startNode; Bending3_startNode; Bending2_startNode;
        #                ...Axial_endNode; Shear2_endNode; Shear3_endNode; Torque_endNode; Bending3_endNode; Bending2_endNode]
        #mean values of the force is reported for each edge
        self.signed_node_forces = torch.zeros(len(self.mesh.edges), 2 * DOF, device=self.device)
        self.mean_forces = torch.zeros(len(self.mesh.edges), DOF, device=self.device)
        self.beam_energy = torch.zeros(len(self.mesh.edges), device=self.device)

        for edge_id, edge in enumerate(self.mesh.edges):
            #Selecting edge endpoint displacements.
            edge_dofs = torch.cat((6 * edge[0] + torch.arange(6), 6 * edge[1] + torch.arange(6)), dim=0)
            disp = self.vertex_deformations[edge_dofs]

            #Computing and averaging resulting forces per beam.
            node_forces = self.beam_stiff_matrices[:, :, edge_id] @ self.transition_matrices[:, :, edge_id] @ disp
            self.signed_node_forces[edge_id, :] =  torch.cat((node_forces[0 : DOF], -node_forces[DOF : 2*DOF]), axis=0) 
                                                                        #in the node reference system with modified sign
            self.mean_forces[edge_id, :] = torch.mean(torch.reshape(self.signed_node_forces[edge_id, :], shape=(2, DOF)), axis=0)

            #Computing beam energy.
            self.beam_energy[edge_id] = self.beam_lengths[edge_id]/2 * ( self.mean_forces[edge_id, 0]**2/(self.properties.young*self.properties.cross_area) + 
                                        self.properties.shear * self.mean_forces[edge_id, 1]**2 / (self.G * self.properties.cross_area) + 
                                        self.properties.shear * self.mean_forces[edge_id, 2]**2/ (self.G * self.properties.cross_area) + 
                                        self.mean_forces[edge_id, 3]**2 / (self.G * self.properties.polar) + 
                                        self.mean_forces[edge_id, 4]**2 / (self.properties.young * self.properties.inertia3) + 
                                        self.mean_forces[edge_id, 5]**2/ (self.properties.young * self.properties.inertia2) )
            
    #Displace initial mesh with self.beam_model_solve() computed translations.
    def displace_mesh(self):
        #REQUIRES: self.beam_model_solve() has to be called before executing this.
        if self.vertex_deformations is None:
            raise RuntimeError("self.beam_model_solve() method not called yet.")

        #Updating mesh vertices.
        self.mesh.update_verts(self.mesh.vertices + self.vertex_deformations[:, :int(DOF/2)])

    #Show displaced mesh via meshplot.
    def plot_grid_shell(self):
        colors = torch.norm(self.vertex_deformations[:, :2], p=2, dim=1)
        plot_mesh(self.mesh.vertices, self.mesh.faces, colors)

#lc = LacconianCalculus(file='meshes/nonfuniculartraslationalsrf2.ply')


