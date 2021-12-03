import numpy as np
import polyscope as ps
import pymeshlab

def load_mesh(path):
    #Creating pymeshlab MeshSet, loading mesh from file and selecting it.
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()

    #Selecting vertex and face matrices.
    vertices = np.float32(mesh.vertex_matrix())
    faces = mesh.face_matrix()

    #Deducing vertex constraintness from red coloring.
    colors = mesh.vertex_color_matrix()[:, :3]
    vertex_is_constrained = np.where((colors[:,0] == 1) & (colors[:,1] == 0) & (colors[:,2] == 0), True, False)
   
    return vertices, faces, vertex_is_constrained

def save_mesh(mesh, filename):
    #Changing torch.tensors to np.arrays.
    vertices = np.float64(mesh.vertices.detach().cpu().numpy())
    faces = mesh.faces.detach().cpu().numpy()

    #Creating vertex_color_matrix from vertex_is_constrainted.
    colors = np.array([ [1., 0., 0., 1] if constr else [0.75294118, 0.75294118, 0.75294118, 1]
                        for constr in mesh.vertex_is_constrainted ])

    #Creating pymeshlab MeshSet and adding mesh.
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces, v_color_matrix=colors)
    ms.add_mesh(mesh, set_as_current=True)

    #Saving mesh on filename.
    ms.save_current_mesh(filename)

def plot_mesh(v, f, colors=None, cmap='viridis'):
    if type(v) is not np.ndarray:
            v = v.cpu().numpy()
    if type(f) is not np.ndarray:
            f = f.cpu().numpy()
    if type(colors) is not np.ndarray:
            if colors is not None:
                colors = colors.cpu().numpy()

    ps.init()

    #Setting camera position.
    v0 = np.mean(v[:,0])
    v1 = np.mean(v[:,1])
    ps.look_at((0,0,3*np.max(v)),(v0,v1,0))

    ps.set_ground_plane_mode('none')
    ps.set_transparency_mode('pretty')
    ps.register_surface_mesh('mesh', v, f, smooth_shade=True, edge_width=1, edge_color=(0,0,0))

    if colors is not None:
        ps.get_surface_mesh("mesh").add_scalar_quantity("color", colors, defined_on='vertices', enabled=True, cmap=cmap)

    ps.show()

def get_edge_matrix(face_matrix):
    #Initializing edge list.
    edge_list = []

    for face in face_matrix:
        for i in range(3):
            edge = sorted([face[i], face[(i + 1) % 3]])
            if edge not in edge_list:
                edge_list.append(edge)

    return np.array(edge_list)
