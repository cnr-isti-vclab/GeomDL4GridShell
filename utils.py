import numpy as np
import pymeshlab
import torch

def load_mesh(path):
    # Creating pymeshlab MeshSet, loading mesh from file and selecting it.
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()

    # Selecting vertex and face matrices.
    vertices = np.float32(mesh.vertex_matrix())
    faces = mesh.face_matrix()

    # Deducing vertex constraintness from red or blue coloring.
    colors = mesh.vertex_color_matrix()[:, :3]
    vertex_is_red = np.where((colors[:,0] == 1) & (colors[:,1] == 0) & (colors[:,2] == 0), True, False)
    vertex_is_blue = np.where((colors[:,0] == 0) & (colors[:,1] == 0) & (colors[:,2] == 1), True, False)

    # Getting boundary vertices mask.
    ms.select_border()
    vertex_is_on_boundary = mesh.vertex_selection_array()
   
    return vertices, faces, vertex_is_red, vertex_is_blue, vertex_is_on_boundary

def save_mesh(mesh, filename, v_quality=np.array([], dtype=np.float64)):
    # Changing torch.tensors to np.arrays.
    vertices = np.float64(mesh.vertices.detach().cpu().numpy())
    faces = mesh.faces.detach().cpu().numpy()
    if type(v_quality) is torch.Tensor:
        v_quality = np.float64(v_quality.detach().cpu().numpy())

    # Creating vertex_color_matrix from vertex_is_red, vertex_is_blue.
    colors = np.zeros((vertices.shape[0], 4))
    for idx, red_vertex in enumerate(mesh.vertex_is_red):
        if red_vertex:
            colors[idx, :] = np.array([1., 0., 0., 1.])
        else:
            colors[idx, :] = np.array([0.75294118, 0.75294118, 0.75294118, 1])
    for idx, blue_vertex in enumerate(mesh.vertex_is_blue):
        if blue_vertex:
            colors[idx, :] = np.array([0., 0., 1., 1.])

    # Creating pymeshlab MeshSet and adding mesh.
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces, v_color_matrix=colors, v_quality_array=v_quality)
    ms.add_mesh(mesh, set_as_current=True)

    # Saving mesh on filename.
    ms.save_current_mesh(filename)

def isotrophic_remesh(mesh, filename, target_length):
    # Changing torch.tensors to np.arrays.
    vertices = np.float64(mesh.vertices.detach().cpu().numpy())
    faces = mesh.faces.detach().cpu().numpy()

    # Creating vertex_color_matrix from vertex_is_red, vertex_is_blue.
    colors = np.zeros((vertices.shape[0], 4))
    for idx, red_vertex in enumerate(mesh.vertex_is_red):
        if red_vertex:
            colors[idx, :] = np.array([1., 0., 0., 1.])
        else:
            colors[idx, :] = np.array([0.75294118, 0.75294118, 0.75294118, 1])
    for idx, blue_vertex in enumerate(mesh.vertex_is_blue):
        if blue_vertex:
            colors[idx, :] = np.array([0., 0., 1., 1.])

    # Creating pymeshlab MeshSet, adding mesh.
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces, v_color_matrix=colors)
    ms.add_mesh(mesh, set_as_current=True)
    
    # Getting target_lenght percentage.
    mesh.update_bounding_box()
    bb_diagonal = mesh.bounding_box().diagonal()
    target_length_perc = pymeshlab.Percentage(target_length * 100 / bb_diagonal)

    # Applying isotrophic remeshing.
    ms.remeshing_isotropic_explicit_remeshing(iterations=20, adaptive=True, targetlen=target_length_perc)

    # Saving mesh on filename
    ms.save_current_mesh(filename)

def save_cloud(points, filename):
    # Changing torch.tensors to np.arrays.
    points = np.float64(points.detach().cpu().numpy())

    # Creating pymeshlab MeshSet and adding cloud "mesh".
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=points)
    ms.add_mesh(mesh, set_as_current=True)

    # Saving mesh on filename.
    ms.save_current_mesh(filename)

def edge_connectivity(face_matrix):
    # Initializing edge lists.
    edge_list = []
    edge_per_face = []

    for face in face_matrix:
        current_face_list = []

        for i in range(3):
            edge = sorted([face[i], face[(i + 1) % 3]])
            if edge not in edge_list:
                current_face_list.append(len(edge_list))
                edge_list.append(edge)
            else:
                current_face_list.append(edge_list.index(edge))
                
        edge_per_face.append(current_face_list)

    return np.array(edge_list), np.array(edge_per_face)
