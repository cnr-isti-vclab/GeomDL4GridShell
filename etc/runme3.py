import pymeshlab
import os
from numpy.linalg import norm

f = open('max_displacements.txt', 'w')

for start_mesh in os.listdir('input'):
    path = 'input/' + start_mesh
    meshname = start_mesh[ :-4]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    mesh = ms.current_mesh()
    start_vertices = mesh.vertex_matrix()

    for end_mesh in os.listdir('output'):
        if meshname in end_mesh:
            path = 'output/' + end_mesh
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(path)
            mesh = ms.current_mesh()
            end_vertices = mesh.vertex_matrix()
    
    f.write(str(start_mesh) + ' ' + str(max(norm(end_vertices - start_vertices, ord=2, axis=1))) + '\n')

f.close()                                                                                                                                                                                                                                                             