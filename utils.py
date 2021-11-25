import numpy as np
import polyscope as ps
from plyfile import PlyData

def load_ply(path):
    plyData = PlyData.read(path)
    x, y, z = (np.array([plyData['vertex']['x']]).T, np.array([plyData['vertex']['y']]).T, np.array([plyData['vertex']['z']]).T)

    #Extracting and saving features.
    vertices = np.concatenate((x,y,z),axis=1)
    faces = np.array([face for face in plyData['face']['vertex_indices']])
    vertex_is_constrained = np.where(plyData['vertex']['flags'] == 32, True, False)

    return vertices, faces, vertex_is_constrained

def plot_mesh(v, f, color=None):
    if type(v) is not np.ndarray:
            v = v.cpu().numpy()
    if type(f) is not np.ndarray:
            f = f.cpu().numpy()
    if type(color) is not np.ndarray:
            color = color.cpu().numpy()

    ps.init()

    #Setting camera position.
    v0 = np.mean(v[:,0])
    v1 = np.mean(v[:,1])
    ps.look_at((0,0,3*np.max(v)),(v0,v1,0))
    ps.set_ground_plane_mode('none')
    ps.set_transparency_mode('pretty')
    ps.register_surface_mesh('mesh', v, f, smooth_shade=True, edge_width=1, transparency=0.9, edge_color=(0,0,0))

    if color is not None:
        ps.get_surface_mesh("mesh").add_scalar_quantity("color", color, defined_on='vertices', vminmax=(np.min(color),
        np.max(color)), enabled=True, cmap='coolwarm')

    ps.show()