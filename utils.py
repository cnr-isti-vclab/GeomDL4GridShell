import numpy as np
import meshplot as mp
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

    mp.offline()
    if color is not None:
        mp.plot(v, f, color)
    else:
        mp.plot(v, f)