from models.layers.featured_mesh import FeaturedMesh
import torch

if __name__ == '__main__':
    fm = FeaturedMesh(file='/home/andrea/Scrivania/GridShellLearn-main/meshes/quinta2.ply')
    fm.compute_mesh_input_features()