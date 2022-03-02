from models.layers.featured_mesh import FeaturedMesh
from models.layers.dgcnn_layer import DGCNNLayer

if __name__ == '__main__':
    fm = FeaturedMesh(file='/home/andrea/Scrivania/GridShellLearn-main/meshes/quinta2.ply')
    fm.compute_mesh_input_features()
    in_channels = fm.input_features.shape[1]
    out_channels = 1024
    k = 100
    dgcnn = DGCNNLayer(in_channels, out_channels, k)
    print(dgcnn(fm.input_features))
