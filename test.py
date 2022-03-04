from models.layers.featured_mesh import FeaturedMesh
from models.networks import DGCNNDisplacerNet

if __name__ == '__main__':
    device = 'cpu'
    fm = FeaturedMesh(file='/home/andrea/Scrivania/GridShellLearn-main/meshes/F_waveArch_hi.ply', device=device)
    fm.compute_mesh_input_features()
    in_channels = fm.input_features.shape[1]
    k = 16
    dgcnn = DGCNNDisplacerNet(in_channels, k, aggr='mean').to(device)
    out = dgcnn(fm.input_features)
    print(out)
