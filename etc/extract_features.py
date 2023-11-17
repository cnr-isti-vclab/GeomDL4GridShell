from models.layers.featured_mesh import FeaturedMesh
from utils import save_mesh

mesh_path = 'input/station.ply'

fm = FeaturedMesh(file=mesh_path, device='cpu')
fm.compute_mesh_input_features()

# Saving mesh with principal curvatures (geometric features).
save_path_k1 = mesh_path[ :-4] + '_k1.ply'
save_mesh(fm, save_path_k1, v_quality=fm.input_features[:, 6].unsqueeze(1))

save_path_k2 = mesh_path[ :-4] + '_k2.ply'
save_mesh(fm, save_path_k2, v_quality=fm.input_features[:, 7].unsqueeze(1))

# Saving mesh with geodesic features.
save_path_geod1 = mesh_path[ :-4] + '_geod1.ply'
save_mesh(fm, save_path_geod1, v_quality=fm.input_features[:, 8].unsqueeze(1))

save_path_geod2 = mesh_path[ :-4] + '_geod2.ply'
save_mesh(fm, save_path_geod2, v_quality=fm.input_features[:, 9].unsqueeze(1))

save_path_geod3 = mesh_path[ :-4] + '_geod3.ply'
save_mesh(fm, save_path_geod3, v_quality=fm.input_features[:, 10].unsqueeze(1))

save_path_geod4 = mesh_path[ :-4] + '_geod4.ply'
save_mesh(fm, save_path_geod4, v_quality=fm.input_features[:, 11].unsqueeze(1))
