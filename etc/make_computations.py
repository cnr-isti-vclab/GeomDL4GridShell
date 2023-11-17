import os, sys, torch
from LacconianCalculus import LacconianCalculus
from utils import save_mesh, export_vector, map_to_color_space

# Making output directory.
if not os.path.exists('output'):
    os.mkdir('output')
    print("Directory output created.")
else:    
    print("Directory output already exists.")

input_root = 'input/'
output_root = 'output/'
label = sys.argv[1]
initial_mesh_name = label + '.ply'
initial_path = input_root + initial_mesh_name


lc_initial = LacconianCalculus(file=initial_path, device='cpu')
vmax = 10 * torch.mean(lc_initial.beam_energy)

# Saving edge topology.
export_vector(lc_initial.initial_mesh.edges, output_root + 'edges.csv')

# Computing deflections.
save_path = output_root + 'deflections_' + initial_mesh_name
deflections = torch.norm(lc_initial.vertex_deformations[:, :3], p=2, dim=1)
save_mesh(lc_initial.initial_mesh, save_path, v_quality=deflections)
save_path = output_root + 'deflections_' + initial_mesh_name[ :-4] + '.csv'
export_vector(deflections.detach().cpu(), save_path)

# Computing energy.
save_path = output_root + '[RGBA]energy_' + initial_mesh_name[ :-4] + '.csv' 
export_vector(map_to_color_space(lc_initial.beam_energy.detach().cpu(), vmin=0, vmax=vmax), save_path, format='%d')
save_path = output_root + 'energy_' + initial_mesh_name[ :-4] + '.csv'
export_vector(lc_initial.beam_energy.detach().cpu(), save_path)


for mesh_prefix in ['model_', 'ff_karamba_en_', 'ff_karamba_disp_', 'ff_kangaroo_']:

    current_mesh = input_root + mesh_prefix + initial_mesh_name
    lc_current = LacconianCalculus(file=current_mesh, device='cpu')

    # Computing displacements.
    displacements = torch.norm(lc_current.initial_mesh.vertices - lc_initial.initial_mesh.vertices, dim=1, p=2)
    save_path = output_root + 'displacements_' + mesh_prefix + initial_mesh_name
    save_mesh(lc_current.initial_mesh, save_path, v_quality=displacements)

    # Computing deflections.
    save_path = output_root + 'deflections_' + mesh_prefix + initial_mesh_name
    deflections = torch.norm(lc_current.vertex_deformations[:, :3], p=2, dim=1)
    save_mesh(lc_current.initial_mesh, save_path, v_quality=deflections)
    save_path = output_root + 'deflections_' + mesh_prefix + initial_mesh_name[ :-4] + '.csv'
    export_vector(deflections.detach().cpu(), save_path)

    # Computing energy.
    save_path = output_root + '[RGBA]energy_' + mesh_prefix + initial_mesh_name[ :-4] + '.csv' 
    export_vector(map_to_color_space(lc_current.beam_energy.detach().cpu(), vmin=0, vmax=vmax), save_path, format='%d')
    save_path = output_root + 'energy_' + mesh_prefix + initial_mesh_name[ :-4] + '.csv'
    export_vector(lc_current.beam_energy.detach().cpu(), save_path)