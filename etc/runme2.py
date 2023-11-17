import os
import torch
from LacconianCalculus import LacconianCalculus

# Making output directory.
if not os.path.exists('myres_out'):
    os.mkdir('myres_out')
    print("Directory myres_out created.")
else:    
    print("Directory myres_out already exists.")

for mesh in os.listdir('output'):
    try:
        lc = LacconianCalculus(file='output/' + mesh, device='cuda')
        model_strain_energy = float(torch.mean(lc.beam_energy))
    except:
        model_strain_energy = 'sing_matrix'
    meshname = mesh.split('_')[0]

    for ff_mesh in os.listdir('myres'):
        if meshname in ff_mesh:
            try:
                lc = LacconianCalculus(file='myres/' + ff_mesh, device='cuda')
                ff_strain_energy = float(torch.mean(lc.beam_energy))
            except:
                ff_strain_energy = 'sing_matrix'

            filename = ff_mesh[ :-4] + '.txt'
            f = open('myres_out/' + filename, 'w')
            f.write('model_strain_energy vs ff_strain_energy\n')
            f.write(str(model_strain_energy) + '  ' + str(ff_strain_energy))
            f.close()

