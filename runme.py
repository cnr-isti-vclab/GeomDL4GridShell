import os, subprocess

mesh_list = os.listdir('input')
print(mesh_list)

# Making output directory.
if not os.path.exists('output'):
    os.mkdir('output')
    print("Directory output created.")
else:    
    print("Directory output already exists.")

for mesh in mesh_list:
    meshpath = 'input/' + mesh
    command = 'python'
    scriptfile = 'LacconianNetOptimizer.py'
    device = '--device'
    devicename = 'cuda'
    meshpath = '--meshpath' 
    meshpathname = 'input/' + mesh
    lr = '--lr'
    lrname = '0.0005'
    niter = '--niter'
    nitername = '1500'
    momentum = '--momentum' 
    momentumname = '0.9' 
    losstype = '--losstype' 
    losstypename = 'mean_beam_energy' 
    save = '--save' 
    saveinterval = '--saveinterval'
    saveintervalname = '20' 
    layermode = '--layermode' 
    layermodename = 'gat'
    transforminputfeatures = '--transforminputfeatures' 
    savelabel = '--savelabel' 
    savelabelname = mesh[:len(mesh)-4]
    itertimes = '--itertimes' 
    getloss = '--getloss' 
    saveprefix = '--saveprefix' 
    saveprefixname = 'output/'
    subprocess.run(args=[command, scriptfile, device, devicename, meshpath, meshpathname, lr, lrname, niter, nitername, momentum, momentumname, losstype, losstypename, save, saveinterval, saveintervalname, layermode, layermodename, transforminputfeatures, savelabel, savelabelname, itertimes, getloss, saveprefix, saveprefixname])