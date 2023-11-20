import os, subprocess

print("List of input structures:")
mesh_list = os.listdir('meshes')
print(mesh_list)

# Making output directory.
if not os.path.exists('output'):
    os.mkdir('output')
    print("Directory output created.")
else:    
    print("Directory output already exists.")

print("**Starting Shape Optimization***")

# Computing neural-based shape optimization for each .ply file in the meshes folder
for mesh in mesh_list:
    command = 'python'
    scriptfile = 'optimization_task.py'
    meshpath = 'meshes/' + mesh
    device = 'cuda'
    savelabel = mesh[:len(mesh)-4]
    saveprefix = 'output/'
    subprocess.run(args=[command, scriptfile, '--meshpath', meshpath, '--device', device, '--savelabel', savelabel, '--saveprefix', saveprefix])