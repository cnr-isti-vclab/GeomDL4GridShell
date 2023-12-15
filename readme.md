# Geometric deep learning for statics aware grid shells

**Andrea Favilli<sup>a,b</sup>, Francesco Laccone<sup>a</sup>, Paolo Cignoni<sup>a</sup>, Luigi Malomo<sup>a</sup>, Daniela Giorgi<sup>a</sup>**  
<sup>a</sup>Institute of Information Science and Technologies "A. Faedo" (ISTI), National Research Council of Italy (CNR)  
<sup>b</sup>University of Pisa (Italy)
Paper: [link](https://doi.org/10.1016/j.compstruc.2023.107238)

![image](./images/teaser.png)

# Installation
This code has been tested on Windows® 10 Pro and Ubuntu 22.04. The Windows® machine has an Intel i7-6700K CPU, 32 GB of RAM, a NVIDIA GeForce GTX 1080 GPU with 8 GB of dedicated memory. The Ubuntu machine has an AMD Epyc 7413 CPU, 128 GB of RAM, a NVIDIA GeForce RTX 3080 GPU with 10 GB of dedicated memory. The code runs on Python 3.11.5 with PyTorch 2.1.1, CUDA 11.8, and PyTorch Geometric 2.4.0.

### Installing dependencies on Anaconda
We employed [Anaconda](https://www.anaconda.com/products/distribution), a popular Python distribution for data science and machine learning. After that Anaconda is installed, we can use an Anaconda shell to create virtual environments and run the code. From an Anaconda prompt, we move to the repository root directory and enter the command
~~~
conda env create --file environment.yml
~~~
to create an envirorment named ```GeomDL4GridShell``` that contains all the needed dependencies. We can then activate ```GeomDL4GridShell``` by typing
~~~
conda activate GeomDL4GridShell
~~~
To ensure that CUDA 11.8 works correctly, check for latest NVIDIA card driver update. Now we are ready to run the code.

# Code usage
Into the environment ```GeomDL4GridShell```, the command
~~~
python optimization_task.py --meshpath 'meshes/<modelname>.ply' --device 'cuda' --savelabel <modelname>
~~~
performs shape optimization on a single input structure, encoded in the file ```<modelname>.ply```. By omitting ```--device 'cuda'``` (or writing ```--device 'cpu'```) we make computations run on CPU instead of GPU.

Execution on the whole batch of examples ```models/``` is performed using the command
~~~
python batch_exec.py
~~~
We can set batch execution on CPU by setting ```device = 'cpu'``` on line 21 of ```batch_exec.py```.
