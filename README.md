# Spherical U-Net on Cortical Surfaces
This is the code for paper ["Spherical U-Net on Cortical Surfaces: Methods and Applications"](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_67). The code in the main directory is for the parcellation task. The code in the [prediction](https://github.com/zhaofenqiang/Spherical_U-Net/tree/master/prediction) directory is for the prediction task.

## Data structure
In medical imaging, there are a lot of structures that lie in a non-Euclidean space represented by triangular meshes. For example the brain cortical surface shown in the figure.
![CorticalSurface](https://raw.githubusercontent.com/zhaofenqiang/Spherical_U-Net/images/.jpg)  
We provide the 3 types of filter on the spherical surfaces. The 1-ring filter, 2-ring filter and rectangular patch filter. 
