# Spherical U-Net on Cortical Surfaces
This is the code for paper ["Spherical U-Net on Cortical Surfaces: Methods and Applications"](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_67). The code in the main directory is for the parcellation task. The code in the [prediction](https://github.com/zhaofenqiang/Spherical_U-Net/tree/master/prediction) directory is for the prediction task.

## Data structure
The brain cortical surface lie in a non-Euclidean space represented by triangular meshes.
![CorticalSurface](https://raw.githubusercontent.com/zhaofenqiang/Spherical_U-Net/master/images/figure_OrigSurf_SphereSurf.png) ![IcosahedronDiscretizedSphere](https://raw.githubusercontent.com/zhaofenqiang/Spherical_U-Net/master/images/figure_12-10242_spherical_surfaces.png) 

## 1-ring filter on sphere
The convolution on the spherical surface is performed baed on the 1-ring filter
![convoulution](https://raw.githubusercontent.com/zhaofenqiang/Spherical_U-Net/master/images/figure_convolution.png).

We provide 3 types of filter on the spherical surfaces.
![filters](https://raw.githubusercontent.com/zhaofenqiang/Spherical_U-Net/master/images/figure_filters.png).

## Spherical U-Net architecture
![sphericaluent](https://raw.githubusercontent.com/zhaofenqiang/Spherical_U-Net/master/images/figure_unet.png).


## How to use it
### Prerequisites
- Linux
- NVIDIA GPU
- CUDA CuDNN

### Python Dependencies
- python (3.6)
- pytorch (0.4.1+)
- torchvision (0.2.1+)
- tensorboardx (1.6+)

### Matlab Dependencies
- mvtk_read
- mvtk_write

Setup a new conda environment with the required dependencies via:
```
conda create -n sunet python=3.6 
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
``` 
Activate newly created conda environment via:
```
conda activate sunet
```

### Data preparation
Modify the script [extract_feats.m](https://github.com/zhaofenqiang/Spherical_U-Net/blob/master/matlab_script_for_extracting_data/extract_feats.m) in [matlab_script_for_extracting_data](https://github.com/zhaofenqiang/Spherical_U-Net/tree/master/matlab_script_for_extracting_data) to extract surface data in .vtk to .mat. For example, the surface with N vertices is extracted to NxD feature in .mat, where D is the input channels of the Spherical U-Net. D is typically 3 for parcellation task, representing mean curvature, average convexity, and sulc depth.

### Train
After data prepration, modify the folder in [train.py](https://github.com/zhaofenqiang/Spherical_U-Net/blob/master/train.py) to match the training data in your own path. Then, run:
```
python train.py
```

### Test
Modify the test data folder in [test.py](https://github.com/zhaofenqiang/Spherical_U-Net/blob/master/test.py). Then, run:
```
python test.py
```
The output is in .txt. Then modify the [write_prdicted_sphere.m](https://github.com/zhaofenqiang/Spherical_U-Net/blob/master/matlab_script_for_extracting_data/write_prdicted_sphere.m) to write the .txt to .vtk

## Cite
If you use this code for your research, please cite as:

Fenqiang Zhao, et.al. Spherical U-Net on Cortical Surfaces: Methods and Applications. Information Processing in Medical Imaging (IPMI), 2019.

