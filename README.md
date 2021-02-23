# Spherical U-Net on Cortical Surfaces
This is the code for paper ["Spherical U-Net on Cortical Surfaces: Methods and Applications"](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_67). The code in the main directory is for the parcellation task. The code in the [prediction](https://github.com/zhaofenqiang/Spherical_U-Net/tree/master/prediction) directory is for the prediction task.

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
- pyvista (0.22.4+)

You can use conda to easily create an environment for the experiment using following command:
```
conda create -n sunet python=3.6 
conda activate sunet
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c conda-forge pyvista
```

### Data preparation
The input file is a cortical inner surface of one hemisphere in vtk format reconstructed from neuroimaging pipelines [[1]](https://www.sciencedirect.com/science/article/pii/S1361841515000559)[[2]](https://www.sciencedirect.com/science/article/pii/S1053811912000389), which has been resampled as either 40,962 or 163,842 vertices. Two features, i.e., mean curvature and average convexity, are required for the parcellation, denoted as “curv” and “sulc” field attributes in the vtk file, respectively. For resampling and feature computing, FreeSurfer [[2]](https://www.sciencedirect.com/science/article/pii/S1053811912000389) can be used. To be consistent with the trained model, gyral crests should have negative curvature values, while sulcal bottoms should have positive curvature values.

### Train
After data prepration, modify the [train.py](https://github.com/zhaofenqiang/Spherical_U-Net/blob/master/train.py) file to match the training data in your own path. Then, run:
```
python train.py
```

### Test
You can easily obtain the output parcellation maps on your surfaces via the following commands.
To predict a single surface’ parcellation map:
```
python predict.py -hemi left -l 7 -i input.vtk -o output.vtk
```
To predict the parcellation maps of multiple surfaces in the same folder:
```
python predict.py -hemi left -l 7 -in_folder in_folder -out_folder out_folder
```
You can also view the help information of the whole usage of this command by running 
```
python predict.py -h
```
```
Usage: predict.py [-h] [--hemisphere {left,right}] [--level {7,8}]
                  [--input INPUT] [--in_folder INPUT_FOLDER] [--output OUTPUT] [--out_folder OUT_FOLDER]

Predict the parcellation maps with 36 regions from the input surfaces

optional arguments:
  -h, --help            show this help message and exit
  --hemisphere {left,right}, -hemi {left,right}
                        Specify the hemisphere for parcellation, left or
                        right. (default: left)
  --level {7,8}, -l {7,8}
                        Specify the level of the surfaces. Generally, level
                        7 spherical surface is with 40962 vertices, 8 is with
                        163842 vertices. (default: 7)
  --input INPUT, -i INPUT
                        filename of input surface (default: None)
  --in_folder INPUT_FOLDER, -in_folder INPUT_FOLDER
                        folder path for input files. Will parcelalte all the
                        files end in .vtk in this folder. Accept input or                        
                        in_folder. (default: None)
  --output OUTPUT, -o OUTPUT
                        Filename of ouput surface. (default: [input].parc.vtk)
  --out_folder OUT_FOLDER, -out_folder OUT_FOLDER
                        folder path for ouput surface. Accept output or
                        out_folder. (default: [in_folder])
```
Troubleshoot notes:
1. The code requires `input` or `in_folder` option, not both, for single surface’ parcellation or all surfaces in the folder. 
2. The input data should be end in .vtk.

### Examples
You can test the code using the example surfaces we provided in the `examples` folder. Simply run:
```
python predict.py -hemi left -l 7 -i examples/left_hemisphere/40962/test1.lh.40k.vtk
```
You will get the corresponding output surface at the same folder with name `test1.lh.40k.parc.vtk`.
Or, run the command for all the 3 surface in the same folder:
```
python predict.py -hemi left -l 7 -in_folder examples/left_hemisphere/40962
```

### Visualization
You can use [Paraview](https://www.paraview.org/) software to visualize the parcellated surface in VTK format. An example of the input curvature map and output parcellation map are shown below. More usages about Paraview please refer to [Paraview](https://www.paraview.org/).
![paraview](https://raw.githubusercontent.com/zhaofenqiang/Spherical_U-Net/master/images/paraview.png).


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

## Cite
If you use this code for your research, please cite as:

Fenqiang Zhao, et.al. Spherical U-Net on Cortical Surfaces: Methods and Applications. Information Processing in Medical Imaging (IPMI), 2019.
Fenqiang Zhao, et.al. Spherical Deformable U-Net: Application to Cortical Surface Parcellation and Development Prediction. IEEE Transactions on Medical Imaging, 2021.

