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
- pyvista (0.22.4+)

You can use conda to easily create an environment for the experiment using following command:
```
conda create -n sunet python=3.6 
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install -c conda-forge pyvista
```
Activate the newly created conda environment via:
```
conda activate sunet
```


### Data preparation
The input file is a **vtk** file containing the surfaces reconstructed from neuroimaging pipelines [[1]](https://www.sciencedirect.com/science/article/pii/S1361841515000559)[[2]](https://www.sciencedirect.com/science/article/pii/S1053811912000389). After reconstruction of cortical surface, it is required to use Freesurfer [[2]](https://www.sciencedirect.com/science/article/pii/S1053811912000389) to map inner cortical surface to spherical surface and further resample it with 40,962 vertices. Also, you may need to covert the surface from Freesurfer format to vtk format. In the vtk file, **curv** and **sulc** field attributes data are required for the parcellation, which represent mean curvature and average convexity, respectively. 

### Train
After data prepration, modify the [train.py](https://github.com/zhaofenqiang/Spherical_U-Net/blob/master/train.py) file to match the training data in your own path. Then, run:
```
python train.py
```

### Test
You can easily obtain the output parcellation maps on your surfaces via the following commands.
To predict a single surface’ parcellation map:
```
python predict.py -i input.vtk -o output.vtk
```
To predict the parcellation maps of multiple surfaces in the same folder:
```
python predict.py -in_folder -o out_folder.vtk
```
You can also view the help of the whole usage of this command by running 
```
python predict.py -h
```
```
usage: predict.py [-h] [--model FILE] [--input INPUT] [--in_folder INPUT]
                  [--output INPUT] [--out_folder INPUT]

Predict parcellation map with 36 ROIs based on FreeSurfer protocol from input
surfaces

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: trained_models/left_hemi_40k_curv_sulc.pkl)
  --input INPUT, -i INPUT
                        filename of input surface (default: None)
  --in_folder INPUT, -in_folder INPUT
                        folder path for input files. Will parcelalte all the
                        files end in .vtk in this folder. Accept input or
                        in_folder. (default: surfaces/left_hemisphere)
  --output INPUT, -o INPUT
                        Filename of ouput surface. If not given, default is
                        [input].parc.vtk (default: None)
  --out_folder INPUT, -out_folder INPUT
                        folder path for ouput surface. If not given, default
                        is the same as input_folder. Accept output or
                        out_folder. (default: None)
```
Troubleshoot notes:
1. Remember to modify the model path [left_hemi_40k_curv_sulc.pkl](https://github.com/zhaofenqiang/Spherical_U-Net/blob/master/trained_models/left_hemi_40k_curv_sulc.pkl) and [right_hemi_40k_curv_sulc.pkl](https://github.com/zhaofenqiang/Spherical_U-Net/blob/master/trained_models/right_hemi_40k_curv_sulc.pkl) for left hemispheres and right hemispheres.
2. The code requires `input` or `in_folder` option, not both, for single surface’ parcellation or all surfaces in the folder. 
3. The input data should be end in .vtk.

### Examples
You can test the code using the example surfaces we provided in the `surfaces` folder. Simply run:
```
python predict.py -i surfaces/left_hemisphere/test1.lh.40k.vtk
```
You will get the corresponding output surface at the same folder with name `test1.lh.40k.parc.vtk`.
Or, run the command for all the 5 surface in the same folder:
```
python predict.py -in_folder surfaces/left_hemisphere
```
Note that we also provide the ground truth parcellation maps in `par_fs_vec` field in the vtk file. So you can compare and compute the parcellation accuracy and Dice.

### Visualization
You can use [Paraview](https://www.paraview.org/) software to visualize the parcellated surface in VTK format. An example of the input curvature map and output parcellation map are shown below.


## Cite
If you use this code for your research, please cite as:

Fenqiang Zhao, et.al. Spherical U-Net on Cortical Surfaces: Methods and Applications. Information Processing in Medical Imaging (IPMI), 2019.


