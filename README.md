<!-- Malak: start -->

<!-- see where ur logged in -->
squeue --me 
<!-- allocate to a specific device -->
salloc -w ws-l5-004 
<!-- log out of some device -->
squeue kill --me


<!-- open a persistent terminal -->
tmux 
<!-- see what terminal was tmux -->
tmux ls 
<!-- open the previous tmux -->
tmux a


<!-- Installation Instructions: -->
salloc -n12 -N1 --mem=24G


conda install -n base conda-libmamba-solver
conda config --set solver libmamba


conda create --name pointAttn python=3.10 pip
conda activate pointAttn

conda create --name pointAttn2 python=3.10 pip
conda activate pointAttn2


<!-- conda install nvidia/label/cuda-12.4.0::cuda-nvcc 
conda install cuda-toolkit=12.4.0 -c nvidia -->

conda install nvidia/label/cuda-12.4.0::cuda-toolkit

<!-- conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit nvcc  # More complete CUDA toolkit -->
<!-- conda install -c conda-forge cuda-toolkit=12.4.0 cuda-nvcc -->


<!-- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -->
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia




export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd PointAttN-Modified-main
pip install -r requirements.txt
 

<!-- conda install -c conda-forge cudatoolkit-dev  # For CUDA headers
conda install -c conda-forge ninja  # Often required for CUDA builds -->
<!-- rm -rf build dist *.egg-info -->


cd utils/ChamferDistancePytorch/chamfer3D
python setup.py install
 

cd utils/pointnet2_ops_lib
python setup.py install


cd PointAttN-Modified/PointAttN-Modified-main
python train.py -c PointAttN.yaml



# Running Openpoints inside PointAttn
inside PointAttn-Modified/models:
   git clone openpoints


paste the following imports in models/PointAttN.py:
   from .openpoints.models.backbone.pointnext import PointNextEncoder 
   from .openpoints.models.backbone.dgcnn import DGCNN

prepend models. to the openpoints. paths in all the files inside openpoints folder. Example:
   PointAttN-Modified-main/models/openpoints/models/build.py

   PointAttN-Modified-main/models/openpoints/models/layers/subsample.py
                                                           group.py
                                                           upsampling.py
                                                           graph_conv.py

   PointAttN-Modified-main/models/openpoints/models/backbone/dgcnn.py
                                                            deepgcn.py
                                                            pct.py
                                                            simpleview.py

   PointAttN-Modified-main/models/openpoints/models/reconstruction/maskedpointvit.py

   PointAttN-Modified-main/models/openpoints/loss/build.py


pip install easydict

cd models/openpoints/cpp/pointnet2_batch/
python setup.py install

cd models/openpoints/cpp/chamfer_dist/
python setup.py install


cd ../../../..


In PointAttN-Modified-main/models/openpoints/models/backbone/pointnext.py, change 
   line 128 add:
      conv_args = {}
      conv_args["order"] = "conv-norm-act"

   lines 387-388 change group_args.radius and .nsample to:
      group_args["radius"] = self.radii[i]
      group_args["nsample"] = self.nsample[i]
      
   lines 421-424 change group_args.radius and .nsample to:
      radii =  group_args["radius"] 
      nsample =  group_args["nsample"] 
      group_args["radius"] = radii[0]
      group_args["nsample"] = nsample[0]


- python train.py -c PointAttN.yaml

- add log file path to cfgs/PointAttN.yaml of the cd loss best reported at the latest epoch
- python test_pcn.py -c PointAttN.yaml

- visualize the sim run using visualize_predictions_malak.py, then delete the all folder after ur done

other runs to do: 
- use data_mix/data and data_mix/data_test instead of data and data_test in train.py and test_pcn.py
- take the loss function from train 1 and PointAttn 1.py
- repeat all with dgcnn encoder class



<!-- Malak: end -->


# Point Cloud Completion Based Robotic Fruit Harvesting Pipeline



## 1. Environment setup

### Install related libraries

```
pip install -r requirements.txt
```

```
sudo apt update
sudo apt install g++-9

CC=gcc-9 CXX=g++-9 python setup.py install # might not be needed
sudo apt install ros-humble-octomap-ros
sudo apt install ros-humble-moveit-ros-perception

 ros2 service call /process_pointcloud std_srvs/Trigger "{}"
 ros2 topic pub /reset std_msgs/Empty "{}" --once


```

conda install cuda-toolkit=12.4.0 -c nvidia --force-reinstall
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
# Set environment variables
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

PyTorch Version: 2.5.1
CUDA Version: 12.4
numpy 1.24.2
3.10.16



### Compile Pytorch 3rd-party modules

please compile Pytorch 3rd-party modules [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) and [mm3d_pn2](https://github.com/Colin97/MSN-Point-Cloud-Completion). A simple way is using the following command:

```
cd $PointAttN_Home/utils/ChamferDistancePytorch/chamfer3D
python setup.py install

cd $PointAttN_Home/utils/pointnet2_ops_lib
python setup.py install
```

## 2. Train

### Prepare training datasets

Download the datasets:

+ [PCN(raw data)](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz)
+ [PCN(processed data)](https://gateway.infinitescript.com/?fileName=ShapeNetCompletion)
+ [Completion3D](https://completion3d.stanford.edu/)

### Train a model

To train the PointAttN model, modify the dataset path in `cfgs/PointAttN.yaml `, run:

```
python train.py -c PointAttN.yaml
```

## 3. Test

### Pretrained models

The pretrained models on Completion3D and PCN benchmark are available as follows:

|   dataset    | performance |                          model link                          |
| :----------: | :---------: | :----------------------------------------------------------: |
| Completion3D |  CD = 6.63  | [[BaiDuYun](https://pan.baidu.com/s/17-BZr3QvHYjEVMjPuXHXTg)] (code：nf0m)[[GoogleDrive](https://drive.google.com/drive/folders/1uw0oJ731uLjDpZ82Gp7ILisjeOrNdiHK?usp=sharing)] |
|     PCN      |  CD = 6.86  | [[BaiDuYun](https://pan.baidu.com/s/187GjKO2qEQFWlroG1Mma2g)] (code：kmju)[[GoogleDrive](https://drive.google.com/drive/folders/1uw0oJ731uLjDpZ82Gp7ILisjeOrNdiHK?usp=sharing)] |

### Test for paper result

To test PointAttN on PCN benchmark, download  the pretrained model and put it into `PointAttN_cd_debug_pcn `directory, run:

```
python test_pcn.py -c PointAttN.yaml
```

To test PointAttN on Completion3D benchmark, download  the pretrained model and put it into `PointAttN_cd_debug_c3d `directory, run:

```
python test_c3d.py -c PointAttN.yaml
```

## 4. Acknowledgement

1. We include the following PyTorch 3rd-party libraries:  
   [1] [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch)  
   [2] [mm3d_pn2](https://github.com/Colin97/MSN-Point-Cloud-Completion)

2. Some of the code of this project is borrowed from [VRC-Net](https://github.com/paul007pl/MVP_Benchmark)  

## 5. Cite this work

If you use PointAttN in your work, please cite our paper:

```
@article{Wang_Cui_Guo_Li_Liu_Shen_2024,
   title={PointAttN: You Only Need Attention for Point Cloud Completion},
   volume={38}, 
   url={https://ojs.aaai.org/index.php/AAAI/article/view/28356}, DOI={10.1609/aaai.v38i6.28356}, 
   number={6}, 
   journal={Proceedings of the AAAI Conference on Artificial Intelligence},
   author={Wang, Jun and Cui, Ying and Guo, Dongyan and Li, Junxia and Liu, Qingshan and Shen, Chunhua},
   year={2024},
   month={Mar.},
   pages={5472-5480}
}
```

