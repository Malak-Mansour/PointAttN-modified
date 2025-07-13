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
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Mar_21_19:15:46_PDT_2021
Cuda compilation tools, release 11.3, V11.3.58
Build cuda_11.3.r11.3/compiler.29745058_0

PyTorch Version: 1.12.1
CUDA Version: 11.3

3.10.16



### Compile Pytorch 3rd-party modules

please compile Pytorch 3rd-party modules [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch) and [mm3d_pn2](https://github.com/Colin97/MSN-Point-Cloud-Completion). A simple way is using the following command:

```
cd $PointAttN_Home/utils/ChamferDistancePytorch/chamfer3D
python setup.py install

cd $PointAttN_Home/utils/mm3d_pn2
python setup.py build_ext --inplace
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

