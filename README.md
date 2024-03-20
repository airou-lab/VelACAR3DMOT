# Extended_CRN: Camera Radar Net for Accurate, Robust, Efficient 3D Perception and tracking

## Abstract

(#TODO)


## Getting Started / Detection

### Git
```shell
# clone repo
git clone git@github.com:airou-lab/ExtendedCRN.git

```

### Docker
Creating Docker image and container for this project
```
# Pulling base nvidia image
sudo docker pull nvidia/cuda:11.1.1-devel-ubuntu20.04

# Getting to Docker folder
cd ~/Documents/ExtendedCRN/Docker

# Building CRN image
sudo docker build -t extcrn_image:v1 .

# Creating mounted gpu-enabled container
sudo docker run --name ExtCRN_V1 -v ~/Documents/ExtendedCRN:/home/ws --gpus all --shm-size 10G -it extcrn_image:v1
```
You should now be in the container shell.

Upon encountering any memory-related issue, be sure to check the shared memory of the container using ```df -h```. A simple fix can be to increase the -shm-size.

### Installation
```shell

cd Detection/mmdetection3d
pip install -v -e .
cd ..

python setup.py develop  # GPU required

```

### Data preparation
**Step 0.** Download [nuScenes dataset](https://www.nuscenes.org/nuscenes#download).

**Step 1.** Symlink the dataset folder to `./data/`.
```
ln -s ../../../data/nuScenes ./data/
```

**Step 2.** Create annotation file. 
This will generate `nuscenes_infos_{train,val}.pkl`.
```
python scripts/gen_info.py
```

**Step 3.** Generate ground truth depth.  
*Note: this process requires LiDAR keyframes.*
```
python scripts/gen_depth_gt.py
```

**Step 4.** Generate radar point cloud in perspective view. 
You can download pre-generated radar point cloud [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/youngseok_kim_kaist_ac_kr/EcEoswDVWu9GpGV5NSwGme4BvIjOm-sGusZdCQRyMdVUtw?e=OpZoQ4).  
*Note: this process requires radar blobs (in addition to keyframe) to utilize sweeps.*  
```
python scripts/gen_radar_bev.py  # accumulate sweeps and transform to LiDAR coords
python scripts/gen_radar_pv.py  # transform to camera coords
```

The folder structure will be as follows:
```
ExtendedCRN
├──Detection
|   ├── data
|   │   ├── nuScenes (link)
|   │   │   ├── nuscenes_infos_train.pkl
|   │   │   ├── nuscenes_infos_val.pkl
|   │   │   ├── maps
|   │   │   ├── samples
|   │   │   ├── sweeps
|   |   |   ├── depth_gt
|   |   |   ├── radar_bev_filter  # temporary folder, safe to delete
|   |   |   ├── radar_pv_filter
|   |   |   ├── v1.0-trainval
├──data
|   ├──nuScenes

```

### Training and Evaluation
**Training**
```
python [EXP_PATH] --amp_backend native -b 4 --gpus 4
```

**Evaluation**  
*Note: use `-b 1 --gpus 1` to measure inference time.*
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 4 --gpus 4
```
*Example using R50* 
```
python exps/det/CRN_r50_256x704_128x128_4key.py --ckpt_path checkpoint/CRN_r50_256x704_128x128_4key.pth -e -b 4 --gpus 1
```

### Model Zoo
All models use 4 keyframes and are trained without CBGS.  
All latency numbers are measured with batch size 1, GPU warm-up, and FP16 precision.

|  Method  | Backbone | NDS  | mAP  | FPS  | Params | Config                                                  | Checkpoint                                                                                                  |
|:--------:|:--------:|:----:|:----:|:----:|:------:|:-------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------:|
| BEVDepth |   R50    | 47.1 | 36.7 | 29.7 | 77.6 M | [config](exps/det/BEVDepth_r50_256x704_128x128_4key.py) | [model](https://github.com/youngskkim/CRN/releases/download/v1.0/BEVDepth_r50_256x704_128x128_4key.pth) |
|   CRN    |   R18    | 54.2 | 44.9 | 29.4 | 37.2 M | [config](exps/det/CRN_r18_256x704_128x128_4key.py)      | [model](https://github.com/youngskkim/CRN/releases/download/v1.0/CRN_r18_256x704_128x128_4key.pth)      |
|   CRN    |   R50    | 56.2 | 47.3 | 22.7 | 61.4 M | [config](exps/det/CRN_r50_256x704_128x128_4key.py)      | [model](https://github.com/youngskkim/CRN/releases/download/v1.0/CRN_r50_256x704_128x128_4key.pth)      |


### Features
- [ ] BEV segmentation checkpoints 
- [ ] BEV segmentation code 
- [x] 3D detection checkpoints 
- [x] 3D detection code 
- [x] Code release 

### Acknowledgement
This project is based on excellent open source projects:
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)


### Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@inproceedings{kim2023crn,
    title={Crn: Camera radar net for accurate, robust, efficient 3d perception},
    author={Kim, Youngseok and Shin, Juyeb and Kim, Sanmin and Lee, In-Jae and Choi, Jun Won and Kum, Dongsuk},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={17615--17626},
    year={2023}
}
```

## Tracking

### Data links
Copy detection ouput to keep a backup and pretty print json file. 
```
cd /home/ws/Detection
mkdir detection_output
cp outputs/det/CRN_r50_256x704_128x128_4key/*.json detection_output

cd detection_output
jq . results_nusc.json > pretty_printed_results_nusc.txt
```
In the tracking folder, simlink the data and the detection output to the data folder.
```
cd /home/ws/Tracking
mkdir data
ln -s ../../data/nuScenes ./data/
ln -s ../../Detection/detection_output ./data/
```
Your data folder in the Tracking folder should look like this : 
```
ExtendedCRN
├──Tracking
|   ├── data
|   │   ├── nuScenes (link)
|   │   ├── detection-output (link)
│   |   │   ├── metrics_details.json
│   |   │   ├── metrics_summary.json
│   |   │   ├── pretty_printed_results_nusc.txt
│   |   │   ├── results_nusc.json
├──data
|   ├──nuScenes
├──Detection
|   ├──detection_output
│   |   │   ├── metrics_details.json
│   |   │   ├── metrics_summary.json
│   |   │   ├── pretty_printed_results_nusc.txt
│   |   │   ├── results_nusc.json

```
### Data formatting

(#TODO)






