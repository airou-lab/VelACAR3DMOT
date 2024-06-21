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
xhost local:root	# Input this for every new shell
sudo docker run --name ExtCRN_V1 -v ~/Documents/ExtendedCRN:/home/ws --gpus all --shm-size 10G -it extcrn_image:v1

# To have a GUI-enabled container :
sudo docker run --name ExtCRN_V1 -v ~/Documents/ExtendedCRN:/home/ws --gpus all --shm-size 10G -it \
		--env="DISPLAY" \
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		extcrn_image:v1
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
Make sure your nuscenes devkit is updated to the latest version :
```shell
pip install nuscenes-devkit -U
```


### Data preparation
**Step 0.** Download [nuScenes dataset](https://www.nuscenes.org/nuscenes#download).

**Step 1.** Symlink the dataset folder to `./data/`.
```shell
ln -s ../../../data/nuScenes ./data/
```

**Step 2.** Create annotation file. 
This will generate `nuscenes_infos_{train,val}.pkl`.
```shell
python scripts/gen_info.py
```

**Step 3.** Generate ground truth depth.  
*Note: this process requires LiDAR keyframes.*
```shell
python scripts/gen_depth_gt.py
```

**Step 4.** Generate radar point cloud in perspective view. 
You can download pre-generated radar point cloud [here](https://kaistackr-my.sharepoint.com/:u:/g/personal/youngseok_kim_kaist_ac_kr/EcEoswDVWu9GpGV5NSwGme4BvIjOm-sGusZdCQRyMdVUtw?e=OpZoQ4).  
*Note: this process requires radar blobs (in addition to keyframe) to utilize sweeps.*  
```shell
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
```shell
python [EXP_PATH] --amp_backend native -b 4 --gpus 4
```

**Evaluation**  
*Note: use `-b 1 --gpus 1` to measure inference time.*
```shell
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 4 --gpus 4
```
*Example using R50* 
```shell
python exps/det/CRN_r50_256x704_128x128_4key.py --ckpt_path checkpoint/CRN_r50_256x704_128x128_4key.pth -e -b 1 --gpus 1
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
```shell
cd /home/ws/Detection
mkdir detection_output
cp outputs/det/CRN_r50_256x704_128x128_4key/*.json detection_output

cd detection_output
jq . results_nusc.json > pretty_printed_results_nusc.txt
```
In the tracking folder, simlink the data and the detection output to the data folder.
```shell
cd /home/ws/Tracking
mkdir data
ln -s ../../data/nuScenes ./data/
ln -s ../../Detection/detection_output ./data/
```
Your data folder in the Tracking folder should look like this within the project structure: 
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

#### (This is a provisional description that will be changed as the project evolves and gets cleaner once everything works)

*N.B: As of now, the ground truth tracking still needs the detections folders to work. Because of this, you'll find the detection folder for the cars and pedestrian in the git project in case you do not want to run CRN. This should allow it to work, but you should change the cat_list list to ```['car', 'pedestrian']```.*

___

AB3DMOT requires the detection data to be separated in different files by object class.<br>
To do this, run:
```shell
python workfile.py --go_sep
```
This should generate a folder named 'cat_detection'. Inside this folder should be one folder per category, inside which you should find one .txt file per scene.<br>
Those text files contain the detections for this scenes+category, with a integer indice indicating the frame from the scene.<br>
Your Tracking data folder should now look like this :
```
ExtendedCRN
├──Tracking
|   ├── data
|   │   ├── nuScenes (link)
|   │   ├── detection-output (link)
|   │   ├── cat_detection
│   |   │   ├── CRN_bicycle
│   |   │   ├── CRN_bus
│   |   │   ├── CRN_car
|   │   |   │   ├── scene-0103.txt
|   │   |   │   ├── scene-0553.txt
|   │   |   │   ├── scene-0796.txt
|   │   |   │   ├── scene-0916.txt
│   |   │   ├── ...
│   |   │   ├── CRN_truck
```
You can now run the Tracking in different configurations:<br>
- ```python  workfile.py --viz``` will display each frame and add the tracked object bounding boxes. The tracking is done category by category.<br>
- ```python  workfile.py --log_viz``` will save each aforementionned frame in .png files in a folder for each category, all under a 'results' folder.<br>
- ```python  workfile.py --gt_track``` will display each frame and add the tracking bounding box, using the ground truth as detection backbone. This is used for debugging purposes.<br>

To generate the nuScenes-formatted results, run:
```shell
python workfile.py
```
Finally, to evaluate the output using nuScenes official evauation:
```shell
# Create a json output file nammed : 'track_results_nusc.json' in the Tracking/output/track_output_CRN/ directory, containing all the detections and formatted for evaluation.
python workfile.py --concat	

# Display metrics and log them inside 'output/track_output_CRN/'
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ --eval_set val --dataroot ./data/nuScenes 
```

Alternatively, you can use the shell script launcher to run the whole tracking pipeline, including the separation, concatenation, and evaluation:
```shell
bash launch.sh
```

The final output will be found in ./Tracking/output/track_output_CRN/


