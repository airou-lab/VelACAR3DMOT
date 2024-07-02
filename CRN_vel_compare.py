#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import json 
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any, Tuple
from pyquaternion import Quaternion
import argparse
import random

import colorsys
import cv2

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from Tracking.libs.config import *
from Tracking.libs.utils import load_nusc, box_name2color, render_box
Box.render_box = render_box

def get_gt_boxes(nusc,sample_data):

    sensor_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    GT_box_list=[]
    GT_token_mem=[]


    sample_token = sample_data['sample_token']
    sample = nusc.get('sample',sample_token)

    GT_ID = 0

    for sensor in sensor_list:
        sample_data = nusc.get('sample_data', sample['data'][sensor])   

        sample_data_token = sample_data['token']
        _, nusc_box_list,_ = nusc.get_sample_data(sample_data_token = sample_data_token)


        sd_record = sample_data
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        ego_pose = nusc.get('ego_pose', sd_record['ego_pose_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])

        for box in nusc_box_list: 
            if box.token not in GT_token_mem:  
                GT_ID+=1

                #  Move box from sensor coord to ego.
                box.rotate(Quaternion(cs_record['rotation']))
                box.translate(np.array(cs_record['translation']))

                # Move box from ego vehicle to global.
                box.rotate(Quaternion(ego_pose['rotation']))
                box.translate(np.array(ego_pose['translation']))
                
                gt_vel = tuple(nusc.box_velocity(box.token))
                if np.isnan(gt_vel).any():              # In case box_velocity cannot calculate the speed, handle the nan output case
                    gt_vel = tuple([0,0,0])

                box.velocity = gt_vel
                box.token = 'GT_'+str(GT_ID)
                
                GT_token_mem.append(box.token)
                GT_box_list.append(box)

    return(GT_box_list)

def visualization_by_frame(args,det_box_list,gt_box_list,nusc,token):

    sample = nusc.get('sample', token)
    sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    imsize = (sample_data['width'], sample_data['height'])

    image_path = os.path.join(args.data_root,sample_data['filename'])    
    img = cv2.imread(image_path) 

    if args.color_method == 'random':
        max_color = 30
        colors = random_colors(max_color)       # Generate random colors
        ID = 0

    for box in det_box_list:

        # Move box to ego vehicle coord system.
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        if args.color_method == 'class':
            c = box_name2color(box.name)

        elif args.color_method =='random':
            ID+=1
            color_float = colors[(int(ID)-1) % max_color]           # loops back to first color if more than max_color
            color_int = tuple([int(tmp * 255) for tmp in color_float])
            c = color_int

        if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
            box.render_box(im=img,text=box.token,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)

    for box in gt_box_list:

        # Move box to ego vehicle coord system.
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        c = (0,0,0)

        if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
            box.render_box(im=img,text=box.token,vshift=10,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
    

    cv2.imshow('image', img) 

    key = cv2.waitKeyEx(0)

    if key == 65363 or key == 13:   # right arrow or enter
        if args.verbose>=1 : print("next frame:")
        cv2.destroyAllWindows()
        return 0

    elif key == 65361:              # left arrow
        if args.verbose>=1 : print("previous frame:")
        cv2.destroyAllWindows()
        return 1

    elif key == 113:              # q key
        cv2.destroyAllWindows()
        exit()

def load_det(args):

    nusc = load_nusc(args.split,args.data_root)
    # cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']

    det_file = args.det_data_dir+'/results_nusc.json'

    print('Loading data from:',det_file)

    with open(det_file) as json_file:

        det_data = json.load(json_file)

        # scenes_list = /os.listdir(args.data_dir)
        split_scenes = create_splits_scenes()
        scenes_list = split_scenes[args.split]
        if args.verbose>=2:
            print('List of scenes :')
            print(scenes_list)

        for scene in nusc.scene:
            scene_name = scene['name']
            if scene_name not in scenes_list:   #only parsing scenes we used (i.e the corrrect dataset split). Done this way in case CRN uses a != split than official one
                continue

            print ('Processing scene', scene_name)
            
            # Parsing scene token. Using nuscenes loop to go through all the tokens even if no detections
            first_token = scene['first_sample_token']
            sample_token = first_token
            sample = nusc.get('sample', first_token) # sample 0
            sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0
            
            while(True):

                sample_token = sample_data['sample_token']

                if sample_data['is_key_frame'] == True and sample_token in det_data['results']:
                # if sample_token in det_data['results']:
                    
                    print(150*'_')
                    print('sample:')
                    print(sample)
                    print()

                    print('sample_data:')
                    print(sample_data)
                    print()

                    print('sample_token:')
                    print(sample_token)
                    print()
                    print(150*'_')

                    det_box_list = []
                    boxID=0
                    
                    for det_sample in det_data['results'][sample_token]:

                        if det_sample['detection_score'] >= get_score_thresh(det_sample['detection_name']):   # Discard low confidence score detection 

                            
                            boxID+=1

                            box = Box(center = det_sample['translation'],
                                        size = det_sample['size'],
                                        orientation = Quaternion(det_sample['rotation']),
                                        score = float(det_sample['detection_score']),
                                        velocity = [det_sample['velocity'][0],det_sample['velocity'][1],0],
                                        name = det_sample['detection_name'],
                                        token = 'CRN_'+str(boxID)
                                        )

                            box.translate(np.array([0.0,0.0,box.wlh[2]/2]))

                            # print(box)
                            det_box_list.append(box)

                            if args.verbose>=1:
                                print(box)
                                print(det_sample)

                    gt_box_list=get_gt_boxes(nusc,sample_data)

                    print('\n\n')
                    print(150*'-')
                    print(*det_box_list,sep = "\n")
                    print(150*'-')
                    print(*gt_box_list,sep = "\n")
                    print(150*'-')
                    print('\n\n')
                    # input()

                    key = visualization_by_frame(args,det_box_list,gt_box_list,nusc,sample_token)
                    
                if sample_data['next'] == "":
                    #GOTO next scene
                    break
                else:
                    if key == 0:
                        #GOTO next sample
                        sample_token = sample_data['next']
                        sample_data = nusc.get('sample_data', sample_token)

                    elif key == 1 and sample_data['prev'] != "":
                        #GOTO prev sample
                        sample_token = sample_data['prev']
                        sample_data = nusc.get('sample_data', sample_token)

                    elif key == 1 and sample_data['prev'] == "":
                        #GOTO same sample
                        sample_token = sample_token
                        sample_data = nusc.get('sample_data', sample_token)

def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data_mini/nuScenes', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='val', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='train_val/test')

    parser.add_argument('--color_method',type=str, default='class', help='class/random')
    
    parser.add_argument('--verbose','-v' ,action='count',default=0,help='verbosity level')

    parser.add_argument('--det_data_dir', type=str, default='./Detection/detection_output_mini', help='detection data folder')

    return parser



if __name__ == '__main__':

    sensor_list =['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                    'LIDAR_TOP',
                    'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']   
    
    parser = create_parser()
    args = parser.parse_args()

    assert args.split in ['train','val','test'], 'Unknown split type'
    assert args.sensor in sensor_list,'Unknown sensor selected' 
    assert args.color_method in ['class','random'],'Unknown color_method selected' 
    assert os.path.exists(args.data_root), 'data root at %s not found'%(args.data_root)
    assert os.path.exists(args.det_data_dir), 'detection data at %s not found'%(args.det_data_dir)

    load_det(args)