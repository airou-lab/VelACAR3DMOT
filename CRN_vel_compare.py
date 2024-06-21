import os 
import json 
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any
from pyquaternion import Quaternion
import argparse
import random

import colorsys
import cv2

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes



def load_nusc(split,data_root):
    assert split in ['train','val','test'], "Bad nuScenes version"

    if split in ['train','val']:
        nusc_version = 'v1.0-trainval'
    elif split =='test':
        nusc_version = 'v1.0-test'
    
    nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=True)

    return nusc

def get_gt_at_t(nusc,t,sample,next_sample_tokens_dict):
    '''
    To get every detections at one frame we need the tokens from all the cameras. 
    To know these while not changing the pipeline too much from the regular tracking, this function returns a 
    dict containing those values for the next frame.
    The same dictionnary is taken as input at the next frame to know the sensor tokens at that frame.
    when the dictionnary is empty (i.e for the first sample of the scene) wee get the tokens from the sample argument
    '''

    sensor_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    GT_box_list=[]

    for sensor in sensor_list:

        if sensor not in list(next_sample_tokens_dict.keys()):
            sample_data = nusc.get('sample_data', sample['data'][sensor])   # data for sample 0
            next_sample_tokens_dict[sensor]=sample_data['next'] #update sensor
        else :
            token = next_sample_tokens_dict[sensor]
            if next_sample_tokens_dict[sensor] == '':   #Security if one sensor datastream is smaller
                continue
            sample_data=nusc.get('sample_data', token)
            next_sample_tokens_dict[sensor]=sample_data['next']

        metadata_token = sample_data['token']
        _, nusc_box_list,_ = nusc.get_sample_data(sample_data_token = metadata_token)

        cs_record, ego_pose, _ = get_sample_metadata(nusc,sensor,metadata_token,verbose=False)


    for box in nusc_box_list:                
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

            print(box)
            exit()
            
            GT_box_list.append(box)

    return(GT_box_list,next_sample_tokens_dict)

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

            print(sample)
            print(sample_data)
            input()
            
            while(True):

                token = sample_data['sample_token']

                if sample_data['is_key_frame'] == True and token in det_data['results']:

                    det_box_list = []
                    
                    for det_sample in det_data['results'][token]:

                        if det_sample['detection_score'] >= args.score_thresh:   # Discard low confidence score detection 

                            box = Box(center = det_sample['translation'],
                                        size = det_sample['size'],
                                        orientation = Quaternion(det_sample['rotation']),
                                        score = float(det_sample['detection_score']),
                                        velocity = [det_sample['velocity'][0],det_sample['velocity'][1],0],
                                        name = det_sample['detection_name'],
                                        token = det_sample['sample_token']
                                        )

                            det_box_list.append(box)

                            if args.verbose>=1:
                                print(box)
                                print(det_sample)

                    # gt_box_list=get_gt_at_t(nusc,t,sample,next_sample_tokens_dict)

                    
                if sample_data['next'] == "":
                    #GOTO next scene
                    break
                else:
                    #GOTO next sample
                    sample_token = sample_data['next']
                    sample_data = nusc.get('sample_data', sample_token)

def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data_mini/nuScenes', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='val', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='train_val/test')

    parser.add_argument('--score_thresh', type=float, default=0.4, help='minimum score threshold')    
    parser.add_argument('--color_method',type=str, default='class', help='class/random')
    
    parser.add_argument('--verbose','-v' ,action='count',default=0,help='verbosity level')

    parser.add_argument('--det_data_dir', type=str, default='./Detection/detection_output_bckp_mini', help='detection data folder')

    return parser



if __name__ == '__main__':

    sensor_list =['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                    'LIDAR_TOP',
                    'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']   
    
    parser = create_parser()
    args = parser.parse_args()

    assert args.split in ['train','val','test'], 'Unknown split type'
    assert args.score_thresh >= 0,'Score threshold needs to be a positive number' 
    assert args.sensor in sensor_list,'Unknown sensor selected' 
    assert args.color_method in ['class','random'],'Unknown color_method selected' 

    load_det(args)


# launch with :
# python visualizer.py -vvv