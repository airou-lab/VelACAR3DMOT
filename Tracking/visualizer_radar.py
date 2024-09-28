#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

# Visualize radar point cloud along with corresponding image


import os 
import json 
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any, Tuple
from pyquaternion import Quaternion
import argparse
import copy

import colorsys
import cv2

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, RadarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from libs.box import Box3D
from libs.utils import load_nusc, random_colors, fixed_colors, box_name2color, render_box, mkdir_if_missing
Box.render_box = render_box


def visualizer(args):

    nusc = load_nusc(args.split,args.data_root)
    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']
    sensor_list = ['RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']

    split_scenes = create_splits_scenes()
    scenes_list = split_scenes[args.split]

    if args.verbose>=2:
        print('List of scenes :')
        print(scenes_list)

    for scene in nusc.scene:
        scene_name = scene['name']
        if scene_name not in scenes_list:   #only parsing scenes we used (i.e the corrrect dataset split)
            continue

        print ('Processing scene', scene_name)

        # Parsing scene token. Using nuscenes loop to go through all the tokens even if no detections
        first_token = scene['first_sample_token']
        sample_data_token = first_token
        sample = nusc.get('sample', first_token) # sample 0

        while(True):

            sample_token = sample['token']

            sample_data_radar = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
            sample_data_cam = nusc.get('sample_data', sample['data']['CAM_FRONT'])

            if sample_data_cam['is_key_frame'] == True:

                # extracting metadata for camera
                cs_record = nusc.get('calibrated_sensor', sample_data_cam['calibrated_sensor_token'])
                ego_pose = nusc.get('ego_pose', sample_data_cam['ego_pose_token'])
                cam_intrinsic = np.array(cs_record['camera_intrinsic'])

                image_path = os.path.join(args.data_root,sample_data_cam['filename'])    
                print(image_path)

                img = cv2.imread(image_path)

                # cv2.imshow('image', img)
                # cv2.waitKeyEx(0)
                # cv2.destroyAllWindows()

                if image_path == './data_mini/nuScenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151605512404.jpg':
                    nusc.render_sample_data(sample['data']['RADAR_FRONT'], nsweeps=5, underlay_map=True, with_anns = False) 
                    nusc.render_sample_data(sample['data']['CAM_FRONT'], with_anns = False) 
                    exit()



            if sample['next'] == "":
                #GOTO next scene
                break
            else:
                #GOTO next sample
                sample_token = sample['next']
                sample = nusc.get('sample', sample_token)


                    # # extracting metadata
                    # cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                    # ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
                    # cam_intrinsic = np.array(cs_record['camera_intrinsic'])

                    # image_path = os.path.join(args.data_root,sample_data['filename'])    
                    # img = cv2.imread(image_path) 

    #                 # max_color = 30
    #                 # colors = random_colors(max_color)       # Generate random colors
    #                 colors = fixed_colors()                   # Using pre-made color list to identify ID
    #                 max_color = len(colors)

    #                 for track_sample in track_data['results'][sample_token]:

    #                     q = Quaternion(track_sample['rotation'])

    #                     box = Box(center = track_sample['translation'],
    #                                 size = track_sample['size'],
    #                                 orientation = q,
    #                                 label = int(track_sample['tracking_id'].split('_')[0]),
    #                                 score = float(track_sample['tracking_score']),
    #                                 velocity = [track_sample['velocity'][0],track_sample['velocity'][1],0],
    #                                 name = track_sample['tracking_name'],
    #                                 token = sample_token
    #                                 )                    
                            
    #                     if args.add_bottom_box:
    #                         bot_box = get_bot_box(args,box,ego_pose,cs_record)

    #                     # Move box to ego vehicle coord system.
    #                     box.translate(-np.array(ego_pose['translation']))
    #                     box.rotate(Quaternion(ego_pose['rotation']).inverse)

    #                     #  Move box to sensor coord system.
    #                     box.translate(-np.array(cs_record['translation']))
    #                     box.rotate(Quaternion(cs_record['rotation']).inverse)
                        
    #                     if args.color_method == 'class':
    #                         c = box_name2color(box.name)

    #                     elif args.color_method =='id':
    #                         ID = box.label
    #                         color_float = colors[(int(ID)-1) % max_color]           # loops back to first color if more than max_color
    #                         color_int = tuple([int(tmp * 255) for tmp in color_float])
    #                         c = color_int

                        
    #                     if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
    #                         box.render_box(im=img,text=box.name+'_'+str(ID),view=cam_intrinsic,normalize=True,bottom_disp=True,colors=(c, c, c),linewidth=2, text_scale=1.0)
                            
    #                         if args.add_bottom_box:
    #                             c=(255,255,255)
    #                             bot_box.render_cv2(im=img,view=cam_intrinsic,normalize=True,colors=(c, c, c),linewidth=1)                    

    #                 filename = os.path.join(output_folder,sample_data['filename'].split('/')[1],sample_data['filename'].split('/')[2])
    #                 cv2.imwrite(filename,img)
    #                 print("Image saved in: %s"%(filename))


    #             if sample_data['next'] == "":
    #                 #GOTO next scene
    #                 break
    #             else:
    #                 #GOTO next sample
    #                 sample_data_token = sample_data['next']
    #                 sample_data = nusc.get('sample_data', sample_data_token)


def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data_mini/nuScenes', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='val', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='Camera views')

    parser.add_argument('--verbose','-v' ,action='count',default=0,help='verbosity level')

    return parser



if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    assert args.split in ['train','val','test'], "wrong split type"

    visualizer(args)
    exit()