#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

import os 
import json 
import numpy as np
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
from Tracking.libs.utils import load_nusc, box_name2color, random_colors, render_box
Box.render_box = render_box




def visualization_by_frame_and_cat(args,box_list,nusc,token,sample_data_token):

    # sample = nusc.get('sample', token)
    sample_data = nusc.get('sample_data', sample_data_token)   # data for sample 0

    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    imsize = (sample_data['width'], sample_data['height'])

    image_path = os.path.join(args.data_root,sample_data['filename'])    
    # img = cv2.imread(image_path)  

    if args.color_method == 'random':
        max_color = 30
        colors = random_colors(max_color)       # Generate random colors
        ID = 0

    img = cv2.imread(image_path) 
    print(image_path)

    skip_flag = True

    for box in box_list:
        if box.name==args.cat:
        
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
                skip_flag = False
    
                if args.disp_custom : 
                    box.render_box(im=img,text=str(box.score)[0:3],view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
                else :
                    box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
        

    if args.add_gt:
        _, nusc_box_list,_ = nusc.get_sample_data(sample_data_token = sample_data_token)
        c = (0,0,0)
        
        for gt_box in nusc_box_list: 
            if (gt_box.name).split('.')[1] == args.cat:
                skip_flag = False
                gt_box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)



    if skip_flag and args.skip:
        return 0

    cv2.imshow('image', img) 

    while (True):
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

def visualization_by_frame(args,box_list,nusc,token,sample_data_token):

    # sample = nusc.get('sample', token)
    sample_data = nusc.get('sample_data', sample_data_token)   # data for sample 0

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

    for box in box_list:
        
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
            if args.disp_custom : 
                box.render_box(im=img,text=str(box.score)[0:3],view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
            else :
                box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)

    if args.add_gt:
        _, nusc_box_list,_ = nusc.get_sample_data(sample_data_token = sample_data_token)
        c = (0,0,0)
        for gt_box in nusc_box_list: 
            gt_box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)

    cv2.imshow('image', img) 

    while (True):
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

def visualization(args):

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
            sample_data_token = sample_data['token']

            while(True):

                sample_token = sample_data['sample_token']
                # sample_data_token = sample_data['token']
                # print(sample_data)

                if sample_token in det_data['results']:

                    if (args.keyframes_only == True and sample_data['is_key_frame'] == True) or args.keyframes_only == False:

                        box_list = []
                        
                        for det_sample in det_data['results'][sample_token]:
                            if det_sample['detection_score'] >= get_score_thresh(args,det_sample['detection_name']):   # Discard low confidence score detection 

                                box = Box(center = det_sample['translation'],
                                            size = det_sample['size'],
                                            orientation = Quaternion(det_sample['rotation']),
                                            score = float(det_sample['detection_score']),
                                            velocity = [det_sample['velocity'][0],det_sample['velocity'][1],0],
                                            name = det_sample['detection_name'],
                                            token = det_sample['sample_token']
                                            )

                                if args.detection_method == 'CRN':
                                    #translate the box upward by h/2, fixing CRN bbox shift (see CRN git issues)
                                    box.center[2]+=box.wlh[2]/2

                                # print(box)
                                box_list.append(box)
            
                                if args.verbose>=2:
                                    print(box)
                                    print(det_sample)

                        if args.viz_by_cat:
                            key = visualization_by_frame_and_cat(args,box_list,nusc,sample_token,sample_data_token)
                        else:
                            key = visualization_by_frame(args,box_list,nusc,sample_token,sample_data_token)
                        
                if sample_data['next'] == "":
                    #GOTO next scene
                    break
                else:
                    if key == 0:
                        #GOTO next sample
                        sample_data_token = sample_data['next']
                        sample_data = nusc.get('sample_data', sample_data_token)

                    elif key == 1 :
                        if sample_data['prev'] != "":
                            #GOTO prev sample
                            sample_data_token = sample_data['prev']

                        elif sample_data['prev'] == "":
                            #GOTO same sample
                            sample_data_token = sample_data_token
                        
                        sample_data = nusc.get('sample_data', sample_data_token)

def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root','--nusc_data_root', type=str, default='./data_mini/nuScenes', help='nuScenes data folder path')
    parser.add_argument('--split', type=str, default='val', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='Camera views (see sensor_list)')

    parser.add_argument('--detection_method', type=str, default='CRN', help='CRN/Radiant')
    parser.add_argument('--color_method',type=str, default='class', help='class/random')
    
    parser.add_argument('--viz_by_cat','-vbc',action='store_true', default=False, help='visualize each frame category by category')
    parser.add_argument('--cat', type=str, default=None, help='specify the desired category to visualize')

    parser.add_argument('--verbose','-v' ,action='count',default=0,help='verbosity level')
    parser.add_argument('--disp_custom', action='store_true', default=False, help='Add a custom display to the boxes')
    parser.add_argument('--keyframes_only', '-kf' , action='store_true', default=False, help='Only use keyframes')
    parser.add_argument('--skip' , action='store_true', default=False, help='skip when no detection')

    parser.add_argument('--add_gt','-gt', action='store_true', default=False, help='also display ground truth')


    parser.add_argument('--det_data_dir', type=str, default='./Detection/detection_output_mini', help='Detection data folder path')

    return parser

def check_args(args):
    sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                    'LIDAR_TOP',
                    'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']  

    if 'mini' in args.split:
        args.split = (args.split).split('mini_')[1]
        
        if 'mini' not in args.data_root :
            args.data_root= (args.data_root).split('data')[0]+'data_mini'+(args.data_root).split('data')[1]

        if 'mini' not in args.det_data_dir :
            args.det_data_dir= (args.det_data_dir).split('detection_output')[0]+'detection_output_mini'+(args.det_data_dir).split('detection_output')[1]

    assert args.split in ['train','val','test'], 'Wrong split type'
    assert args.sensor in sensor_list, 'Unknown sensor selected'        
    assert args.color_method in ['class','random'],'Unknown color_method selected' 

    assert os.path.exists(args.data_root), 'Data folder at %s not found'%(args.data_root)
    assert os.path.exists(args.det_data_dir), 'Detection folder at %s not found'%(args.det_data_dir)
    assert os.path.exists(args.det_data_dir+'/results_nusc.json'), 'Missing json detection file at %s'%(args.det_data_dir)

    print(args)


if __name__ == '__main__':

    sensor_list =['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                    'LIDAR_TOP',
                    'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT'] 

    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'construction_vehicle', 'motorcycle', 'trailer']

    parser = create_parser()
    args = parser.parse_args() 
    check_args(args)

    visualization(args)

'''
launch with :

mini:
python det_viz.py --detection_method CRN --color_method class --disp_custom -vvv

val:
python det_viz.py --data_root ./data/nuScenes --split val --det_data_dir ./Detection/detection_output \
                    --detection_method CRN --color_method class --disp_custom -vvv

test:
python det_viz.py --data_root ./data/nuScenes --split test --det_data_dir ./Detection/detection_output \
                    --detection_method CRN --color_method class --disp_custom -vvv                

'''