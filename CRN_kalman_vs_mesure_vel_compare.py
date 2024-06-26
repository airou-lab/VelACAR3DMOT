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

def load_nusc(split,data_root):
    assert split in ['train','val','test'], "Bad nuScenes version"

    if split in ['train','val']:
        nusc_version = 'v1.0-trainval'
    elif split =='test':
        nusc_version = 'v1.0-test'
    
    nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=True)

    return nusc

def box_name2color(name):
    if name == 'car':
        c = (255,0,0)       # red
    
    elif name == 'pedestrian':
        c = (0,0,255)       # blue
    
    elif name == 'truck':
        c = (255,255,0)     # yellow
    
    elif name == 'bus':
        c = (255,0,255)     # magenta
    
    elif name == 'bicycle':
        c = (0,255,0)       # green
    
    elif name == 'motorcycle':
        c = (192,192,192)   # silver
    
    elif name == 'trailer':
        c = (165,42,42)     # brown

    else :
        c = (255,255,255)     # brown

    return c

def render_box(self, im: np.ndarray, vshift:int = 0, hshift:int = 0, view: np.ndarray = np.eye(3), normalize: bool = False, colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)), linewidth: int = 2) -> None:
    """
    Renders box using OpenCV2.
    :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
    :param vshift/hshift : int. Add a vertical/horizontal shift to the bbox label
    :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
    :param normalize: Whether to normalize the remaining coordinate.
    :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
    :param linewidth: Linewidth for plot.
    """
    corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(im,
                     (int(prev[0]), int(prev[1])),
                     (int(corner[0]), int(corner[1])),
                     color, linewidth)
            prev = corner

    # Draw the sides
    for i in range(4):
        cv2.line(im,
                 (int(corners.T[i][0]), int(corners.T[i][1])),
                 (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                 colors[2][::-1], linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners.T[:4], colors[0][::-1])
    draw_rect(corners.T[4:], colors[1][::-1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    cv2.line(im,
             (int(center_bottom[0]), int(center_bottom[1])),
             (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
             colors[0][::-1], linewidth)

    h = corners.T[3][1] - corners.T[0][1]
    # l = corners.T[0][0] - corners.T[1][0]

    center = [center_bottom[0],center_bottom[1]-h/2]

    cv2.putText(im,
                self.token,
                org=(int(center[0])+hshift, int(center[1])+vshift),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 0),thickness=1,lineType=cv2.LINE_AA
                )

render_box
Box.render_box = render_box

def get_det_data(args,det_data,sample_token,label):

    det_box_list = []
    boxID=0

    for det_sample in det_data['results'][sample_token]:

        if det_sample['detection_score'] >= args.score_thresh:   # Discard low confidence score detection 
            
            boxID+=1

            box = Box(center = det_sample['translation'],
                        size = det_sample['size'],
                        orientation = Quaternion(det_sample['rotation']),
                        score = float(det_sample['detection_score']),
                        velocity = [det_sample['velocity'][0],det_sample['velocity'][1],0],
                        name = det_sample['detection_name'],
                        token = label+'_'+str(boxID)
                        )

            det_box_list.append(box)
    return det_box_list

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

def visualization_by_frame(args,det_box_list1,det_box_list2,nusc,token):

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

    for box in det_box_list1:

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
            box.render_box(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)

    for box in det_box_list2:

        # Move box to ego vehicle coord system.
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        c = (0,0,0)

        if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
            box.render_box(im=img,vshift=10,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
    
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



def nusc_loop(args):

    nusc = load_nusc(args.split,args.data_root)
    # cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']

    vel_file = 'output/CRN_vel_exp/with_vel/results_nusc.json'
    no_vel_file = 'output/CRN_vel_exp/without_vel/results_nusc.json'

    print('Loading vel data from:',vel_file)
    print('Loading no_vel data from:',no_vel_file)

    vel_json_file = open(vel_file,'r')
    no_vel_json_file = open(vel_file,'r')

    vel_det_data = json.load(vel_json_file)
    no_vel_det_data = json.load(no_vel_json_file)

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

            if sample_data['is_key_frame'] == True and sample_token in vel_det_data['results'] and sample_token in no_vel_det_data['results']:
                
                if args.verbose >=1:
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

                vel_det_box_list = get_det_data(args,vel_det_data,sample_token,'vel')
                no_vel_det_box_list = get_det_data(args,no_vel_det_data,sample_token,'kalman')

                print('\n\n')
                print(150*'-')
                print('vel_det_box_list:')
                print(*vel_det_box_list,sep = "\n")
                print(150*'-')
                print()
                print('no_vel_det_box_list:')
                print(*no_vel_det_box_list,sep = "\n")
                print(150*'-')
                print('\n\n')
                # input()

                key = visualization_by_frame(args,vel_det_box_list,no_vel_det_box_list,nusc,sample_token)
                
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
    




    vel_json_file.close()
    no_vel_json_file.close()

def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/nuScenes', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='val', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='train_val/test')

    parser.add_argument('--score_thresh', type=float, default=0.4, help='minimum score threshold')    
    parser.add_argument('--color_method',type=str, default='class', help='class/random')
    
    parser.add_argument('--verbose','-v' ,action='count',default=0,help='verbosity level')

    parser.add_argument('--det_data_dir', type=str, default='./Detection/detection_output', help='detection data folder')

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

    nusc_loop(args)