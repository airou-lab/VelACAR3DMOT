#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------
# Extracts and save camera image + radar point cloud + detection data  
# at t, t+1 and t+2
# using a specific sample token

import os 
import json 
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any, Tuple
from pyquaternion import Quaternion
import argparse
import copy
import shutil

import colorsys
import cv2

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, RadarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from libs.config import get_score_thresh
from libs.box import Box3D
from libs.utils import load_nusc, random_colors, fixed_colors, box_name2color, render_box, mkdir_if_missing
Box.render_box = render_box

def add_det_to_img(args,nusc,sample_token,box_list):

    sample = nusc.get('sample', sample_token)
    sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    imsize = (sample_data['width'], sample_data['height'])

    image_path = os.path.join(args.data_root,sample_data['filename'])    
    img = cv2.imread(image_path) 

    for box in box_list:
        
        # Move box to ego vehicle coord system.
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        c = box_name2color(box.name)

        if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
            box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=3)

    
    img_token = (image_path.split('/')[-1]).split('.jpg')[0]
    det_output_path = os.path.join(args.output_dir,'DET',img_token+'.jpg')

    cv2.imwrite(det_output_path, img)

    print('Det data copied to:', det_output_path)


def add_track_to_img(args,nusc,sample_token,box_list):
    
    sample = nusc.get('sample', sample_token)
    sample_data = nusc.get('sample_data', sample['data'][args.sensor])

    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    imsize = (sample_data['width'], sample_data['height'])

    image_path = os.path.join(args.data_root,sample_data['filename'])    
    img = cv2.imread(image_path) 

    # max_color = 30
    # colors = random_colors(max_color)       # Generate random colors
    colors = fixed_colors()                   # Using pre-made color list to identify ID
    max_color = len(colors)

    for box in box_list:

        # Move box to ego vehicle coord system.
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        ID = box.label
        color_float = colors[(int(ID)-1) % max_color]           # loops back to first color if more than max_color
        color_int = tuple([int(tmp * 255) for tmp in color_float])
        c = color_int

        if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
            box.render_box(im=img,text=box.name+'_'+str(ID),view=cam_intrinsic,normalize=True,bottom_disp=True,colors=(c, c, c),linewidth=3,text_scale = 1.5)
            # box.render_box(im=img,text='',view=cam_intrinsic,normalize=True,bottom_disp=True,colors=(c, c, c),linewidth=3,text_scale = 1.5)
    
    img_token = (image_path.split('/')[-1]).split('.jpg')[0]
    track_output_path = os.path.join(args.output_dir,'TRACK',img_token+'.jpg')

    cv2.imwrite(track_output_path, img)
    print('Track data copied to:', track_output_path)


def add_track_to_img_without_txt(args,nusc,sample_token,box_list):
    # Add track data without any txt 
    sample = nusc.get('sample', sample_token)
    sample_data = nusc.get('sample_data', sample['data'][args.sensor])

    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    imsize = (sample_data['width'], sample_data['height'])

    image_path = os.path.join(args.data_root,sample_data['filename'])    
    img = cv2.imread(image_path) 

    # max_color = 30
    # colors = random_colors(max_color)       # Generate random colors
    colors = fixed_colors()                   # Using pre-made color list to identify ID
    max_color = len(colors)

    for box in box_list:

        # Move box to ego vehicle coord system.
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        ID = box.label
        color_float = colors[(int(ID)-1) % max_color]           # loops back to first color if more than max_color
        color_int = tuple([int(tmp * 255) for tmp in color_float])
        c = color_int

        if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
            box.render_box(im=img,text='',view=cam_intrinsic,normalize=True,bottom_disp=True,colors=(c, c, c),linewidth=3,text_scale = 1.5)
    
    img_token = (image_path.split('/')[-1]).split('.jpg')[0]
    track_output_path = os.path.join(args.output_dir,'TRACK_no_txt',img_token+'.jpg')

    cv2.imwrite(track_output_path, img)
    print('Track data copied to:', track_output_path)


def data_extract(args,nusc):
    # Extracting image and radar point cloud at a specific token from t to t+2

    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']
    sensor_list = ['RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']

    mkdir_if_missing(args.output_dir)
    mkdir_if_missing(args.output_dir+'/CAM_FRONT')
    mkdir_if_missing(args.output_dir+'/RADAR_FRONT')
    print(100*'=')


    sample_token = args.token

    sample = nusc.get('sample', sample_token) # sample 0


    for i in range(args.period):
        if i==0:
            print('t=t')
        else:
            print('t=t+%i'%(i))
        print()


        print('token:')
        print(sample_token)
        
        print('sample:')
        print(sample)
        print(100*'-')

        # Extracting camera image
        # ------------------------------------------------------------------------
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

        image_path = os.path.join(args.data_root,cam_data['filename'])    
        print('Image found at:', image_path)

        img_token = (image_path.split('/')[-1]).split('.jpg')[0]

        cam_output_path = os.path.join(args.output_dir,'CAM_FRONT',img_token+'.jpg')


        shutil.copyfile(image_path,cam_output_path)
        print('Image copied to:', cam_output_path)

        # img = cv2.imread(image_path)
        # cv2.imsave('image', img)
        # cv2.waitKeyEx(0)
        # cv2.destroyAllWindows()
        # nusc.render_sample_data(sample['data']['CAM_FRONT'], with_anns = False) 

        # Extracting Radar point_cloud
        # ------------------------------------------------------------------------
        cam_output_path = os.path.join(args.output_dir,'RADAR_FRONT',img_token)

        nusc.render_sample_data(sample['data']['RADAR_FRONT'], nsweeps=5, with_anns = False, verbose=False, out_path=cam_output_path) 
        print('Radar data copied to:', cam_output_path+'.png')

        # next sample:
        sample_token = sample['next']
        sample = nusc.get('sample', sample_token)
 
def detection_file_extract(args,nusc):
    # Extracting detection output at specific token from t to t+2

    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']
    sensor_list = ['RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']

    sample_token = args.token
    sample = nusc.get('sample', sample_token) # sample 0

    mkdir_if_missing(args.output_dir+'/DET')
    print(100*'=')

    if args.mini:
        det_file_path = '../Detection/detection_output_mini/results_nusc.json'
    else:
        det_file_path = '../Detection/detection_output/results_nusc.json'   # val
    print('Loading data from:',det_file_path)
    
    #open detection file
    with open(det_file_path) as json_file:
        det_data = json.load(json_file)


        # looping from to to t+2
        for i in range(args.period):
            if i==0:
                print('t=t')
            else:
                print('t=t+%i'%(i))
                print()


            print('token:')
            print(sample_token)
            
            print('sample:')
            print(sample)
            print(100*'-')

            #initialize bbox list
            box_list = []


            # extracting bboxes from det_file
            for det_sample in det_data['results'][sample_token]:

                # Discard low confidence score detection 
                if det_sample['detection_score'] >= get_score_thresh(args,det_sample['detection_name']):   

                    box = Box(center = det_sample['translation'],
                                size = det_sample['size'],
                                orientation = Quaternion(det_sample['rotation']),
                                score = float(det_sample['detection_score']),
                                velocity = [det_sample['velocity'][0],det_sample['velocity'][1],0],
                                name = det_sample['detection_name'],
                                token = det_sample['sample_token']
                                )

                    #translate the box upward by h/2, fixing CRN bbox shift (see CRN git issues)
                    box.center[2]+=box.wlh[2]/2

                    # adding box to list
                    box_list.append(box)

            # disp bbox to image and save to output folder
            add_det_to_img(args,nusc,sample_token,box_list)

            # next sample:
            sample_token = sample['next']
            sample = nusc.get('sample', sample_token)

def tracking_file_extract(args,nusc):
    # Extracting tracking output at specific token from t to t+2

    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']
    sensor_list = ['RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']

    sample_token = args.token
    sample = nusc.get('sample', sample_token) # sample 0

    mkdir_if_missing(args.output_dir+'/TRACK')
    mkdir_if_missing(args.output_dir+'/TRACK_no_txt')
    print(100*'=')

    if args.mini:
        track_file_path = './output/best_res_mini/global_test_R_kf_thresh_vel/track_results_nusc.json'
    else:
        track_file_path = './output/best_res_val/kf_R_with_vel/track_results_nusc.json'
    print('Loading data from:',track_file_path)


    #open tracking file
    with open(track_file_path) as json_file:
        track_data = json.load(json_file)

        # looping from to to t+2
        for i in range(args.period):
            if i==0:
                print('t=t')
            else:
                print('t=t+%i'%(i))
                print()


            print('token:')
            print(sample_token)
            
            print('sample:')
            print(sample)
            print(100*'-')



            #initialize bbox list
            box_list = []

            # extracting bboxes from det_file
            for track_sample in track_data['results'][sample_token]:

                box = Box(center = track_sample['translation'],
                            size = track_sample['size'],
                            orientation = Quaternion(track_sample['rotation']),
                            label = int(track_sample['tracking_id'].split('_')[0]),
                            score = float(track_sample['tracking_score']),
                            velocity = [track_sample['velocity'][0],track_sample['velocity'][1],0],
                            name = track_sample['tracking_name'],
                            token = sample_token
                            )

                # adding box to list
                box_list.append(box)

            # disp bbox to image and save to output folder
            add_track_to_img(args,nusc,sample_token,copy.deepcopy(box_list))
            add_track_to_img_without_txt(args,nusc,sample_token,copy.deepcopy(box_list))

            # next sample:
            sample_token = sample['next']
            sample = nusc.get('sample', sample_token)


def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/nuScenes', help='nuScenes data folder')
    parser.add_argument('--output_dir', type=str, default='./saved_img/paper_img', help='Output folder')
    parser.add_argument('--split', type=str, default='val', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='Camera sensor selection')
    parser.add_argument('--detection_method', type=str, default='CRN', help='Detection method')

    parser.add_argument('--period','-T', type=int, default=3,help='Extracting images from t to t+T')

    parser.add_argument('--mini' ,action='store_true',default=False,help='Use mini_dataset(debug)')

    parser.add_argument('--verbose','-v' ,action='count',default=0,help='verbosity level')

    parser.add_argument('--token', type=str, default=None, help='t=t token')

    return parser



if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    assert args.split in ['train','val','test'], "wrong split type"
    assert args.token != None
    assert os.path.exists(args.data_root), 'Data folder at %s not found'%(args.data_root)
    assert os.path.exists('./saved_img'), './saved_img not found'

    if args.mini:
        args.data_root = './data_mini/nuScenes'

    nusc = load_nusc(args.split,args.data_root)

    data_extract(args,nusc)
    detection_file_extract(args,nusc)
    tracking_file_extract(args,nusc)

    exit()