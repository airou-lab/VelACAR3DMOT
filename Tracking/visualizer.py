import os 
import json 
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any
from pyquaternion import Quaternion
import argparse

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

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

def fixed_colors():
    f = open('color_scheme.txt')     # rewritting file
    text = f.readlines()
    color_name = []
    color_val_txt = []
    color_val = []
    for item in text:
        color_name.append(item.split(',')[0])
        color_val_txt.append(item.split('(')[1].split(')')[0])
    for color in color_val_txt:
        color_tmp = (float(color.split(',')[0])/255,float(color.split(',')[1])/255,float(color.split(',')[2])/255,)
        color_val.append(color_tmp)

    f.close()
    return color_val

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

    return c

def visualization_by_frame(args,box_list,nusc,token):

    sample = nusc.get('sample', token)
    sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

    # print(sample_data)
    # print()

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
        
        # print(box)

        # Move box to ego vehicle coord system.
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        
        if args.color_method == 'class':
            c = box_name2color(box.name)

        elif args.color_method =='id':
            ID = box.label
            color_float = colors[(int(ID)-1) % max_color]           # loops back to first color if more than max_color
            color_int = tuple([int(tmp * 255) for tmp in color_float])
            c = color_int

        
        if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
            box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
    

    cv2.imshow('image', img) 

    key = cv2.waitKeyEx(0)

    if key == 65363 or key == 13:   # right arrow or enter
        print("next frame:")
        cv2.destroyAllWindows()
        return 0

    elif key == 65361:              # left arrow
        print("previous frame:")
        cv2.destroyAllWindows()
        return 1


def visualization_concatenated(args):

    nusc = load_nusc(args.split,args.data_root)
    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']

    tracking_file = args.data_dir+'/track_results_nusc.json'

    print('Loading data from:',tracking_file)

    with open(tracking_file) as json_file:

        track_data = json.load(json_file)

        # scenes_list = /os.listdir(args.data_dir)
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
            sample_token = first_token
            sample = nusc.get('sample', first_token) # sample 0
            sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

            while(True):

                token = sample_data['sample_token']

                if sample_data['is_key_frame'] == True and token in track_data['results']:

                    box_list = []
                    
                    for track_sample in track_data['results'][token]:

                        # print(track_sample)
                        # input()

                        q = Quaternion(track_sample['rotation'])

                        box = Box(center = track_sample['translation'],
                                    size = track_sample['size'],
                                    orientation = q,
                                    label = int(track_sample['tracking_id'].split('_')[0]),
                                    score = float(track_sample['tracking_score']),
                                    velocity = [track_sample['velocity'][0],track_sample['velocity'][1],0],
                                    name = track_sample['tracking_name'],
                                    token = token
                                    )
                        # print(box)
                        box_list.append(box)
    
                        if args.verbose>=1:
                            print(box)
                            print(track_sample)

                    key = visualization_by_frame(args,box_list,nusc,token)
                    
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

def visualization_logs(args):

    nusc = load_nusc(args.split,args.data_root)
    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']


    scenes_list = os.listdir(args.data_dir)

    print('Loading data from dataframes stored in:',args.data_dir)

    for scene in nusc.scene:
        scene_name = scene['name']
        if scene_name not in scenes_list:   #only parsing scenes we used (i.e the corrrect dataset split)
            continue

        print ('Processing scene', scene_name)
        scene_df = pd.DataFrame(columns =['w','l','h','x','y','z','vx','vy','ID','r1','r2','r3','r4','score','token','object'])
        token_mem=[]

        for cat in cat_list:
            df_tmp = pd.read_pickle(os.path.join(args.data_dir,scene_name,cat+'.pkl'))
            df_tmp = df_tmp.drop(['theta','t'],axis=1)

            if df_tmp.empty:
                # print('no detection for %s in %s'%(cat,scene_name)) 
                continue

            df_tmp.loc[:,'object']=cat


            scene_df = pd.concat([scene_df,df_tmp])

        scene_df = scene_df.reset_index(drop=True)


        # Parsing scene token. Using nuscenes loop to go through all the tokens even if no detections
        first_token = scene['first_sample_token']
        sample_token = first_token
        sample = nusc.get('sample', first_token) # sample 0
        sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

        while(True):

            token = sample_data['sample_token']

            if sample_data['is_key_frame'] == True:
        
                sample_result_list = []
                
                df_by_token = scene_df.loc[scene_df['token']==token]

                # print(df_by_token)
                # input()

                box_list = []

                for i in range(len(df_by_token)):
                    sample_df = df_by_token.iloc[i]

                    q = Quaternion([sample_df['r1'],sample_df['r2'],sample_df['r3'],sample_df['r4']])

                    box = Box(center = [sample_df['x'],sample_df['y'],sample_df['z']],
                                size = [sample_df['w'],sample_df['l'],sample_df['h']],
                                orientation = q,
                                label = int(sample_df['ID']),
                                score = float(sample_df['score']),
                                velocity = [sample_df['vx'],sample_df['vy'],0],
                                name = sample_df['object'],
                                token = token
                                )
                    box_list.append(box)

                    if args.verbose>=1:
                        print(box)


                visualization_by_frame(args,box_list,nusc,token)
                
            if sample_data['next'] == "":
                #GOTO next scene
                break
            else:
                #GOTO next sample
                sample_token = sample_data['next']
                sample_data = nusc.get('sample_data', sample_token)


def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/nuScenes', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='val', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='train_val/test')
    parser.add_argument('--detection_method','--det', type=str, default='CRN', help='detection method')
    parser.add_argument('--viz_concat','-c', action='store_true', default=False, help='visualize concatenated results')
    parser.add_argument('--color_method',type=str, default='class', help='class/id')
    
    parser.add_argument('--verbose','-v' ,action='count',default=0,help='verbosity level')

    parser.add_argument('--data_dir', type=str, default='./output/CRN_hyper_exp/metrics/iou_2d', help='tracking data folder')

    return parser



if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    if args.viz_concat:        
        visualization_concatenated(args)
    else:
        visualization_logs(args)


# launch with :
# python visualizer.py --sensor CAM_FRONT --color_method class --data_dir output/CRN_hyper_exp/metrics/iou_2d -vvv