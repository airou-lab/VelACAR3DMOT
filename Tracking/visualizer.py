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
import copy

import colorsys
import cv2

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from libs.box import Box3D
from libs.utils import load_nusc, random_colors, fixed_colors, box_name2color, render_box


def get_bot_box(args,box,ego_pose,cs_record):

    if args.verbose >=3:
        # -----------------------------Creating a box object with AB3DMOT functions------------------------------
        box_3d = Box3D(x=box.center[0],
                        y=box.center[1],
                        z=box.center[2],
                        w=box.wlh[0],
                        h=box.wlh[2],
                        l=box.wlh[1],
                        ry=box.orientation.radians
                        )
        print()
        print('nusc box:')
        print(box)
        print()
        print('AB3DMOT box_3d:')
        print(box_3d)
        print(200*'_')

        # Proving corners with nusc built-in and AB3DMOT function are equivalent:
        box_3d_corners = Box3D.box2corners3d_camcoord(box_3d)
        print('AB3DMOT box_corners:')
        print(box_3d_corners)   # slight difference due to rotation matrices
        print()
        print('nusc box.corners():')
        print(box.corners().T)
        print(200*'.')

        # Proving bottom corners with nusc built-in and AB3DMOT function are equivalent:
        box_3d_bot = box_3d_corners[[2,3,7,6], :3] # taken from dist_metrics.py
        print('AB3DMOT bottom box corners:')
        print(box_3d_bot)
        print()
        print('nusc bottom box corners:')
        print(box.bottom_corners().T)
        print(200*'=')

    # --------------------------Creating bottom_box by projecting box on x,y plane---------------------------
    bot_box = copy.deepcopy(box)
    bot_box.center[2]=bot_box.center[2]-bot_box.wlh[2]/2
    bot_box.wlh[2]=0

    # making sure the boxes are equivalent
    print('nusc box:')
    print(box)
    print()
    print('projected box on xy plane:')
    print(bot_box)
    print(200*'-')

    print('projected box corners')
    print(bot_box.corners().T)  # showing [0,1,4,5] = [3,2,7,6] (resp) => rectangular box wwith no height
    print()                 
    print('projected box bottom corners')
    print(bot_box.bottom_corners().T) # showing bottom box is equivalent to previous ones

    if args.verbose>=4:
        input()


    # --------------------------------Transforming bottom box to camera coord-------------------------------- 

    # Move box to ego vehicle coord system.
    bot_box.translate(-np.array(ego_pose['translation']))
    bot_box.rotate(Quaternion(ego_pose['rotation']).inverse)

    #  Move box to sensor coord system.
    bot_box.translate(-np.array(cs_record['translation']))
    bot_box.rotate(Quaternion(cs_record['rotation']).inverse)

    return bot_box

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
        
        if args.add_bottom_box:
            bot_box = get_bot_box(args,box,ego_pose,cs_record)

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
            box.render_box(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
            
            if args.add_bottom_box: 
                c=(255,255,255)
                bot_box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)



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

    elif key == 113:              # q key
        cv2.destroyAllWindows()
        exit()

def render_box(self,im: np.ndarray,view: np.ndarray = np.eye(3),normalize: bool = False,colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),linewidth: int = 2) -> None:
    """
    Renders box using OpenCV2.
    :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
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

    print(self.corners().T)
    print(corners.T)
    # exit()

    # Draw line indicating the front
    center_bottom_forward = np.mean(corners.T[2:4], axis=0)
    center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
    cv2.line(im,
             (int(center_bottom[0]), int(center_bottom[1])),
             (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
             colors[0][::-1], linewidth)

    h = corners.T[3][1] - corners.T[0][1]
    l = corners.T[0][0] - corners.T[1][0]

    center = [center_bottom[0]-l/2,center_bottom[1]-h/2]

    cv2.putText(im,
                str(self.label),
                org=(int(center[0]), int(center[1])),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 0),thickness=1,lineType=cv2.LINE_AA
                )

render_box
Box.render_box = render_box



def visualization_json(args):

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

                    elif key == 1 :
                        if sample_data['prev'] != "":
                            #GOTO prev sample
                            sample_data_token = sample_data['prev']

                        elif sample_data['prev'] == "":
                            #GOTO same sample
                            sample_data_token = sample_data_token
                        
                        sample_data = nusc.get('sample_data', sample_data_token)

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
    parser.add_argument('--data_root', type=str, default='./data_mini/nuScenes', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='val', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='train_val/test')
    parser.add_argument('--detection_method','--det', type=str, default='CRN', help='detection method')
    parser.add_argument('--viz_method', type=str, default='json', help='visualize json_file or logs')
    parser.add_argument('--color_method',type=str, default='class', help='class/id')
    
    parser.add_argument('--verbose','-v' ,action='count',default=0,help='verbosity level')

    parser.add_argument('--add_bottom_box', '-b', action='store_true', default=False, help='add bottom bounding box to plot (in white)')

    parser.add_argument('--data_dir', type=str, default='./output/track_output_CRN', help='tracking data folder')

    return parser



if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    assert args.viz_method in ['json','logs'], "unknown visualization method"
    assert args.color_method in ['class','id'], "unknown color method"
    assert args.split in ['train','val','test'], "wrong split type"

    if args.viz_method == 'json':        
        visualization_json(args)
    elif args.viz_method == 'logs':
        visualization_logs(args)


'''
launch with :
python visualizer.py --sensor CAM_FRONT --color_method class --data_dir output/CRN_hyper_exp/metrics/iou_2d -vvv
python visualizer.py --color_method id --viz_method logs --data_dir results/logs/CRN -b -vvv

mini :
python visualizer.py --data_root ./data_mini/nuScenes --data_dir ./output/track_output_CRN_mini \
                    --color_method id --viz_method json -v
'''