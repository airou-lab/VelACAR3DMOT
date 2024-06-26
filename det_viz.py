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
    random.shuffle(colors)
    return colors

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

def get_score_thresh_by_cat(cat):
    if cat == 'car':
        return 0.4
    elif cat =='pedestrian':
        return 0.6
    elif cat =='truck':
        return 0.4
    elif cat =='bus':
        return 0.2
    elif cat =='bicycle':
        return 0.3
    elif cat =='motorcycle':
        return 0.3
    elif cat =='trailer':
        return 0.1
    else :
        return 0.8

def render_box(self, im: np.ndarray, text: str, vshift:int = 0, hshift:int = 0, view: np.ndarray = np.eye(3), normalize: bool = False, colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)), linewidth: int = 2) -> None:
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
                text,
                org=(int(center[0])+hshift, int(center[1])+vshift),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0, 0, 0),thickness=1,lineType=cv2.LINE_AA
                )

render_box
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
                            if det_sample['detection_score'] >= get_score_thresh_by_cat(det_sample['detection_name']):   # Discard low confidence score detection 

                                box = Box(center = det_sample['translation'],
                                            size = det_sample['size'],
                                            orientation = Quaternion(det_sample['rotation']),
                                            score = float(det_sample['detection_score']),
                                            velocity = [det_sample['velocity'][0],det_sample['velocity'][1],0],
                                            name = det_sample['detection_name'],
                                            token = det_sample['sample_token']
                                            )

                                if args.det_method == 'CRN':
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
                        
                        # key will be defined as the first data is a sample

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
    parser.add_argument('--data_root','--nusc_data_root', type=str, default='./data/nuScenes', help='nuScenes data folder path')
    parser.add_argument('--split', type=str, default='val', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='see sensor_list')

    parser.add_argument('--score_thresh', type=float, default=0.4, help='Minimum detection score threshold')    
    parser.add_argument('--color_method',type=str, default='class', help='class/random')
    parser.add_argument('--det_method',type=str, default='CRN', help='CRN/Radiant')

    parser.add_argument('--viz_by_cat','-vbc',action='store_true', default=False, help='visualize each frame category by category')
    parser.add_argument('--cat', type=str, default=None, help='specify the desired category to visualize')

    parser.add_argument('--verbose','-v' ,action='count',default=0,help='verbosity level')
    parser.add_argument('--disp_custom', action='store_true', default=False, help='Add a custom display to the boxes')
    parser.add_argument('--keyframes_only', '-kf' , action='store_true', default=False, help='Only use keyframes')
    parser.add_argument('--skip' , action='store_true', default=False, help='skip when no detection')

    parser.add_argument('--add_gt','-gt', action='store_true', default=False, help='also display ground truth')


    parser.add_argument('--det_data_dir', type=str, default='./Detection/detection_output', help='Detection data folder path')

    return parser



if __name__ == '__main__':

    sensor_list =['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                    'LIDAR_TOP',
                    'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT'] 

    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'construction_vehicle', 'motorcycle', 'trailer']

    parser = create_parser()
    args = parser.parse_args()

    assert args.split in ['train','val','test'], 'Unknown split type'
    assert args.score_thresh >= 0,'Score threshold needs to be a positive number' 
    assert args.sensor in sensor_list,'Unknown sensor selected' 
    assert args.color_method in ['class','random'],'Unknown color_method selected' 

    if args.cat!=None or args.viz_by_cat==True: 
        assert args.viz_by_cat==True,'Argument -vbc must be set to true to use argument cat'
        assert args.cat in cat_list, 'Unregognized or missing object category'

    visualization(args)

'''
launch with :

python det_viz.py --det_method CRN --color_method class --disp_custom -vvv

'''