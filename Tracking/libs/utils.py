#-----------------------------------------------
# Modified by : Mathis Morales                       
# Email       : mathis-morales@outlook.fr             
# git         : https://github.com/MathisMM            
#-----------------------------------------------

import os 
# import sys 
import json 
import numpy as np
import pandas as pd
import pickle
import random
from typing import List, Dict, Any, Tuple
from pyquaternion import Quaternion

import colorsys
import cv2

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix

from .config import get_score_thresh


global sampling_freq
sampling_freq = 12 # default value of sampling frequency with sweeps


# Misc
def mkdir_if_missing(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print("created directory at:",path)

# nuScenes functions utils
def load_nusc(split,data_root):
    assert split in ['train','val','test'], "Bad nuScenes version"

    if split in ['train','val']:
        nusc_version = 'v1.0-trainval'
    elif split =='test':
        nusc_version = 'v1.0-test'
    
    nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=True)

    return nusc

def get_sensor_param(nusc, sample_token, cam_name='CAM_FRONT'):	# Unused function


    sample = nusc.get('sample', sample_token)

    # get camera sensor
    cam_token = sample['data'][cam_name]
    sd_record_cam = nusc.get('sample_data', cam_token)
    cs_record_cam = nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record_cam['ego_pose_token'])

    return pose_record, cs_record_cam

def get_sample_info(nusc,sensor,token,verbose=False):	# Unused function

    scenes = nusc.scene
    # print(scenes)
    # input()
    for scene in scenes:

        first_sample = nusc.get('sample', scene['first_sample_token']) # sample 0
        sample_data = nusc.get('sample_data', first_sample['data'][sensor])   # data for sample 0

        while True:
            if sample_data['sample_token']==token:
                if verbose :
                    print('\nscene: ',scene)
                    print('\nsample: ',first_sample)
                    print ('\nsample_data: ',sample_data)
                return scene['name'], sample_data['filename']

            if sample_data['next'] == "":
                #GOTO next scene
                # print("no next data")
                if verbose:
                    print ('token NOT in:',scene['name'])
                break
            else:
                #GOTO next sample
                next_token = sample_data['next']
                sample_data = nusc.get('sample_data', next_token)

        # #Looping scene samples
        # while(sample_data['next'] != ""):       
        #     # if sample_token corresponds to token
        #     if sample_data['sample_token']==token:

        #         if verbose :
        #             print('\nscene: ',scene)
        #             print('\nsample: ',first_sample)
        #             print ('\nsample_data: ',sample_data)
        #         return scene['name'], sample_data['filename']

        #     else:
        #         # going to next sample
        #         sample_data = nusc.get('sample_data', sample_data['next'])

    return 0

def get_total_scenes_list(nusc,sensor): # Unused function
    scenes = nusc.scene
    scenes_list=[]
    for scene in scenes :
        scenes_list.append(scene['name'])
        # print (scene)
        # first_sample = nusc.get('sample', scene['first_sample_token']) # sample 0
        # sample_data = nusc.get('sample_data', first_sample['data'][sensor])   # data for sample 0
        # print(sample_data)
        # input()

    return scenes_list

def get_scenes_list(path):
    scenes_list = []

    # listing scenes
    for scene in os.listdir(path):
        scenes_list.append(scene) if scene.split('.')[-1]=='txt' else ''
    
    return scenes_list

def get_sample_metadata (nusc,sensor,token,verbose=False):
    scenes = nusc.scene
    
    if verbose:
        print('Looking for metadata for token: %s'%(token))

    for scene in scenes:

        first_sample = nusc.get('sample', scene['first_sample_token']) # sample 0
        sample_data = nusc.get('sample_data', first_sample['data'][sensor])   # data for sample 0
        
        #Looping scene samples
        while(True):
            # if sample_token corresponds to token
            if sample_data['token']==token:
                if verbose:
                    print('\nscene:',scene)
                    print('\nfirst sample:',first_sample)
                    print('\nsample_data:',sample_data)

                    print('\nego token:',sample_data['ego_pose_token'])
                    print('\nsensor token:',sample_data['calibrated_sensor_token'],'\n')

                sd_record = nusc.get('sample_data', sample_data['token'])
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                cam_intrinsic = np.array(cs_record['camera_intrinsic'])
                imsize = (sd_record['width'], sd_record['height'])

                if verbose:
                    print('-----------------------------------------------------')
                    print('sd_record: ',sd_record)
                    print('\n cs_record: ',cs_record)
                    print('\n sensor_record: ',sensor_record)
                    print('\n pose_record: ',pose_record)
                    print('\n cam_intrinsic: ',cam_intrinsic)
                    print('\n imsize: ',imsize)
                    print('-----------------------------------------------------')
                    print ('\n\n')
                    print ()

                return cs_record, pose_record, cam_intrinsic

            if sample_data['next']=="":
                # going to next scene
                break
            else:
                # going to next sample
                sample_data = nusc.get('sample_data', sample_data['next'])

        if verbose:
            print ('token NOT in:',scene['name'])
    return 0

def get_ego_pose(nusc,sensor,token,verbose=False):# Unused function
    _, pose_record, _ = get_sample_metadata(nusc,sensor,token)
    return pose_record

def get_sensor_data(nusc,sensor,token,verbose=False):# Unused function
    cs_record, _, cam_intrinsic = get_sample_metadata(nusc,sensor,token)
    return cs_record, cam_intrinsic

def render_box(self, im: np.ndarray, text: str, vshift:int = 0, hshift:int = 0,
				view: np.ndarray = np.eye(3), normalize: bool = False,
				colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)), linewidth: int = 2) -> None:
    """
    Renders box using OpenCV2.
    :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
    :param text : str. Add any text to the image, by default the first letter is centered on the bbox 3d center.
    :param vshift/hshift : int. Add a vertical/horizontal shift to the bbox text.
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


# dataframes extraction
def get_det_df(args,cat,det_file):

        f = open(os.path.join(args.cat_detection_root,args.detection_method+'_'+cat,det_file),'r')
        detection_list = []

        if args.score_thresh>0:
            score_thresh = args.score_thresh #Forced global score threshold
        else :
            score_thresh=get_score_thresh(args,cat)


        for det in f:
            # Extracting all values 
            t,x,y,z,w,l,h,r1,r2,r3,r4,vx,vy,score,token,_ = det.split(',')
            
            if float(score)<score_thresh:
                continue

            # Setting vertical velocity to 0
            vz = 0

            if args.detection_method == 'CRN':
                # Correcting box center from bottom center to 3D center
                z = float(z)+float(h)/2
            else :
                z = float(z)

            if args.keyframes_only == True:
                global sampling_freq
                sampling_freq = 2

            if args.verbose>=5 :
                print (det)
                print ('t = ',t)
                print ('x = ',x)
                print ('y = ',y)
                print ('z = ',z)
                print ('w = ',w)
                print ('l = ',l)
                print ('h = ',h)
                print ('r1 = ',r1)
                print ('r2 = ',r2)
                print ('r3 = ',r3)
                print ('r4 = ',r4)
                print ('vx = ',vx)
                print ('vy = ',vy) 
                print ('vz = ',vz)
                print ('score'),score
                print ('token = ',token)

            detection_list.append([int(t),
                                float(x),float(y),z,
                                float(w),float(l),float(h),
                                float(r1),float(r2),float(r3),float(r4),
                                float(vx)/sampling_freq,float(vy)/sampling_freq,vz, # compensating sampling frequency
                                float(score),
                                token])

        f.close()
        detection_df = pd.DataFrame(detection_list,columns =['t','x','y','z','w','l','h','r1','r2','r3','r4','vx','vy','vz','score','token'])
        return(detection_df)

def get_det_df_at_t(df,t):
    df_at_t = df.loc[df['t']==t]
    df_at_t.reset_index(drop=True,inplace=True)
    return df_at_t

def get_gt_at_t(args,nusc,cat,t,sample,next_sample_tokens_dict):
    '''
    To get every detections at one frame we need the tokens from all the cameras. 
    To know these while not changing the pipeline too much from the regular tracking, this function returns a 
    dict containing those values for the next frame.
    The same dictionnary is taken as input at the next frame to know the sensor tokens at that frame.
    when the dictionnary is empty (i.e for the first sample of the scene) wee get the tokens from the sample argument
    '''

    sensor_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    global sampling_freq

    GT_list=[]

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

        if args.keyframes_only:
            sampling_freq=2

        for box in nusc_box_list:
            if (box.name).split('.')[1]==cat:
                
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

                # Extracting all values 
                t=t
                x,y,z = box.center
                w,l,h = box.wlh
                r1,r2,r3,r4 = box.orientation
                score = 0.99                #random.random()             # random score provides better eval results
                token = box.token
                vx,vy,vz = box.velocity
                vz=0
                
                GT_list.append([int(t),
                                float(x),float(y),float(z),
                                float(w),float(l),float(h),
                                float(r1),float(r2),float(r3),float(r4),
                                float(vx)/sampling_freq,float(vy)/sampling_freq,vz,
                                float(score),
                                token])

    GT_df = pd.DataFrame(GT_list,columns =['t','x','y','z','w','l','h','r1','r2','r3','r4','vx','vy','vz','score','token'])
    GT_df = GT_df.drop_duplicates(subset=['token']) # removing duplicate detection (happens when 2 sensors see the same object)
    GT_df = GT_df.reset_index(drop=True)

    GT_df.loc[:,'token'] = sample_data['sample_token']  # token is sample_token in the rest of the pipeline

    return(GT_df,next_sample_tokens_dict)


# colors and boxes
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

    else :
        c = (255,255,255)     # brown

    return c


# Results handling
def log_track_results_at_t(args,results_df,cat,scene_name,detection_method):
    '''
    results logged at results/logs/<detection_method>/<scene>/<cat>.pkl
    '''

    mkdir_if_missing('results')
    mkdir_if_missing(os.path.join('results','logs'))

    if 'mini' in args.data_root:    # mini dataset
        mkdir_if_missing(os.path.join('results','logs',detection_method+'_mini'))
        output_path = os.path.join('results','logs',detection_method+'_mini',scene_name)
    else:       # Complete dataset
        mkdir_if_missing(os.path.join('results','logs',detection_method))
        output_path = os.path.join('results','logs',detection_method,scene_name)


    mkdir_if_missing(output_path)

    # resetting dataframe index 
    results_df=results_df.reset_index(drop=True)

    # reformatting df in wlh format
    h_col =  results_df.pop('h')
    results_df.insert(2,'h',h_col)

    # dumping output
    results_df.to_pickle(os.path.join(output_path,cat+'.pkl'))

def concat_results(args,data_dir,cat_list,nusc):
    '''
    cat_list: list of used categories for this run
    data_dir : output folder of logged track output. Data fetched at data_dir/<scene>/<cat>.pkl
    output at output/track_output_<detection_method>/track_results_nusc.json
    '''
    # cnt = 0
    # cnt_tot = 0
    meta_dict = {"use_camera": True,
        "use_lidar": False,
        "use_radar": True,
        "use_map": False, 
        "use_external": False, 
        }

    results_dict = dict()
    
    scenes_list = os.listdir(data_dir)

    for scene in nusc.scene:
        scene_name = scene['name']
        if scene_name not in scenes_list:   #only parsing scenes we used (i.e the corrrect dataset split)
            if args.verbose>=1:
                print('skipping',scene_name)
            continue

        print ('Processing scene', scene_name)
        scene_df = pd.DataFrame(columns =['w','l','h','x','y','z','vx','vy','ID','r1','r2','r3','r4','score','token','object'])

        for cat in cat_list:
            df_tmp = pd.read_pickle(os.path.join(data_dir,scene_name,cat+'.pkl'))
            df_tmp = df_tmp.drop(['theta','t'],axis=1)

            if df_tmp.empty:
                if args.verbose>=1:
                    print('no detection for %s in %s'%(cat,scene_name)) 
                continue

            df_tmp.loc[:,'object']=cat


            scene_df = pd.concat([scene_df,df_tmp])

        scene_df = scene_df.reset_index(drop=True)

        # cnt += len(scene_df.loc[scene_df['object']=='car'])
        # cnt_tot += len(scene_df)

        # Parsing scene token. Using nuscenes loop to go through all the tokens even if no detections
        first_token = scene['first_sample_token']
        sample_token = first_token
        sample = nusc.get('sample', first_token) # sample 0
        sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

        while(True):

            sample_data_token = sample_data['sample_token']

            if sample_data['is_key_frame'] == True:
        
                sample_result_list = []
                
                df_by_token = scene_df.loc[scene_df['token']==sample_data_token]

                for i in range(len(df_by_token)):
                    sample_df = df_by_token.iloc[i]

                    # Creating unique object ID (tracking ID + object number)
                    trk_id = str(sample_df['ID'])

                    if sample_df['object'] == 'car':
                        trk_id += '_1'
                    if sample_df['object'] == 'pedestrian':
                        trk_id += '_2'
                    if sample_df['object'] == 'truck':
                        trk_id += '_3'
                    if sample_df['object'] == 'bus':
                        trk_id += '_4'
                    if sample_df['object'] == 'bicycle':
                        trk_id += '_5'
                    if sample_df['object'] == 'motorcycle':
                        trk_id += '_6'
                    if sample_df['object'] == 'trailer':
                        trk_id += '_7'

                    sample_result={'sample_token':sample_data_token,
                                    'translation': [sample_df['x'],sample_df['y'],sample_df['z']],
                                    'size': [sample_df['w'],sample_df['l'],sample_df['h']],
                                    'rotation': [sample_df['r1'],sample_df['r2'],sample_df['r3'],sample_df['r4']],
                                    'velocity': [sample_df['vx'],sample_df['vy']],
                                    'tracking_id': trk_id,
                                    'tracking_name': sample_df['object'],
                                    'tracking_score': str(sample_df['score']) #random.random()
                            }

                    sample_result_list.append(sample_result)

                results_dict[sample_data_token] = sample_result_list

            if sample_data['next'] == "":
                #GOTO next scene
                break
            else:
                #GOTO next sample
                sample_data_token = sample_data['next']
                sample_data = nusc.get('sample_data', sample_data_token)

    output_dict={'meta':meta_dict, 'results':results_dict}

    output_dir = os.path.join('output','track_output_'+args.detection_method)

    if 'mini' in args.data_root: # mini dataset
        output_dir+= '_mini'

    mkdir_if_missing('output')
    mkdir_if_missing(output_dir)

    print('Dumping output at :', output_dir+'/track_results_nusc.json')
    with open(output_dir+'/track_results_nusc.json', 'w') as file: 
        json.dump(output_dict, file)

    print('Done.')


def log_args(args):
    output_dir = os.path.join('output','track_output_'+args.detection_method)

    if 'mini' in args.data_root: # mini dataset
        output_dir+= '_mini'


    mkdir_if_missing('output')

    mkdir_if_missing(output_dir)

    print('writing arguments to shell_output.txt file.')
    with open(output_dir+'/shell_output.txt', 'w') as f:
        f.write('Arguments :\n')
        f.write(str(args))
        f.write(3*'\n')
    print('Done')