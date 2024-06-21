import os 
import sys 
import json 
import numpy as np
import pandas as pd
import pickle
import math
import time
import random
from typing import List, Dict, Any
from shutil import copyfile
from pyquaternion import Quaternion
import argparse

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from PIL import Image
import colorsys

import cv2

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box, RadarPointCloud
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.data_classes import TrackingConfig

# load AN3DMOT model
from my_libs.my_model import AB3DMOT

############################################################################################################################################################################
# Utils
############################################################################################################################################################################

sampling_freq = 12 

'''
# def get_sensor_param(nusc, sample_token, cam_name='CAM_FRONT'):

#     sample = nusc.get('sample', sample_token)

#     # get camera sensor
#     cam_token = sample['data'][cam_name]
#     sd_record_cam = nusc.get('sample_data', cam_token)
#     cs_record_cam = nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
#     pose_record = nusc.get('ego_pose', sd_record_cam['ego_pose_token'])

#     return pose_record, cs_record_cam

# def get_sample_info(nusc,sensor,token,verbose=False):
#     scenes = nusc.scene
#     # print(scenes)
#     # input()
#     for scene in scenes:

#         first_sample = nusc.get('sample', scene['first_sample_token']) # sample 0
#         sample_data = nusc.get('sample_data', first_sample['data'][sensor])   # data for sample 0

#         while True:
#             if sample_data['sample_token']==token:
#                 if verbose :
#                     print('\nscene: ',scene)
#                     print('\nsample: ',first_sample)
#                     print ('\nsample_data: ',sample_data)
#                 return scene['name'], sample_data['filename']

#             if sample_data['next'] == "":
#                 #GOTO next scene
#                 # print("no next data")
#                 if verbose:
#                     print ('token NOT in:',scene['name'])
#                 break
#             else:
#                 #GOTO next sample
#                 next_token = sample_data['next']
#                 sample_data = nusc.get('sample_data', next_token)

#         # #Looping scene samples
#         # while(sample_data['next'] != ""):       
#         #     # if sample_token corresponds to token
#         #     if sample_data['sample_token']==token:

#         #         if verbose :
#         #             print('\nscene: ',scene)
#         #             print('\nsample: ',first_sample)
#         #             print ('\nsample_data: ',sample_data)
#         #         return scene['name'], sample_data['filename']

#         #     else:
#         #         # going to next sample
#         #         sample_data = nusc.get('sample_data', sample_data['next'])

#     return 0
'''

def mkdir_if_missing(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print("created directory at:",path)

def load_nusc(split,data_root):
    assert split in ['train','val','test'], "Bad nuScenes version"

    if split in ['train','val']:
        nusc_version = 'v1.0-trainval'
    elif split =='test':
        nusc_version = 'v1.0-test'
    
    nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=True)

    return nusc


def get_total_scenes_list(nusc,sensor):
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

def get_ego_pose(nusc,sensor,token,verbose=False):
    _, pose_record, _ = get_sample_metadata(nusc,sensor,token)
    return pose_record

def get_sensor_data(nusc,sensor,token,verbose=False):
    cs_record, _, cam_intrinsic = get_sample_metadata(nusc,sensor,token)
    return cs_record, cam_intrinsic


def get_det_df(cat_detection_root,detection_method,cat,det_file,verbose=False):

        f = open(os.path.join(cat_detection_root,detection_method+'_'+cat,det_file),'r')
        detection_list = []

        for det in f:
            # Extracting all values 
            t,x,y,z,w,l,h,r1,r2,r3,r4,vx,vy,score,token,_ = det.split(',')
            
            # Setting vertical velocity to 0
            vz = 0

            # Correcting box center from bottom center to 3D center
            z = float(z)+float(h)

            if verbose :
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

def get_det_df_at_t(cat_detection_root,detection_method,cat,det_file,t):
    df = get_det_df(cat_detection_root,detection_method,cat,det_file)
    df_at_t = df.loc[df['t']==t]
    df_at_t.reset_index(drop=True,inplace=True)
    return df_at_t

def get_gt_at_t(nusc,cat,t,sample,next_sample_tokens_dict):
    '''
    To get every detections at one frame we need the tokens from all the cameras. 
    To know these while not changing the pipeline too much from the regular tracking, this function returns a 
    dict containing those values for the next frame.
    The same dictionnary is taken as input at the next frame to know the sensor tokens at that frame.
    when the dictionnary is empty (i.e for the first sample of the scene) wee get the tokens from the sample argument
    '''

    sensor_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

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


def log_results(results_df,cat,scene_name,detection_method):
    '''
    results logged at results/logs/<detection_method>/<scene>/<cat>.pkl
    '''

    mkdir_if_missing('results')
    mkdir_if_missing(os.path.join('results','logs'))
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
    data fetched at results/logs/<detection_method>/<scene>/<cat>.pkl
    output at output/track_output_<detection_method>/track_results_nusc.json
    '''
    cnt = 0
    cnt_tot = 0
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

        cnt += len(scene_df.loc[scene_df['object']=='car'])
        cnt_tot += len(scene_df)

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

                    sample_result={'sample_token':token,
                                    'translation': [sample_df['x'],sample_df['y'],sample_df['z']],
                                    'size': [sample_df['w'],sample_df['l'],sample_df['h']],
                                    'rotation': [sample_df['r1'],sample_df['r2'],sample_df['r3'],sample_df['r4']],
                                    'velocity': [sample_df['vx'],sample_df['vy']],
                                    'tracking_id': trk_id,
                                    'tracking_name': sample_df['object'],
                                    'tracking_score': str(sample_df['score']) #random.random()
                            }

                    sample_result_list.append(sample_result)

                results_dict[token] = sample_result_list

            if sample_data['next'] == "":
                #GOTO next scene
                break
            else:
                #GOTO next sample
                sample_token = sample_data['next']
                sample_data = nusc.get('sample_data', sample_token)
    
    # print(cnt)
    # print(cnt_tot)
    # input()

    output_dict={'meta':meta_dict, 'results':results_dict}

    output_dir = os.path.join('output','track_output_'+args.detection_method)
    mkdir_if_missing('output')
    mkdir_if_missing(output_dir)

    print('Dumping output at :', output_dir+'/track_results_nusc.json')
    with open(output_dir+'/track_results_nusc.json', 'w') as file: 
        json.dump(output_dict, file)

    print('Done.')

    exit()


############################################################################################################################################################################
# Pipeline
############################################################################################################################################################################

def separate_det_by_cat_and_scene(args,detection_file,cat_list,nusc):

    output_root = args.cat_detection_root

    mkdir_if_missing(output_root)

    # load detection file
    print('opening results file at %s' % (detection_file))
    with open(detection_file) as json_file:
        '''
        Splits detection output by object categories.
        For each category, we further split the data by scene.
        '''
        det_data = json.load(json_file)
        num_frames = len(det_data['results'])

        for cat in cat_list:
            count = 0
            scene_mem = []

            print ('category: ',cat)
            cat_folder = args.detection_method+'_'+cat
            mkdir_if_missing(os.path.join(output_root,cat_folder))


            for scene in nusc.scene:
                # Extracting first scene sample
                nusc_sample = nusc.get('sample', scene['first_sample_token'])
                nusc_data = nusc.get('sample_data', nusc_sample['data'][args.sensor])
                # print(100*'*')
                # print(scene)
                # print()
                # print(nusc_sample)
                # print()

                while True:

                    if nusc_data['sample_token'] in det_data['results']:
                            
                        # opening file (r/w)
                        if scene['name'] not in scene_mem:
                            count=0     # timestamp
                            scene_mem.append(scene['name'])
                            f = open(os.path.join(output_root,cat_folder,scene['name']+'.txt'),'w')     # rewritting file
                        else :
                            count+=1        # incrementing timestamp
                            f = open(os.path.join(output_root,cat_folder,scene['name']+'.txt'),'a')     # rewritting file

                        # Logging detections
                        det_cnt = 0
                        sample_token = nusc_data['sample_token']
                        for det_sample in det_data['results'][sample_token]:
                            
                            if det_sample['detection_name']==cat and det_sample['detection_score']>=args.score_thresh:

                                det_cnt+=1          #detection counter

                                f.write(str(count))
                                for item in det_sample['translation']:
                                    f.write(',%s'%(str(item)))
                                for item in det_sample['size']:
                                    f.write(',%s'%(str(item)))
                                for item in det_sample['rotation']:
                                    f.write(',%s'%(str(item)))
                                for item in det_sample['velocity']:
                                    f.write(',%s'%(str(item)))
                                f.write(',%s'%(str(det_sample['detection_score'])))
                                f.write(',%s'%(str(sample_token)))
                                f.write(',\n')
                        
                        f.close()
                        print('\nfound %d detections of category:'%(det_cnt),cat,',in scene:',scene['name'],'token:',sample_token)
                        print('results logged at: ',f.name)
                        print('corresponding image:',nusc_data['filename'])
                        print('t = ',count)
                        # input()


                    if nusc_data['next'] == "":
                        #GOTO next scene
                        print("no next data in scene %s"%(scene['name']))
                        break
                    else:
                        #GOTO next sample
                        next_token = nusc_data['next']
                        nusc_data = nusc.get('sample_data', next_token)


    print(100*'#','\nfinished separating detections\n',100*('#'))
    exit()

def initialize_tracker(args, cat, ID_start, nusc, det_file):

    tracker = AB3DMOT(args, cat, ID_init=ID_start) 

    assert det_file in list(scene['name']+'.txt' for scene in nusc.scene)

    for scene in nusc.scene:
        if scene['name']+'.txt' == det_file:
            first_sample_token = scene['first_sample_token']
            # last_sample_token = scene['last_sample_token']
            break

    return tracker, scene, first_sample_token


def tracking(args,cat_list,nusc):
    for cat in cat_list:        

        print("category: ",cat)
        det_file_list=get_scenes_list(os.path.join(args.cat_detection_root,args.detection_method+'_'+cat))

        for det_file in det_file_list:
            
            # Computation time for this file
            comp_time_start = time.time()
            
            # if det_file != 'scene-0103.txt':    # DEBUG
            #     continue

            # initializing output dataframe
            results_df = pd.DataFrame(columns =['h','w','l','x','y','z','theta','vx','vy','ID','r1','r2','r3','r4','score','token','t'])    # full results for this scene

            # initializing AB3DDMOT
            tracker, scene, first_token = initialize_tracker(args=args, cat=cat, ID_start=1, nusc=nusc, det_file=det_file)

            if args.verbose>=3:
                print ('initial trackers :',tracker.trackers)
            
            print ('scene :',scene['name'])

            if args.verbose>=2:
                print (scene)

            # first token init
            t=0
            sample_token = first_token
            sample = nusc.get('sample', first_token) # sample 0
            sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

            while(True):

                print ('t = ',t)

                # if t<25:                 # DEBUG
                #     #GOTO next sample
                #     sample_token = sample_data['next']
                #     sample_data = nusc.get('sample_data', sample_token)
                #     t+=1
                #     continue

                if args.verbose>=3:
                    print(200*'*','\n')
                    print('Sample: ',sample) 
                    print('\nSample data:',sample_data)
                    print('\n',200*'-','\n')

                metadata_token = sample_data['token']
                cs_record, ego_pose, cam_intrinsic = get_sample_metadata(nusc,args.sensor,metadata_token,verbose=False)

                det_df = get_det_df_at_t(args.cat_detection_root,args.detection_method,cat,det_file,t)
                # gt_df = get_gt_at_t(cat,t,sample,sample_data,cs_record,ego_pose)

                if args.verbose>=2:
                    print('\n',200*'-','\n')
                    print('Ego pose:',ego_pose)
                    print('\nCalibrated Sensor record:',cs_record)
                    print('\nCam intrinsic:',cam_intrinsic)
                    print('\n',200*'-','\n')
                
                if args.verbose>=1:
                    if len(det_df)>0:
                        print('\n',200*'-','\n')
                        print('Detection dataframe:')
                        print(det_df)
                        print('\nDetection token:',det_df['token'][0])
                        print('\n',200*'-','\n')
                    else :
                        print('\n',200*'-','\n')
                        print('No detection for this sample:')
                        print('\n',200*'-','\n')

                    print('resume tracking :')


                results, affi = tracker.track(det_df, t, scene['name'], args.verbose)

                # displaying results
                if len(results)>0:
                    results_df_at_t = pd.DataFrame(results,columns =['h','w','l','x','y','z','theta','vx','vy','ID','r1','r2','r3','r4','score','token','t'])
                    results_df_at_t = results_df_at_t.iloc[::-1]    # flip rows
                    results_df_at_t = results_df_at_t.reset_index(drop=True)

                    if args.verbose>=1:
                        print('\n',200*'-','\n')
                        print ('tracking results:\n',results_df_at_t)
                        print('\n',200*'-','\n')
       
                else :
                    # reinitialize results_df_at_t 
                    results_df_at_t = pd.DataFrame(columns =['h','w','l','x','y','z','theta','vx','vy','ID','r1','r2','r3','r4','score','token','t'])
                    
                    if args.verbose>=1:
                        print('\n',200*'-','\n')
                        print ('tracking results:\n',results)
                        print('\n',200*'-','\n')
                
                if args.verbose>=2:
                    print ('affinity matrix:',affi)
                    print('\n',200*'-','\n')


                # logging results for metrics
                if sample_data['is_key_frame'] == True:         # Only using samples for eval

                    # Correcting t and sample token values for un-updated tracklet at this frame (still in memory)
                    results_df_at_t['t']=t 
                    results_df_at_t['token']=sample_data['sample_token']
                    
                    # Adding to global results
                    results_df=pd.concat([results_df,results_df_at_t])
                    
                    if args.verbose>=3:
                        print(100*'$')
                        print(results_df)


                if args.log_viz:
                    log_tracking_visualization(args=args,
                                                nusc=nusc,
                                                data_root=args.data_root,
                                                sample_data=sample_data,
                                                results=results,
                                                cs_record=cs_record,
                                                cam_intrinsic=cam_intrinsic,
                                                ego_pose=ego_pose,
                                                cat=cat,
                                                scene_name=scene['name'],
                                                t=t
                                                )
                elif args.viz :
                    tracking_visualization(args=args,
                                            nusc=nusc,
                                            data_root=args.data_root,
                                            sample_data=sample_data,
                                            results=results,
                                            cs_record=cs_record,
                                            cam_intrinsic=cam_intrinsic,
                                            ego_pose=ego_pose,
                                            det_df=det_df,
                                            score_thresh=args.score_thresh,
                                            t=t
                                            )


                if sample_data['next'] == "":
                    #GOTO next scene
                    print("no next data")
                    comp_time_end = time.time() - comp_time_start
                    comp_time_by_det = divmod(comp_time_end,60)                    
                    print('handled %d frames in %d m %.2f s'%(t,comp_time_by_det[0],comp_time_by_det[1]))

                    # Logging computation time
                    mkdir_if_missing('results')
                    mkdir_if_missing(os.path.join('results','logs'))

                    if 'time'+args.detection_method+'.txt' in os.listdir(os.path.join('results','logs')):
                        f = open('./results/logs/time'+args.detection_method+'.txt','a')
                    else :
                        f = open('./results/logs/time'+args.detection_method+'.txt','w')

                    f.write('%s, %s : handled %d frames in %dm %.2fs'%(scene['name'], cat, t, comp_time_by_det[0], comp_time_by_det[1]))
                    f.write(',\n')
                    f.close()

                    break
                else:
                    #GOTO next sample
                    sample_token = sample_data['next']
                    sample_data = nusc.get('sample_data', sample_token)
                    t+=1

            # Logging results for this scene
            log_results(results_df,cat,scene['name'],args.detection_method)

def gt_tracking(args,cat_list,nusc):
    for cat in cat_list:        

        print("category: ",cat)
        # det_file_list = list(scene['name']+'.txt' for scene in nusc.scene)

        # det_file_list = ['scene-0553.txt','scene-0796.txt','scene-0103.txt', 'scene-0916.txt']
        det_file_list = ['scene-0103.txt','scene-0553.txt','scene-0796.txt','scene-0916.txt']

        for det_file in det_file_list:
            
            # Computation time for this file
            comp_time_start = time.time()

            # dictionary used to get every gt at a frame
            next_sample_tokens_dict = dict()

            # if det_file != 'scene-0103.txt':    # DEBUG
            #     continue
                        
            results_df = pd.DataFrame(columns =['h','w','l','x','y','z','theta','vx','vy','ID','r1','r2','r3','r4','score','token','t'])    # full results for this scene

            # getting tracker scene and token using nuscene indexing
            tracker, scene, first_token = initialize_tracker(args=args, cat=cat, ID_start=1, nusc=nusc, det_file=det_file)

            # print ('initial trackers :',tracker.trackers)
            print ('scene :',scene['name'])
            
            if args.verbose>=2:
                print (scene)

            t=0
            sample_token = first_token
            sample = nusc.get('sample', first_token) # sample 0
            sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

            while(True):

                print ('t = ',t)

                if args.verbose>=3:
                    print(200*'*','\n')
                    print('Sample: ',sample) 
                    print('\nSample data:',sample_data)
                    print('\n',200*'-','\n')


                metadata_token = sample_data['token']
                cs_record, ego_pose, cam_intrinsic = get_sample_metadata(nusc,args.sensor,metadata_token,verbose=False)

                gt_df, next_sample_tokens_dict = get_gt_at_t(nusc,cat,t,sample,next_sample_tokens_dict)

                if args.verbose>=1:
                    if len(gt_df)>0:
                        print('\n',200*'-','\n')
                        print('Detection dataframe:')
                        print(gt_df)
                        print('\nDetection token:',gt_df['token'][0])
                        print('\n',200*'-','\n')
                    else :
                        print('\n',200*'-','\n')
                        print('No detection for this sample:')
                        print('\n',200*'-','\n')

                    print('resume tracking :')


                results, affi = tracker.track(gt_df, t, scene['name'], args.verbose)

                # displaying results
                if len(results)>0:
                    results_df_at_t = pd.DataFrame(results,columns =['h','w','l','x','y','z','theta','vx','vy','ID','r1','r2','r3','r4','score','token','t'])
                    results_df_at_t = results_df_at_t.iloc[::-1]
                    results_df_at_t = results_df_at_t.reset_index(drop=True)

                    if args.verbose>=1:
                        print('\n',200*'-','\n')
                        print ('tracking results:\n',results_df_at_t)
                        print('\n',200*'-','\n')
     
                else :
                    # reinitialize results_df_at_t 
                    results_df_at_t = pd.DataFrame(columns =['h','w','l','x','y','z','theta','vx','vy','ID','r1','r2','r3','r4','score','token','t'])
                    
                    if args.verbose>=1:
                        print('\n',200*'-','\n')
                        print ('tracking results:\n',results)
                        print('\n',200*'-','\n')

                if args.verbose>=2:
                    print ('affinity matrix:',affi)
                    print('\n',200*'-','\n')


                # logging results for metrics
                if sample_data['is_key_frame'] == True:         # Only using samples for eval

                    # Correcting t and sample token values for un-updated tracklet at this frame (still in memory)
                    results_df_at_t['t']=t 
                    results_df_at_t['token']=sample_data['sample_token']

                    # Adding to global results
                    results_df=pd.concat([results_df,results_df_at_t])

                    if args.verbose>=3:
                        print(100*'$')
                        print(results_df)

                if args.log_viz:
                    log_tracking_visualization(args=args,
                                                nusc=nusc,
                                                data_root=args.data_root,
                                                sample_data=sample_data,
                                                results=results,
                                                cs_record=cs_record,
                                                cam_intrinsic=cam_intrinsic,
                                                ego_pose=ego_pose,
                                                cat=cat,
                                                scene_name=scene['name'],
                                                t=t
                                                )
                elif args.viz :
                    tracking_visualization(args=args,
                                            nusc=nusc,
                                            data_root=args.data_root,
                                            sample_data=sample_data,
                                            results=results,
                                            cs_record=cs_record,
                                            cam_intrinsic=cam_intrinsic,
                                            ego_pose=ego_pose,
                                            det_df=gt_df,
                                            score_thresh=args.score_thresh,
                                            t=t
                                            )


                if sample_data['next'] == "":
                    #GOTO next scene
                    print("no next data")
                    comp_time_end = time.time() - comp_time_start
                    comp_time_by_det = divmod(comp_time_end,60)                    
                    print('handled %d frames in %d m %.2f s'%(t,comp_time_by_det[0],comp_time_by_det[1]))

                    # Logging computation time
                    mkdir_if_missing('results')
                    mkdir_if_missing(os.path.join('results','logs'))

                    if 'time'+args.detection_method+'.txt' in os.listdir(os.path.join('results','logs')):
                        f = open('./results/logs/time'+args.detection_method+'.txt','a')
                    else :
                        f = open('./results/logs/time'+args.detection_method+'.txt','w')

                    f.write('%s, %s : handled %d frames in %dm %.2fs'%(scene['name'], cat, t, comp_time_by_det[0], comp_time_by_det[1]))
                    f.write(',\n')
                    f.close()

                    break
                else:
                    #GOTO next sample
                    sample_token = sample_data['next']
                    sample_data = nusc.get('sample_data', sample_token)
                    t+=1

            # Logging results for this scene
            log_results(results_df,cat,scene['name'],args.detection_method)    


def tracking_visualization(args,nusc,data_root,sample_data,results,cs_record,cam_intrinsic,ego_pose,det_df,score_thresh,t):

    image_path = os.path.join(data_root,sample_data['filename'])    
    img = cv2.imread(image_path) 

    max_color = 30
    # colors = random_colors(max_color)       # Generate random colors
    colors = fixed_colors()                   # Using pre-made color list to identify ID

    # Tracking bboxes
    for res in results:

        h,w,l,x,y,z,theta,vx,vy,ID,r1,r2,r3,r4,score,token,t_res=res
        q = Quaternion(r1,r2,r3,r4)

        box = Box(center = [x,y,z],
                    size = [w,l,h],
                    orientation = q,
                    label = ID,
                    score = score,
                    velocity = [vx,vy,0],
                    name = ID
                    )
        # Move box to ego vehicle coord system.
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        
        color_float = colors[(int(ID)-1) % max_color]           # loops back to first color if more than max_color
        color_int = tuple([int(tmp * 255) for tmp in color_float])
        c = color_int

        if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
            box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
        

    # Detection bboxes
    '''
    for i in range(len(det_df)):
        det = det_df.loc[i]
        score = det['score']
        if score>score_thresh:

            h = det['h']
            w = det['w']
            l = det['l']
            x = det['x']
            y = det['y']
            z = det['z'] 

            vx = det['vx']
            vy = det['vy']
            vz = det['vz']

            r1 = det['r1']
            r2 = det['r2']
            r3 = det['r3']
            r4 = det['r4']
            q = Quaternion(r1,r2,r3,r4)

            box = Box(center = [x,y,z],
                        size = [w,l,h],
                        orientation = q,
                        label = i,
                        score = score,
                        velocity = [vx,vy,vz],
                        name = i
                        )
            # print('xyz det :',box.center,' score:', box.score, 'velocity:', box.velocity, 'angle:', box.orientation.degrees)

            # Move box to ego vehicle coord system.
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)
            
            # print('xyz det :',box.center,' score:', box.score, 'velocity:', box.velocity, 'angle:', box.orientation.degrees)
            c = (0,0,0)
            # box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)

            # exit()
    '''
    
    while True :
        # showing the image 
        cv2.imshow('image', img) 
          
        # waiting using waitKey method 
        if cv2.waitKey(1) == ord("\r"):
            break

    cv2.destroyAllWindows()

def log_tracking_visualization(args,nusc,data_root,sample_data,results,cs_record,cam_intrinsic,ego_pose,scene_name,cat,t):
    image_path = os.path.join(data_root,sample_data['filename'])    

    img = cv2.imread(image_path) 

    max_color = 30
    colors = fixed_colors()

    # Tracking bboxes
    for res in results:
        # print(res)
        h,w,l,x,y,z,theta,vx,vy,ID,r1,r2,r3,r4,score,token,t_res=res
        q = Quaternion(r1,r2,r3,r4)

        box = Box(center = [x,y,z],
                    size = [w,l,h],
                    orientation = q,
                    label = ID,
                    score = score,
                    velocity = [vx,vy,0],
                    name = ID
                    )
        # Move box to ego vehicle coord system.
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        
        # generating colors
        color_float = colors[(int(ID)-1) % max_color]           # loops back to first color if more than max_color
        color_int = tuple([int(tmp * 255) for tmp in color_float])
        c = color_int

        if t_res == t and box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
            box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
    

    # Results at results/visualization_method/scene/cat/*.png
    mkdir_if_missing('results')
    viz_path = 'results/visualization_'+args.detection_method
    mkdir_if_missing(viz_path)

    mkdir_if_missing(os.path.join(viz_path,scene_name))

    output_path = os.path.join(viz_path,scene_name,cat)
    mkdir_if_missing(output_path)
    cv2.imwrite(os.path.join(output_path,sample_data['filename'].split('/')[-1]),img)

def detection_visualization (args, cat_list, nusc):

    for cat in cat_list:        

        print("category: ",cat)
        det_file_list=get_scenes_list(os.path.join(args.cat_detection_root,args.detection_method+'_'+cat))

        for det_file in det_file_list:
            
            for item in nusc.scene:
                if det_file.split('.')[0] == item['name']:
                    scene = item

            t=0
            first_token = scene['first_sample_token']
            sample = nusc.get('sample', first_token) # sample 0
            sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

            while(True):

                print ('t=', t)

                metadata_token = sample_data['token']
                cs_record, ego_pose, cam_intrinsic = get_sample_metadata(nusc,args.sensor,metadata_token,verbose=False)

                det_df = get_det_df_at_t(args.cat_detection_root,args.detection_method,cat,det_file,t)
                
                image_path = os.path.join(args.data_root,sample_data['filename'])

                # read the image 
                img = cv2.imread(image_path) 

                _, nusc_box_list,_ = nusc.get_sample_data(sample_data_token = sample_data['token'])

                print()
                print(*nusc_box_list,sep='\n')
                print()

                # displaying nuscenes gt
                for obj in nusc_box_list: 
                    if (obj.name).split('.')[1]==cat : 
                        print('gt:',obj.center, 'angle:',obj.orientation.degrees)      
                        c = (0,255,255)
                        obj.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)

                # displaying detections
                for i in range(len(det_df)):
                    det = det_df.loc[i]
                    score = det['score']

                    h = det['h']
                    w = det['w']
                    l = det['l']

                    x = det['x']
                    y = det['y']
                    z = det['z'] 

                    r1 = det['r1']
                    r2 = det['r2']
                    r3 = det['r3']
                    r4 = det['r4']

                    q = Quaternion(r1,r2,r3,r4)

                    box = Box(center = [x,y,z],
                                size = [w,l,h],
                                orientation = q,
                                label = i,
                                score = score,
                                name = i
                                )
                    # Move box to ego vehicle coord system.
                    box.translate(-np.array(ego_pose['translation']))
                    box.rotate(Quaternion(ego_pose['rotation']).inverse)

                    #  Move box to sensor coord system.
                    box.translate(-np.array(cs_record['translation']))
                    box.rotate(Quaternion(cs_record['rotation']).inverse)
                    
                    # print('xyz det :',box.center,' score:', box.score, 'angle:', box.orientation.degrees)
                    c = np.array([90,200,220]) / 255.0
            
                    if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
                        box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
            
                while True :
                    # showing the image 
                    cv2.imshow('image', img) 

                    if cv2.waitKey(1) == ord("\r"):
                        break

                cv2.destroyAllWindows()

                if sample_data['next'] == "":
                    #GOTO next scene
                    print("no next data")
                    break
                else:
                    #GOTO next sample
                    sample_token = sample_data['next']
                    sample_data = nusc.get('sample_data', sample_token)
                    t+=1

############################################################################################################################################################################
# Main
############################################################################################################################################################################

# sensor_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

def create_parser():

    parser = argparse.ArgumentParser()
    
    # nuScenes loading
    parser.add_argument('--data_root', type=str, default='./data/nuScenes', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='train', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='train_val/test')

    # Detection config
    parser.add_argument('--detection_method', type=str, default='CRN', help='detection method')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='minimum detection confidence')
    parser.add_argument('--cat_detection_root', type=str, default='./data/cat_detection/', help='category-splitted detection folder')

    # Tracking config
    parser.add_argument('--use_vel', action='store_true', dest='use_vel', default=True, help='use radar velocity for prediction')
    parser.add_argument('--no-use_vel', action='store_false', dest='use_vel', default=False, help='use radar velocity for prediction')

    parser.add_argument('--affi_pro', action='store_true', dest='affi_pro', default=True, help='use post-processing affinity')
    parser.add_argument('--no-affi_pro', action='store_false', dest='affi_pro', default=True, help='do not use post-processing affinity')

    # Action
    parser.add_argument('--go_sep', action='store_true', default=False, help='separate detections by category (required once)')
    parser.add_argument('--gt_track', action='store_true', default=False, help='tracking using ground thruth instead of detections (debug)')
    parser.add_argument('--log_viz', action='store_true', default=False, help='Logging tracking visualization (saving .png files) directly instead of displaying')
    parser.add_argument('--viz', action='store_true', default=False, help='display tracking visualization (superseeded by log_viz)')
    parser.add_argument('--det_viz', action='store_true', default=False, help='display detection visualization')
    parser.add_argument('--concat', action='store_true', default=False, help='Concatenate results for evaluation')

    # debug and display
    parser.add_argument('--debug','-d' , action='store_true', default=False, help='debug')
    parser.add_argument('--verbose','-v' , action='count', default=0, help='verbosity level')


    # other exp param
    parser.add_argument('--run_hyper_exp','-hexp' , action='store_true', default=False, help='running an experiment testing hyperparam')
    parser.add_argument('--use_R','-R' , action='store_true', default=False, help='use measurement noise matrix') #to remove --> ablation study
    parser.add_argument('--metric', type=str, default='giou_3d', help='[dist_3d, dist_2d, m_dis, iou_2d, iou_3d, giou_2d, giou_3d]') 
    parser.add_argument('--thresh', type=float, default=None, help='distance treshold, bounds depend on metric') 
    parser.add_argument('--min_hits', type=int, default=None, help='min hits') 
    parser.add_argument('--max_age', type=int, default=None, help='max memory in frames') 

    return parser



if __name__ == '__main__':

    # cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'construction_vehicle', 'motorcycle', 'trailer']
    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']
    # cat_list = ['bicycle']

    parser = create_parser()
    args = parser.parse_args()

    assert args.metric in ['dist_3d', 'dist_2d', 'm_dis', 'iou_2d', 'iou_3d', 'giou_2d', 'giou_3d'], 'wrong metric type'
    assert args.min_hits == None or args.min_hits > 0, 'must be postive integer' 
    assert args.max_age == None or args.max_age > 0, 'must be postive integer' 

    nusc = load_nusc(args.split,args.data_root)

    # Separation of all detections by their categories and scenes (required first step)
    if args.go_sep == True :
        separate_det_by_cat_and_scene(args,
                                    detection_file = './data/detection_output/results_nusc.json',
                                    cat_list=cat_list,
                                    nusc=nusc
                                    )

    if args.det_viz:
        detection_visualization(args,
                                cat_list=cat_list,
                                nusc=nusc
                                )

    # Running evaluation on tracking data (needs to be generated first)
    if args.concat:
        concat_results(args,
                        data_dir='./results/logs/'+args.detection_method,
                        cat_list=cat_list,
                        nusc=nusc
                        )

    # Tracking pipeline, with custom deteciton or with ground truth for debugging purposes
    if args.gt_track == False :
        # Tracking using CNN detections
        tracking(args,
                cat_list=cat_list,
                nusc=nusc
                )
    else:
        # Tracking using Ground truth detections
        gt_tracking(args,
                    cat_list=cat_list,
                    nusc=nusc
                    )
