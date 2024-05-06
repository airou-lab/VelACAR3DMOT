import os 
import sys 
import json 
import numpy as np
import pandas as pd
import pickle
import math
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

# load AN3DMOT model
from my_libs.my_model import AB3DMOT

############################################################################################################################################################################
# Utils
############################################################################################################################################################################

sampling_freq = 12 

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
                    print('sensor token:',sample_data['calibrated_sensor_token'],'\n')

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
            vz = 0

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
                                float(x),float(y),float(z),
                                float(w),float(l),float(h),
                                float(r1),float(r2),float(r3),float(r4),
                                float(vx)/sampling_freq,float(vy)/sampling_freq,vz, # compensating sampling frequency
                                float(score),
                                token])


        detection_df = pd.DataFrame(detection_list,columns =['t','x','y','z','w','l','h','r1','r2','r3','r4','vx','vy','vz','score','token'])
        return(detection_df)

def get_det_df_at_t(cat_detection_root,detection_method,cat,det_file,t):
    df = get_det_df(cat_detection_root,detection_method,cat,det_file)
    df_at_t = df.loc[df['t']==t]
    df_at_t.reset_index(drop=True,inplace=True)
    return df_at_t

def get_gt_at_t(cat,t,sample,sample_data,cs_record,ego_pose):

    _, nusc_box_list,_ = nusc.get_sample_data(sample_data_token = sample_data['token'])

    GT_list=[]

    for box in nusc_box_list:
        if (box.name).split('.')[1]==cat:

            # print(box)
            
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
            score = 1
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
    return(GT_df)




    exit()


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

    return color_val


def compute_metrics(results_df):
    return 0


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
                # print(scene['name'])
                # input()

                while True:

                    if nusc_data['sample_token'] in det_data['results']:
                        
                        # opening file (r/w)
                        if scene['name'] not in scene_mem:
                            count=0
                            scene_mem.append(scene['name'])
                            f = open(os.path.join(output_root,cat_folder,scene['name']+'.txt'),'w')     # rewritting file
                        else :
                            count+=1        # incrementing token counter
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
                                
                        print('\nfound %d detections of category:'%(det_cnt),cat,',in scene:',scene['name'],'token:',sample_token)
                        print('results logged at: ',f.name)
                        print('corresponding image:',nusc_data['filename'])
                        print('count = ',count)
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

def initialize_tracker(data_root, cat, ID_start, nusc, det_file):

    tracker = AB3DMOT(cat, ID_init=ID_start) 

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
        results_df = pd.DataFrame(columns =['w','l','h','x','y','z','theta','vx','vy','ID','r1','r2','r3','r4','score','token','t'])

        for det_file in det_file_list:
        
            # if det_file != 'scene-0103.txt':    # DEBUG
                # continue

            tracker, scene, first_token = initialize_tracker(data_root=os.path.join(args.cat_detection_root,args.detection_method+'_'+cat),cat=cat, 
                                                            ID_start=0, nusc=nusc, det_file=det_file)

            # print ('initial trackers :',tracker.trackers)
            print ('scene :',scene['name'])
            print (scene)

            t=0
            sample_token = first_token
            sample = nusc.get('sample', first_token) # sample 0
            sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

            while(True):

                print ('t = ',t)

                # if t<95:                 # DEBUG
                #     #GOTO next sample
                #     sample_token = sample_data['next']
                #     sample_data = nusc.get('sample_data', sample_token)
                #     t+=1
                #     continue

                print(200*'*','\n')
                print('Sample: ',sample) 
                print('\nSample data:',sample_data)
                print('\n',200*'-','\n')

                metadata_token = sample_data['token']
                cs_record, ego_pose, cam_intrinsic = get_sample_metadata(nusc,args.sensor,metadata_token,verbose=False)

                det_df = get_det_df_at_t(args.cat_detection_root,args.detection_method,cat,det_file,t)
                # gt_df = get_gt_at_t(cat,t,sample,sample_data,cs_record,ego_pose)

                # print('\n',200*'-','\n')
                # print('Ego pose:',ego_pose)
                # print('\nCalibrated Sensor record:',cs_record)
                # print('\nCam intrinsic:',cam_intrinsic)
                # print('\n',200*'-','\n')

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


                results, affi = tracker.track(det_df, t, scene['name'])

                # displaying results
                if len(results[0])>0:
                    results_df_at_t = pd.DataFrame(results[0],columns =['w','l','h','x','y','z','theta','vx','vy','ID','r1','r2','r3','r4','score','token','t'])
                    results_df_at_t = results_df_at_t.iloc[::-1]
                    results_df_at_t = results_df_at_t.reset_index(drop=True)

                    print('\n',200*'-','\n')
                    print ('tracking results:\n',results_df_at_t)
                    # print ('affinity matrix:',affi)
                    print('\n',200*'-','\n')        
                else :
                    print('\n',200*'-','\n')
                    print ('tracking results:\n',results)
                    # print ('affinity matrix:',affi)
                    print('\n',200*'-','\n')


                # logging results for metrics
                results_df_at_t['t']=t
                results_df=results_df.append(results_df_at_t)
                print(100*'$')
                print(results_df)


                if args.log_viz:
                    log_tracking_visualization(nusc=nusc,
                                                data_root=args.data_root,
                                                sample_data=sample_data,
                                                results=results[0],
                                                cs_record=cs_record,
                                                cam_intrinsic=cam_intrinsic,
                                                ego_pose=ego_pose,
                                                cat=cat,
                                                scene_name=scene['name'],
                                                t=t
                                                )
                elif args.viz :
                    tracking_visualization(nusc=nusc,
                                            data_root=args.data_root,
                                            sample_data=sample_data,
                                            results=results[0],
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
                    compute_metrics(results_df)
                    break
                else:
                    #GOTO next sample
                    sample_token = sample_data['next']
                    sample_data = nusc.get('sample_data', sample_token)
                    t+=1

def gt_tracking(args,cat_list,nusc):
    for cat in cat_list:        

        print("category: ",cat)
        det_file_list=get_scenes_list(os.path.join(args.cat_detection_root,args.detection_method+'_'+cat))

        for det_file in det_file_list:
        
            if det_file != 'scene-0103.txt':    # DEBUG
                continue

            tracker, scene, first_token = initialize_tracker(data_root=os.path.join(args.cat_detection_root,args.detection_method+'_'+cat),cat=cat, 
                                                            ID_start=0, nusc=nusc, det_file=det_file)

            # print ('initial trackers :',tracker.trackers)
            print ('scene :',scene['name'])
            print (scene)

            t=0
            sample_token = first_token
            sample = nusc.get('sample', first_token) # sample 0
            sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

            while(True):

                print ('t = ',t)

                if t<95:                 # DEBUG
                    #GOTO next sample
                    sample_token = sample_data['next']
                    sample_data = nusc.get('sample_data', sample_token)
                    t+=1
                    continue

                print(200*'*','\n')
                print('Sample: ',sample) 
                print('\nSample data:',sample_data)
                print('\n',200*'-','\n')

                metadata_token = sample_data['token']
                cs_record, ego_pose, cam_intrinsic = get_sample_metadata(nusc,args.sensor,metadata_token,verbose=False)

                # det_df = get_det_df_at_t(args.cat_detection_root,args.detection_method,cat,det_file,t)
                gt_df = get_gt_at_t(cat,t,sample,sample_data,cs_record,ego_pose)

                # print('\n',200*'-','\n')
                # print('Ego pose:',ego_pose)
                # print('\nCalibrated Sensor record:',cs_record)
                # print('\nCam intrinsic:',cam_intrinsic)
                # print('\n',200*'-','\n')

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


                results, affi = tracker.track(gt_df, t, scene['name'])

                print('\n',200*'-','\n')
                print ('tracking results:',results)
                # print ('affinity matrix:',affi)
                print('\n',200*'-','\n')                

                if args.log_viz:
                    log_tracking_visualization(nusc=nusc,
                                                data_root=args.data_root,
                                                sample_data=sample_data,
                                                results=results[0],
                                                cs_record=cs_record,
                                                cam_intrinsic=cam_intrinsic,
                                                ego_pose=ego_pose,
                                                cat=cat,
                                                scene_name=scene['name'],
                                                t=t
                                                )
                elif args.viz:
                    tracking_visualization(nusc=nusc,
                                            data_root=args.data_root,
                                            sample_data=sample_data,
                                            results=results[0],
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
                    break
                else:
                    #GOTO next sample
                    sample_token = sample_data['next']
                    sample_data = nusc.get('sample_data', sample_token)
                    t+=1


def tracking_visualization(nusc,data_root,sample_data,results,cs_record,cam_intrinsic,ego_pose,det_df,score_thresh,t):

    image_path = os.path.join(data_root,sample_data['filename'])    
    img = cv2.imread(image_path) 


    max_color = 30
    # colors = random_colors(max_color)       # Generate random colors
    colors = fixed_colors()                   # Using pre-made color list to identify ID

    # img_data = Image.open(image_path)
    # read the image 

    # _, ax = plt.subplots(1, 1, figsize=(16, 9))   
    # sd_record = nusc.get('sample_data', sample_data['token'])
    # cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    # imsize = (sd_record['width'], sd_record['height'])

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
        
        color_float = colors[int(ID) % max_color]           # loops back to first color if more than max_color
        color_int = tuple([int(tmp * 255) for tmp in color_float])
        c = color_int
        # box.render(ax,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)

        # if t_res == t:  # Filtering out tracklets with no detection for that frame
        #     if box.center[2]>0: # z value (front) cannot be < 0  
        #         box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
        # print(box)
        # input()
        if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
            box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)

    # ax.imshow(img_data)
    # plt.show()

    # Detection bboxes
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
            # z = -0.5

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



    
    while True :
        # showing the image 
        cv2.imshow('image', img) 
          
        # waiting using waitKey method 
        if cv2.waitKey(1) == ord("\r"):
            break

        # cv2.waitKey(200)
        # break

    cv2.destroyAllWindows()

    # exit()

def detection_visualization (nusc,data_root,sample_data,det_df,ego_pose,token,cat):
    image_path = os.path.join(data_root,sample_data['filename'])

    # read the image 
    img = cv2.imread(image_path) 

    sd_record = nusc.get('sample_data', sample_data['token'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    print(sd_record)
    print(cs_record)
    # input()

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


            #  Move box to sensor coord system.
            obj.rotate(Quaternion(cs_record['rotation']))
            obj.translate(np.array(cs_record['translation']))

            # Move box to ego vehicle coord system.
            obj.rotate(Quaternion(ego_pose['rotation']))
            obj.translate(np.array(ego_pose['translation']))

            print('gt in global coord:',obj.center,' ', obj.orientation.degrees)


            # # Move box to ego vehicle coord system.
            # obj.translate(-np.array(ego_pose['translation']))
            # obj.rotate(Quaternion(ego_pose['rotation']).inverse)

            # #  Move box to sensor coord system.
            # obj.translate(-np.array(cs_record['translation']))
            # obj.rotate(Quaternion(cs_record['rotation']).inverse)

            # print('gt in camera coord:',obj.center,' ', obj.orientation.degrees)


    # displaying detections
    for i in range(len(det_df)):
        det = det_df.loc[i]
        score = det['score']
        if score>0.5:

            h = det['h']
            w = det['w']
            l = det['l']
            x = det['x']
            y = det['y']
            z = det['z'] 
            # z = 0.5
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

            print(100*'-')
            print('global cord (detections)',(x,y,z),'|',(w,l,h))
            print()
            print('ego pose box center :',box.center)
            print()
            print('ego pose box size :',box.wlh)
            print()

            #  Move box to sensor coord system.
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)
            
            print('xyz det :',box.center,' score:', box.score, 'angle:', box.orientation.degrees)
            c = np.array([90,200,220]) / 255.0

            if box.center[2]>0: # z value (front) cannot be < 0  
                box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
    
                while True :
                    # showing the image 
                    cv2.imshow('image', img) 
                      
                    # # waiting using waitKey method 
                    # if cv2.waitKey(1) == ord("\r"):
                    #     break

                    if cv2.waitKey(1) == ord("\r"):
                        break

                cv2.destroyAllWindows()
    # exit()

def log_tracking_visualization(nusc,data_root,sample_data,results,cs_record,cam_intrinsic,ego_pose,scene_name,cat,t):
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
        color_float = colors[int(ID) % max_color]           # loops back to first color if more than max_color
        color_int = tuple([int(tmp * 255) for tmp in color_float])
        c = color_int

        if t_res == t:  # Filtering out tracklets with no detection for that frame
            if box.center[2]>0 and abs(box.center[0])<box.center[2]: # z value (front) cannot be < 0  
                box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
    
    mkdir_if_missing('results')
    mkdir_if_missing(os.path.join('results',scene_name))

    output_path = os.path.join('results',scene_name,cat)
    mkdir_if_missing(output_path)
    cv2.imwrite(os.path.join(output_path,sample_data['filename'].split('/')[-1]),img)

############################################################################################################################################################################
# Main
############################################################################################################################################################################

def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/nuScenes', help='nuScenes data folder')
    parser.add_argument('--cat_detection_root', type=str, default='./data/cat_detection/', help='category-splitted detection folder')
    parser.add_argument('--detection_method', type=str, default='CRN', help='detection method')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='train_val/test')
    parser.add_argument('--split', type=str, default='train', help='train/val/test')
    parser.add_argument('--score_thresh', type=float, default=0.5, help='minimum detection confidence')
    parser.add_argument('--go_sep', action='store_true', default=False, help='separate detections by category (required once)')
    parser.add_argument('--gt_track', action='store_true', default=False, help='tracking using ground thruth instead of detections (debug)')
    parser.add_argument('--log_viz', action='store_true', default=False, help='Logging tracking visualization directly instead of displaying')
    parser.add_argument('--viz', action='store_true', default=False, help='display tracking visualization (superseeded by log_viz)')

    return parser



if __name__ == '__main__':

    # cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'construction_vehicle', 'motorcycle', 'trailer']
    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']
    # cat_list = ['pedestrian']

    parser = create_parser()
    args = parser.parse_args()
    
    nusc = load_nusc(args.split,args.data_root)

    if args.go_sep == True : # (pass as arg later)
        separate_det_by_cat_and_scene(args,
                                    detection_file = './data/detection_output/results_nusc.json',
                                    cat_list=cat_list,
                                    nusc=nusc
                                    )


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


'''
#TODO : 
- Metrics
'''