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

def get_sample_info(nusc,sensor,token,verbose=False):
    scenes = nusc.scene
    print(scenes)
    for scene in scenes:

        first_sample = nusc.get('sample', scene['first_sample_token']) # sample 0
        sample_data = nusc.get('sample_data', first_sample['data'][sensor])   # data for sample 0

        #Looping scene samples
        while(sample_data['next'] != ""):       
            # if sample_token corresponds to token
            if sample_data['sample_token']==token:

                if verbose :
                    print('\nscene: ',scene)
                    print('\nsample: ',first_sample)
                    print ('\nsample_data: ',sample_data)
                return scene['name'], sample_data['filename']

            else:
                # going to next sampl
                sample_data = nusc.get('sample_data', sample_data['next'])

        if verbose:
            print ('token NOT in:',scene['name'])
    return 0

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
        while(sample_data['next'] != ""):       
            # if sample_token corresponds to token
            if sample_data['sample_token']==token:
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

def get_gt_at_t(cat,t,sample,sample_data,cs_record,ego_pose,det_df):

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
            box.velocity = gt_vel

            # print(box)

            # Extracting all values 
            t=t
            x,y,z = box.center
            w,l,h = box.wlh
            r1,r2,r3,r4 = box.orientation
            score = 1
            token = box.token
            vx,vy,vz = box.velocity
            vz=0
            
            # vz = 0                                

            
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


############################################################################################################################################################################
# Pipeline
############################################################################################################################################################################

def separate_det_by_cat_and_samples(output_root,detection_file,detection_method,sensor,cat_list,nusc,score_thresh):

    mkdir_if_missing(output_root)

    # load detection file
    print('opening results file at %s' % (detection_file))
    with open(detection_file) as json_file:
        '''
        Splits detection output into their categories.
        For each category, we have 
        '''
        data = json.load(json_file)
        num_frames = len(data['results'])

        for cat in cat_list:
            count = 0
            scene_num_mem = []

            print ('category: ',cat)
            cat_folder = detection_method+'_'+cat
            mkdir_if_missing(os.path.join(output_root,cat_folder))

            for sample_token, dets in data['results'].items():
                scene_num, image_path = get_sample_info(nusc,sensor,sample_token)

                if scene_num not in scene_num_mem:
                    count=0
                    f = open(os.path.join(output_root,cat_folder,scene_num+'.txt'),'w')     # rewritting file

                else :
                    f = open(os.path.join(output_root,cat_folder,scene_num+'.txt'),'a')     # appending to existing scene file

                det_cnt = 0
                for det_sample in dets:

                    if det_sample['detection_name']==cat and det_sample['detection_score']>=score_thresh:
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
                        

                count+=1        # incrementing token counter
                scene_num_mem.append(scene_num)

                print('\nfound %d detections of category:'%(det_cnt),cat,',in scene:',scene_num,'token:',sample_token)
                print('results logged at: ',f.name)
                print('corresponding image:',image_path)
                print('count = ',count-1)

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

def tracking (cat_list,data_root,cat_detection_root,detection_method,sensor,score_thresh,nusc,log_viz):
    for cat in cat_list:        

        '''
        TODO : if detection of this category : check mahalanobis (or IoU) distance with Kalman pred
        if association found : add to Tm
        if no association found : Tn = Birth of new track

        after a certain time :
        if no association found for a trajectory => Tu = Death 

        Add the speed as I do that => Basically object speed can go in kalman filter (vx,vy) for better prediction.
        => get direction and better approximation of where the oject should beat t=t+1
        => better tracking
        '''


        print("category: ",cat)
        det_file_list=get_scenes_list(os.path.join(cat_detection_root,detection_method+'_'+cat))

        for det_file in det_file_list:

            tracker, scene, first_token = initialize_tracker(data_root=os.path.join(cat_detection_root,detection_method+'_'+cat),cat=cat, ID_start=0, nusc=nusc, det_file=det_file)
            
            print ('initial trackers :',tracker.trackers)
            print ('scene :',scene['name'])

            print(scene)
            print()
            
            print('first sample :')

            sample_token = first_token

            for sample_number in range(scene['nbr_samples']):   # or while sample_tokem != ''

                print ('t = ',sample_number)
                # input()

                sample = nusc.get('sample', sample_token) # sample 0
                sample_data = nusc.get('sample_data', sample['data'][sensor])   # data for sample 0

                cs_record, ego_pose, cam_intrinsic = get_sample_metadata(nusc,sensor,sample_token,verbose=False)

                det_df = get_det_df_at_t(cat_detection_root,detection_method,cat,det_file,sample_number)

                print(200*'-','\n')
                print('Sample: ',sample) 
                print('\nSample data:',sample_data)
                print('\n',200*'-','\n')

                print('\n',200*'-','\n')
                print('Ego pose:',ego_pose)
                print('\nCalibrated Sensor record:',cs_record)
                print('\nCam intrinsic:',cam_intrinsic)
                print('\n',200*'-','\n')

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

                # pkl_path = './data/nuScenes/nuscenes_infos_val.pkl'
                # pkl_meta = pd.read_pickle(pkl_path)

                # with open(pkl_path, 'rb') as file:  
                #     try:
                #         while True:
                #             meta = pickle.load(file)
                #     except EOFError:
                #         pass

                # for sample in meta:
                #     print(sample)
                #     print(100*'-')
                #     print(type(sample))
                #     print(sample.keys())
                #     exit()
                #     ann_info = sample['ann_infos']
                #     ann_info = sample['cam_infos']
                #     print(ann_info)
                #     print(type(ann_info))
                #     print()
                #     exit()

                # exit()
                # detection_visualization(nusc=nusc,
                #                         data_root=data_root,
                #                         sample_data=sample_data,
                #                         det_df=det_df,
                #                         ego_pose=ego_pose,
                #                         token = sample_token,
                #                         cat=cat)

                print('resume tracking :')
                # input()

                results, affi = tracker.track(det_df, sample_number, scene['name'])

                print('\n',200*'-','\n')
                print ('tracking results:',results)
                print ('affinity matrix:',affi)
                print('\n',200*'-','\n')                


                tracking_visualization(nusc=nusc,
                                        data_root=data_root,
                                        sample_data=sample_data,
                                        results=results[0],
                                        cs_record=cs_record,
                                        cam_intrinsic=cam_intrinsic,
                                        ego_pose=ego_pose,
                                        det_df=det_df,
                                        score_thresh=score_thresh,
                                        t=t
                                        )   #det_df and score_thresh are for debugging purposes

                sample_token = sample['next']   # last sample should be: ''                



                # if sample_number==4:
                #     exit()

def gt_tracking (cat_list,data_root,cat_detection_root,detection_method,sensor,score_thresh,nusc,log_viz):
    for cat in cat_list:        

        print("category: ",cat)
        det_file_list=get_scenes_list(os.path.join(cat_detection_root,detection_method+'_'+cat))

        for det_file in det_file_list:
            # if det_file != 'scene-0103.txt':    # TO REMOVE / 4DEBUG
            #     continue

            tracker, scene, first_token = initialize_tracker(data_root=os.path.join(cat_detection_root,detection_method+'_'+cat),cat=cat, 
                                                            ID_start=0, nusc=nusc, det_file=det_file)

            # print ('initial trackers :',tracker.trackers)
            print ('scene :',scene['name'])

            # print(scene)
            # print()
            
            # print('first sample :')

            sample_token = first_token

            for t in range(scene['nbr_samples']):   # or while sample_tokem != ''

                print ('t = ',t)

                sample = nusc.get('sample', sample_token) # sample 0
                sample_data = nusc.get('sample_data', sample['data'][sensor])   # data for sample 0

                cs_record, ego_pose, cam_intrinsic = get_sample_metadata(nusc,sensor,sample_token,verbose=False)

                det_df = get_det_df_at_t(cat_detection_root,detection_method,cat,det_file,t)
                gt_df = get_gt_at_t(cat,t,sample,sample_data,cs_record,ego_pose,det_df)


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

                if log_viz:
                    log_tracking_visualization(nusc=nusc,
                                                data_root=data_root,
                                                sample_data=sample_data,
                                                results=results[0],
                                                cs_record=cs_record,
                                                cam_intrinsic=cam_intrinsic,
                                                ego_pose=ego_pose,
                                                cat=cat,
                                                scene_name=scene['name'],
                                                t=t
                                                )
                else :
                    tracking_visualization(nusc=nusc,
                                            data_root=data_root,
                                            sample_data=sample_data,
                                            results=results[0],
                                            cs_record=cs_record,
                                            cam_intrinsic=cam_intrinsic,
                                            ego_pose=ego_pose,
                                            det_df=gt_df,
                                            score_thresh=score_thresh,
                                            t=t
                                            )

                sample_token = sample['next']   # last sample should be: ''                


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

        if t_res == t:  # Filtering out tracklets with no detection for that frame
            if box.center[2]>0: # z value (front) cannot be < 0  
                box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
        # print(box)
        # input()

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
            if box.center[2]>0: # z value (front) cannot be < 0  
                box.render_cv2(im=img,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
    
    mkdir_if_missing('results')
    mkdir_if_missing(os.path.join('results',scene_name))

    output_path = os.path.join('results',scene_name,cat)
    mkdir_if_missing(output_path)
    cv2.imwrite(os.path.join(output_path,sample_data['filename'].split('/')[-1]),img)


if __name__ == '__main__':

    cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'construction_vehicle', 'motorcycle', 'trailer']
    data_root = './data/nuScenes'
    cat_detection_root = './data/cat_detection/'
    detection_method = 'CRN'  # pass as argument
    sensor = 'CAM_FRONT'    # pass as argument
    split = 'train'         # pass as argument
    score_thresh = 0.5
    go_sep = False      # pass as argument
    gt_track = True     # pass as argument
    log_viz = True
    
    nusc = load_nusc(split,data_root)

    if go_sep == True : # (pass as arg later)
        separate_det_by_cat_and_samples(output_root=cat_detection_root,
                                        detection_file = './data/detection_output/results_nusc.json',
                                        detection_method = detection_method,
                                        sensor = sensor,
                                        cat_list=cat_list,
                                        nusc=nusc,
                                        score_thresh=score_thresh
                                        )


    if gt_track == False :
        # Tracking using CNN detections
        tracking(cat_list=cat_list,
                data_root=data_root,
                cat_detection_root=cat_detection_root,
                detection_method=detection_method,
                sensor=sensor,
                score_thresh=score_thresh,
                nusc=nusc,
                log_viz=log_viz
                )
    else:
        # Tracking using Ground truth detections
        gt_tracking(cat_list=cat_list,
                    data_root=data_root,
                    cat_detection_root=cat_detection_root,
                    detection_method=detection_method,
                    sensor=sensor,
                    score_thresh=score_thresh,
                    nusc=nusc,
                    log_viz=log_viz
                    )




'''
#TODO : 
- Metrics
'''