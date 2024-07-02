#-----------------------------------------------
# Author : Mathis Morales                       
# Email  : mathis-morales@outlook.fr             
# git    : https://github.com/MathisMM            
#-----------------------------------------------

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

import cv2

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box, RadarPointCloud
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.tracking.data_classes import TrackingConfig

# load AN3DMOT model
from libs.model import AB3DMOT
from libs.utils import *
from libs.config import *

from exp import *

############################################################################################################################################################################
# Pipeline
############################################################################################################################################################################

def separate_det_by_cat_and_scene(args,cat_list,nusc):

    output_root = args.cat_detection_root
    mkdir_if_missing(output_root)

    detection_file = args.detection_root+'results_nusc.json'

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
                            
                            if det_sample['detection_name']==cat:

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


                    if nusc_data['next'] == "":
                        #GOTO next scene
                        print("no next data in scene %s"%(scene['name']))
                        break
                    else:
                        #GOTO next sample
                        next_token = nusc_data['next']
                        nusc_data = nusc.get('sample_data', next_token)


    print(100*'#','\nfinished separating detections\n',100*('#'))

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

        print('category list:',cat_list)
        print('current category:',cat)

        if args.gt_track:
            det_file_list = ['scene-0103.txt','scene-0553.txt','scene-0796.txt','scene-0916.txt']
        else:
            det_file_list=get_scenes_list(os.path.join(args.cat_detection_root,args.detection_method+'_'+cat))

        progress_cnt = 0
        progress_tot = len(det_file_list)


        for det_file in det_file_list:

            # Computation time for this file
            comp_time_start = time.time()
            
            if args.debug and det_file != 'scene-0103.txt':    # DEBUG
                continue

            if args.gt_track:
                # dictionary used to get every gt at a frame
                next_sample_tokens_dict = dict()


            # initializing output dataframe
            results_df = pd.DataFrame(columns =['h','w','l','x','y','z','theta','vx','vy','ID','r1','r2','r3','r4','score','token','t'])    # full results for this scene

            # initializing AB3DDMOT
            tracker, scene, first_token = initialize_tracker(args=args, cat=cat, ID_start=1, nusc=nusc, det_file=det_file)

            if args.verbose>=3:
                print ('initial trackers :',tracker.trackers)

            progress_cnt+=1
            print ('scene :',scene['name'])
            print ('Scene %d/%d'%(progress_cnt,progress_tot))


            if args.verbose>=2:
                print (scene)

            # first token init
            t=0
            sample_token = first_token
            sample = nusc.get('sample', first_token) # sample 0
            sample_data = nusc.get('sample_data', sample['data'][args.sensor])   # data for sample 0

            det_df_scene = get_det_df(args,cat,det_file)
            if args.verbose>=2:
                pd.set_option('display.max_rows', 500)
                pd.set_option('display.min_rows',100)
                print('Full detection dataframe for this scene:')
                print(det_df_scene)
                pd.reset_option('all')

            while(True):

                if args.verbose >= 1: print ('t =',t)

                if args.keyframes_only == True and sample_data['is_key_frame'] == False:
                    if sample_data['next'] == "":
                        #GOTO next scene
                        break
                    else:
                        #GOTO next sample
                        sample_token = sample_data['next']
                        sample_data = nusc.get('sample_data', sample_token)
                        t+=1
                        continue    # skip sweeps

                if args.verbose>=3:
                    print(200*'*','\n')
                    print('Sample: ',sample) 
                    print('\nSample data:',sample_data)
                    print('\n',200*'-','\n')

                metadata_token = sample_data['token']
                cs_record, ego_pose, cam_intrinsic = get_sample_metadata(nusc,args.sensor,metadata_token,verbose=False)

                if args.gt_track:
                    det_df, next_sample_tokens_dict = get_gt_at_t(args,nusc,cat,t,sample,next_sample_tokens_dict)
                else:
                    det_df = get_det_df_at_t(det_df_scene,t)

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
                                            score_thresh=args.score_thresh if args.score_thresh>0 else get_score_thresh(args,cat),
                                            t=t
                                            )

                if sample_data['next'] == "":
                    #GOTO next scene
                    if args.verbose >= 1 : print("no next data")
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
            log_track_results_at_t(args,results_df,cat,scene['name'],args.detection_method)

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


############################################################################################################################################################################
# Main
############################################################################################################################################################################

# sensor_list = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

def create_parser():

    parser = argparse.ArgumentParser()
    
    # nuScenes loading
    parser.add_argument('--data_root', type=str, default='./data/nuScenes', help='nuScenes data folder')
    parser.add_argument('--split', type=str, default='train', help='train/val/test')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT', help='Sensor type (see sensor_list)')
    parser.add_argument('--keyframes_only', action='store_true', default=False, help='Only use keyframes (no sweeps, 2Hz instead of 12)') 

    # Detection config
    parser.add_argument('--detection_method', type=str, default='CRN', help='Detection method')
    parser.add_argument('--score_thresh', type=float, default=-1, help='Force a global minimum detection score')
    parser.add_argument('--cat_detection_root', type=str, default='./data/cat_detection/', help='Category-split detection folder')
    parser.add_argument('--detection_root', type=str, default='./data/detection_output/', help='Softlink to detection output folder')

    # Tracking config
    parser.add_argument('--use_vel', action='store_true', dest='use_vel', default=True, help='Use radar velocity for prediction')
    parser.add_argument('--no-use_vel', action='store_false', dest='use_vel', default=False, help='Turn off radar velocity for prediction, \
                                                                                                    velocity will be estimated by Kalman filter')

    parser.add_argument('--affi_pro', action='store_true', dest='affi_pro', default=False, help='Use post-processing affinity in model.py')

    # Actions
    parser.add_argument('--go_sep', action='store_true', default=False, help='Separate detections by category (required once)')
    parser.add_argument('--gt_track', action='store_true', default=False, help='Tracking using ground thruth instead of detections (debug)')
    parser.add_argument('--log_viz', action='store_true', default=False, help='Logging tracking visualization (saving .png files) directly instead of displaying')
    parser.add_argument('--viz', action='store_true', default=False, help='Display tracking visualization (superseeded by log_viz)')
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
    parser.add_argument('--cat', type=str, default=None, help='define a single category to compute') 

    parser.add_argument('--exp',action='store_true', default=False, help='Run experiment from exp.py')


    return parser

def check_args(args):
    sensor_list = ['CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT',
                    'LIDAR_TOP',
                    'RADAR_BACK_LEFT','RADAR_BACK_RIGHT','RADAR_FRONT','RADAR_FRONT_LEFT','RADAR_FRONT_RIGHT']  

    if 'mini' in args.split:
        args.split = (args.split).split('mini_')[1]
        
        if 'mini' not in args.data_root :
            args.data_root= (args.data_root).split('data')[0]+'data_mini'+(args.data_root).split('data')[1]

        if 'mini' not in args.cat_detection_root :
            args.cat_detection_root= (args.cat_detection_root).split('data')[0]+'data_mini'+(args.cat_detection_root).split('data')[1]

        if 'mini' not in args.detection_root :
            args.detection_root= (args.detection_root).split('data')[0]+'data_mini'+(args.detection_root).split('data')[1]

    assert args.split in ['train','val','test'], 'Wrong split type'
    assert args.sensor in sensor_list, 'Unknown sensor selected'    
    assert args.metric in ['dist_3d', 'dist_2d', 'm_dis', 'iou_2d', 'iou_3d', 'giou_2d', 'giou_3d'], 'wrong metric type'
    assert args.min_hits == None or args.min_hits > 0, 'min_hits must be postive integer' 
    assert args.max_age == None or args.max_age > 0, 'max_age must be postive integer' 
    assert os.path.exists(args.data_root), 'Data folder at %s not found'%(args.data_root)
    assert os.path.exists(args.detection_root), 'Detection folder at %s not found'%(args.detection_root)
    assert os.path.exists(args.detection_root+'/results_nusc.json'), 'Missing json detection file at %s'%(args.detection_root)

    print(args)


if __name__ == '__main__':

    # Argument parser
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)


    if args.cat != None:            # mostly for debug and exps
        cat_list = [args.cat]

    else :
        # cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'construction_vehicle', 'motorcycle', 'trailer'] # detection list of vehicles 
        cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']    # AB3DMOT-supported tracking

    # Launch experiment loop from exp.py
    if args.exp:
        exp_jobs_handler(args)
        exit()



    # --------------------------------CR3DMOT-------------------------------- #
    # Loading scenes
    nusc = load_nusc(args.split,args.data_root)

    # Separation of all detections by their categories and scenes (required first step)
    if args.go_sep:
        separate_det_by_cat_and_scene(args,
                                    cat_list=cat_list,
                                    nusc=nusc
                                    )
        exit(0)

    # Concatenating all tracking files into one json formatted for nuScenes evaluation
    if args.concat:
        concat_results(args,
                        data_dir='./results/logs/'+args.detection_method+'_mini' if 'mini' in args.data_root  else './results/logs/'+args.detection_method,
                        cat_list=cat_list,
                        nusc=nusc
                        )
        exit(2)

    # Tracking pipeline
    log_args(args)

    tracking(args,
            cat_list=cat_list,
            nusc=nusc
            )

    exit(1)
