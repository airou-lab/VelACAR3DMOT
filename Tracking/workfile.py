import os, sys, json, numpy as np, fire
from typing import List, Dict, Any
from shutil import copyfile
from pyquaternion import Quaternion

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes

############################################################################################################################################################################
# Utils
############################################################################################################################################################################


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

def load_nusc(split):
    assert split in ['train','val','test'], "Bad nuScenes version"

    if split in ['train','val']:
        nusc_version = 'v1.0-trainval'
    elif split =='test':
        nusc_version = 'v1.0-test'
    
    nusc = NuScenes(version=nusc_version, dataroot=data_root, verbose=True)

    return nusc

def get_sample_info(nusc,sensor,token,verbose=False):
    scenes = nusc.scene

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


############################################################################################################################################################################
# Pipeline
############################################################################################################################################################################


def separate_det_by_cat_and_samples(output_root,detection_file,detection_method,sensor,cat_list,nusc,det_thresh):

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

                    if det_sample['detection_name']==cat and det_sample['detection_score']>=det_thresh:
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
                        f.write('\n')
                        
                        # f.write('%s \n'%(str(count)))
                        # for key, value in det_sample.items():  
                        #     if key in ['translation','size','rotation','velocity']:
                        #         f.write('%s:%s\n' %(key, value))
                        

                count+=1        # incrementing token counter
                scene_num_mem.append(scene_num)

                print('\nfound %d detections of category:'%(det_cnt),cat,',in scene:',scene_num,'token:',sample_token)
                print('results logged at: ',f.name)
                print('corresponding image:',image_path)
                print('count = ',count-1)


def data_association(cat_list):

    for cat in cat_list:
        '''
        TODO : if detection of this category : check mahalanobis distance with Kalman pred (??) => in any cane use kalman pred to associate
        if association found : add to Tm
        if no association found : Tn = Birth

        after a certain time :
        if no association found for a trajectory => Tu = Death 

        Add the speed as I do that> Basically angular speed can go in kalman filter (??) to get direction and better approximation of where the oject should be
        at t=t+1 
        in any case speed in used for prediction of futur position
        '''
        


if __name__ == '__main__':

    cat_list = ['pedestrian', 'car', 'truck', 'bus', 'bicycle', 'construction_vehicle', 'motorcycle', 'trailer']
    data_root = './data/nuScenes'
    split = 'train'
    det_thresh = 0.5

    nusc = load_nusc(split)

    separate_det_by_cat_and_samples(output_root='./data/tracking_output/',
                                    detection_file = './detection_results/results_nusc.json',
                                    detection_method = 'CRN',
                                    sensor = 'CAM_FRONT',
                                    cat_list=cat_list,
                                    nusc=nusc,
                                    det_thresh=det_thresh
                                    )


    data_association(cat_list=cat_list)
