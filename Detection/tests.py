import os
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.nuscenes import NuScenesExplorer
import json
import argparse
import cv2
import pandas as pd
from pyquaternion import Quaternion

from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils import data_classes

from lib.my_utils import points_cam2img



def process(args):
    nuScenes_data_path = './data/nuScenes'

    # Directory to save plotted images
    save_dir = args.saveDir
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Load NuScenes dataset
    if args.split == 'val' or args.split == 'train':
        nusc = NuScenes(version='v1.0-trainval', dataroot=nuScenes_data_path, verbose=True)
    elif args.split == 'test':
        nusc = NuScenes(version='v1.0-test', dataroot=nuScenes_data_path, verbose=True)

    # with open(args.resAnn, 'r') as file:
    #     detections_data = json.load(file)

    my_scenes = nusc.scene

    sensors = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']
    for my_scene in my_scenes:
        for sensor in sensors:
            my_sample = nusc.get('sample', my_scene['first_sample_token'])
            print ('my_sample :',my_sample)
            print ('my_sample[anns] :',my_sample['anns'])

            my_annotation_token = my_sample['anns'][0]
            my_annotation_metadata =  nusc.get('sample_annotation', my_annotation_token)
            print()
            print()
            print()
            print(my_annotation_metadata)

            data = nusc.get('sample_data', my_sample['data'][sensor])
            while True:

                out_path='{}/{}'.format(save_dir,data['filename'].split(".")[0])
                parsed_out_path = '/'.join(out_path.split('/')[0:len(out_path.split('/'))-1])

                
                # Adding output path if it doesn't exist
                if not os.path.exists(parsed_out_path):
                    print("creating output directory :",parsed_out_path)
                    os.makedirs(parsed_out_path)
                    print("Done")

                print('-----------------------------------------------------')
                print ("raw data :")
                print('translation: ', my_annotation_metadata['translation'])
                print('size: ', my_annotation_metadata['size'])
                print('rotation: ', my_annotation_metadata['rotation'])
                print ()
                print ()
                print ()
                sd_record = nusc.get('sample_data', data['token'])
                cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                print('sd_record: ',sd_record)
                print('cs_record: ',cs_record)
                print('sensor_record: ',sensor_record)
                print('pose_record: ',pose_record)
                print ()
                print ()
                print ()
                cam_intrinsic = np.array(cs_record['camera_intrinsic'])
                imsize = (sd_record['width'], sd_record['height'])

                box = data_classes.Box(my_annotation_metadata['translation'],my_annotation_metadata['size'],Quaternion(my_annotation_metadata['rotation'])) 
                print('box object:')
                print(box)
                print ()
                print ()
                print ()

                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

                print('transformed outputs:')
                print(box)
                print ()
                print ()
                print ()


                box_vis_level: BoxVisibility = BoxVisibility.ANY
                data_path, boxes, camera_intrinsic = nusc.get_sample_data(data['token'],box_vis_level=box_vis_level)
                print('get_sample_data output')
                print('/')
                print('/')
                print('/')
                print('/')
                print('data_path: ',data_path)
                print('/')
                print('/')
                print('boxes: ',boxes[0])
                print('/')
                print('/')
                print('camera_intrinsic: ',camera_intrinsic)
                print('/')
                print('/')
                print('/')
                print('/')

                data_path = './data/crash_test/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg'
                img = cv2.imread(data_path)

                # my_render_cv2(box,img)
                NuScenes.colormap = get_colormap()
                NuScenesExp = NuScenesExplorer(NuScenes)
                c = np.array(NuScenesExp.get_color(category_name = "human.pedestrian.adult")) / 255.0
                box.render_cv2(img,view=cam_intrinsic,normalize=True, colors=(c, c, c))

                cv2.imshow('test',img)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                exit()

                nusc.render_sample_data(data['token'], out_path='{}/{}'.format(save_dir,data['filename'].split(".")[0]),verbose=False)
                print("Rendered and saved as '{}/{}'".format(save_dir,data['filename']))

                # closing plot
                plt.close('all')

                print(data)
                
                # Loading generated image
                img=cv2.imread('{}/{}'.format(save_dir,data['filename'].split(".")[0]+'.png'))

                cv2.imshow('test',img)


                cv2.waitKey(0)
                cv2.destroyAllWindows()
                exit()

                    
                #GOTO next sensor
                if data['next'] == "":
                    print("no next data")
                    break
                else:
                    first_token = data['next']
                    data = nusc.get('sample_data', first_token)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='val/test')
    # parser.add_argument('--resAnn', type=str, default='./results/results_nusc.json', help='result file path')
    parser.add_argument('--saveDir', type=str, default='./outputs', help='save directory')
    parser.add_argument('--verbose', type=str, default='True', help='verbose')
    args = parser.parse_args()
    process(args)

if __name__ == '__main__':
    main()
