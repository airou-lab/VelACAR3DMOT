import os
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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

from PIL import Image


def get_classname():
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """

    detectionname_to_classname = {
        "pedestrian":"human.pedestrian.adult", # Blue
        "barrier":"movable_object.barrier", # Slategrey
        "traffic_cone":"movable_object.trafficcone", 
        "bicycle":"vehicle.bicycle",
        "bus":"vehicle.bus.rigid",
        "construction_vehicle":"vehicle.construction",
        "car":"vehicle.car",
        "motorcycle":"vehicle.motorcycle",
        "trailer":"vehicle.trailer",
        "truck":"vehicle.truck",
    }

    return detectionname_to_classname



def extract_json_metadata(radiant_data,sample_data_token):
    '''
    extracting all the bounding boxes predicted by the network for this token/image.
    informations are stored in a dataframe.
    '''

    df = pd.DataFrame(columns = ['token','translation','size','rotation','velocity','detection_name','detection_score'])
    i=0
    if radiant_data['results'][sample_data_token]:
        for obj in radiant_data['results'][sample_data_token] :
            df.loc[i] = [obj['sample_token'],
                            obj['translation'],
                            obj['size'],
                            obj['rotation'],
                            obj['velocity'],
                            obj['detection_name'],
                            obj['detection_score']]
                            
            i+=1
    # print (df)
    return df

def my_get_sample_data (output_metadata_df,data,nusc,verbose : bool = True):
    '''
    returns a list of the box object, instantiated with the predicted values
    '''
    bboxes=[]
    
    sd_record = nusc.get('sample_data', data['token'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])
    imsize = (sd_record['width'], sd_record['height'])

    if verbose :
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

    for i in range(output_metadata_df.shape[0]):
        if verbose :
            print('-----------------------------------------------------')
            print ("raw data :")
            print('translation: ', output_metadata_df.iloc[i]['translation'])
            print('size: ', output_metadata_df.iloc[i]['size'])
            print('rotation: ', output_metadata_df.iloc[i]['rotation'])
            print('velocity: ', output_metadata_df.iloc[i]['velocity'])
            print('detection_name: ', output_metadata_df.iloc[i]['detection_name'])
            print('detection_score: ', output_metadata_df.iloc[i]['detection_score'])
            print('-----------------------------------------------------')
            print ('\n\n')
        box = data_classes.Box(center = output_metadata_df.iloc[i]['translation'],
                                size = output_metadata_df.iloc[i]['size'],
                                orientation = Quaternion(output_metadata_df.iloc[i]['rotation']),
                                label = i,
                                score = output_metadata_df.iloc[i]['detection_score'],
                                name = output_metadata_df.iloc[i]['detection_name']
                                )
        if verbose :
            print('-----------------------------------------------------')
            print('box object:')
            print(box)
            print('-----------------------------------------------------')
            print ('\n\n')

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        if verbose :
            print('-----------------------------------------------------')
            print('transformed outputs:')
            print(box)
            print('-----------------------------------------------------')
            print ('\n\n')

        bboxes.append(box)
    return bboxes, cam_intrinsic


def my_render_sample_data(data,
                          nusc,
                          radiant_data_json : pd,
                          axes_limit: float = 40,
                          ax: Axes = None,
                          nuScenes_data_path : str = None,
                          out_path: str = None,
                          use_flat_vehicle_coordinates: bool = True,
                          verbose: bool = True,
                          ):
    
    output_metadata_df = extract_json_metadata(radiant_data_json,data['sample_token'])
    bboxes, cam_intrinsic = my_get_sample_data(output_metadata_df=output_metadata_df,data=data,nusc=nusc,verbose=verbose)
    
    NuScenes.colormap = get_colormap()
    NuScenesExp = NuScenesExplorer(NuScenes)

    sd_record = nusc.get('sample_data', data['token'])
    data_path = os.path.join(nusc.dataroot, sd_record['filename'])

    img_data = Image.open(data_path)

    # Init axes
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 16))

    # Show image
    ax.imshow(img_data)

    # Show boxes
    for box in bboxes:
        # Plotting bbox and filtering false detections
        if box.score > 0.2 and box.center[2]>0:
            cat_name = get_classname()[box.name]
            c = np.array(NuScenesExp.get_color(cat_name)) / 255.0
            # c = np.array([90,200,220]) / 255.0
            box.render(ax,view=cam_intrinsic,normalize=True, colors=(c, c, c),linewidth=1)
            if verbose :
                print('-----------------------------------------------------')
                print("box rendered: ",box)
                print('-----------------------------------------------------')


    # Limit visible range.
    ax.set_xlim(0, img_data.size[0])
    ax.set_ylim(img_data.size[1], 0)

    ax.axis('off')
    ax.set_title('{} {labels_type}'.format(sd_record['channel'], labels_type=''))
    ax.set_aspect('equal')

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

    if verbose:
        plt.show()

    plt.close('all')


    # #Loading image, CV2 method
    # print(nuScenes_data_path+data['filename'])
    # img=cv2.imread('{}/{}'.format(nuScenes_data_path,data['filename']))
    
    # for box in bboxes:
    #     c = np.array(NuScenesExp.get_color(category_name = "human.pedestrian.adult")) / 255.0
    #     box.render_cv2(img,view=cam_intrinsic,normalize=True, colors=(c, c, c))

    # cv2.imshow('test',img)
    # if cv2.waitKey(0)&0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     exit()
    # else :
    #     cv2.destroyAllWindows()
    

def process(args):
    nuScenes_data_path = './data/nuscenes'

    # Directory to save plotted images
    save_dir = args.saveDir
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Load NuScenes dataset
    if args.split == 'val' or args.split == 'train':
        nusc = NuScenes(version='v1.0-trainval', dataroot=nuScenes_data_path, verbose=True)
    elif args.split == 'test':
        nusc = NuScenes(version='v1.0-test', dataroot=nuScenes_data_path, verbose=True)

    with open(args.resAnn, 'r') as file:
        detections_data = json.load(file)

    my_scenes = nusc.scene

    # sensors = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']
    sensors = ['CAM_FRONT']
    for my_scene in my_scenes:
        for sensor in sensors:
            # Extracting first scene sample
            my_sample = nusc.get('sample', my_scene['first_sample_token'])
            data = nusc.get('sample_data', my_sample['data'][sensor])
            while True:

                if data['filename'].split('/')[0] == 'samples':
                    # Creating output path
                    out_path='{}/{}'.format(save_dir,data['filename'].split(".")[0])
                    parsed_out_path = '/'.join(out_path.split('/')[0:len(out_path.split('/'))-1])

                    
                    # Adding output path if it doesn't exist
                    if not os.path.exists(parsed_out_path):
                        print("creating output directory :",parsed_out_path)
                        os.makedirs(parsed_out_path)
                        print("Done")

                    my_render_sample_data(data = data, 
                                            nusc=nusc, 
                                            nuScenes_data_path = nuScenes_data_path, 
                                            out_path='{}/{}'.format(save_dir,data['filename'].split(".")[0]),
                                            radiant_data_json=detections_data,
                                            verbose = args.verbose)


                    print("Rendered and saved as '{}/{}'".format(save_dir,data['filename']))
                else :
                    print("skipped: '{}/{}'".format(save_dir,data['filename']))

                    
                #GOTO next sensor
                if data['next'] == "":
                    print("no next data")
                    break
                else:
                    first_token = data['next']
                    data = nusc.get('sample_data', first_token)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='test', help='val/test')
    parser.add_argument('--resAnn', type=str, default='./data/nuscenes/fusion_data/train_result/radiant_pgd/eval_nus_infos_val/fusion/img_bbox/results_nusc.json', help='result file path')
    parser.add_argument('--saveDir', type=str, default='./outputs', help='save directory')
    parser.add_argument('--verbose','-v', action='store_true', help='verbose')
    args = parser.parse_args()
    process(args)

if __name__ == '__main__':
    main()
