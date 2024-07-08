# nuScenes dev-kit.
# Code written by Holger Caesar, 2019.
# Licensed under the Creative Commons [see licence.txt]

"""
This script converts nuScenes data to KITTI format and KITTI results to nuScenes.
It is used for compatibility with software that uses KITTI-style annotations.

We do not encourage this, as:
- KITTI has only front-facing cameras, whereas nuScenes has a 360 degree horizontal fov.
- KITTI has no radar data.
- The nuScenes database format is more modular.
- KITTI fields like occluded and truncated cannot be exactly reproduced from nuScenes data.
- KITTI has different categories.

Limitations:
- We don't specify the KITTI imu_to_velo_kitti projection in this code base.
- We map nuScenes categories to nuScenes detection categories, rather than KITTI categories.
- Attributes are not part of KITTI and therefore set to '' in the nuScenes result format.
- Velocities are not part of KITTI and therefore set to 0 in the nuScenes result format.
- This script uses the `train` and `val` splits of nuScenes, whereas standard KITTI has `training` and `testing` splits.

This script includes three main functions:
- nuscenes_gt_to_kitti(): Converts nuScenes GT annotations to KITTI format.
- render_kitti(): Render the annotations of the (generated or real) KITTI dataset.
- kitti_res_to_nuscenes(): Converts a KITTI detection result to the nuScenes detection results format.

To launch these scripts run:
- python export_kitti.py nuscenes_gt_to_kitti_obj --nusc_kitti_root ~/nusc_kitti
- python export_kitti.py render_kitti --nusc_kitti_root ~/nusc_kitti --render_2d False
- python export_kitti.py kitti_res_to_nuscenes --nusc_kitti_root ~/nusc_kitti
Note: The parameter --render_2d specifies whether to draw 2d or 3d boxes.

To work with the original KITTI dataset, use these parameters:
 --nusc_kitti_root /data/sets/kitti --split training

See https://www.nuscenes.org/object-detection for more information on the nuScenes result format.
"""

import os, sys, json, numpy as np, fire
from typing import List, Dict, Any
from shutil import copyfile
from pyquaternion import Quaternion

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes

from AB3DMOT_libs.nuScenes2KITTI_helper import load_correspondence, load_correspondence_inverse
from AB3DMOT_libs.nuScenes2KITTI_helper import kitti_cam2nuScenes_lidar, nuScenes_transform2KITTI
from AB3DMOT_libs.nuScenes2KITTI_helper import create_KITTI_transform, convert_anno_to_KITTI, save_image, save_lidar
from AB3DMOT_libs.nuScenes_utils import nuScenes_lidar2world, nuScenes_world2lidar, get_sensor_param, split_to_samples, scene_to_samples
from AB3DMOT_libs.nuScenes_utils import box_to_trk_sample_result, create_nuScenes_box, box_to_det_sample_result

# load KITTI libraries
from AB3DMOT_libs.kitti_calib import Calibration, save_calib_file
from AB3DMOT_libs.kitti_trk import Tracklet_3D

class KittiConverter:
    def __init__(self,
                 nusc_kitti_root: str = './data/nuKITTI',   
                 data_root: str = './data/nuScenes/',
                 result_root: str = './results/nuScenes/',
                 result_name: str = 'megvii_val_H1',         
                 cam_name: str = 'CAM_FRONT',
                 split: str = 'val'):
        """
        :param nusc_kitti_root: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export. Note that only one camera is allowed in KITTI.
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """
        self.nusc_kitti_root = nusc_kitti_root; 
        if not os.path.isdir(nusc_kitti_root):
            os.mkdir(nusc_kitti_root)

        self.cam_name = cam_name
        self.split = split
        if split in ['train', 'val', 'trainval']: self.nusc_version = 'v1.0-trainval'
        elif split == 'test':                     self.nusc_version = 'v1.0-test'
        self.result_name = result_name
        self.data_root = data_root
        self.result_root = result_root

        # Select subset of the data to look at.
        self.nusc = NuScenes(version=self.nusc_version, dataroot=data_root, verbose=True)

    def nuscenes_obj_result2kitti(self):
        # convert the detection results in NuScenes format to KITTI object format
        # for example, we will need this when using nuScenes detection results for tracking in the KITTI format

        # load correspondences
        corr_file = os.path.join(self.nusc_kitti_root, 'object', 'produced', 'correspondence', self.split+'.txt')
        corr_dict = load_correspondence(corr_file)

        # path
        save_dir = os.path.join(self.nusc_kitti_root, 'object', 'produced', 'results', self.split, self.result_name, 'data'); 
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)


        # load results
        result_file = os.path.join(self.data_root, 'produced', 'results', 'detection', self.result_name, 'results_%s.json' % self.split)
        print('opening results file at %s' % (result_file))
        with open(result_file) as json_file:
            data = json.load(json_file)
            num_frames = len(data['results'])
            count = 0
            for sample_token, dets in data['results'].items():
                
                # get sensor and transformation
                pose_record, cs_record_lid, cs_record_cam = get_sensor_param(self.nusc, sample_token, cam_name=self.cam_name)
                velo_to_cam_trans, velo_to_cam_rot, r0_rect, p_left_kitti = \
                    nuScenes_transform2KITTI(cs_record_lid, cs_record_cam)

                # loop through every detection
                frame_index = corr_dict[sample_token]
                save_file = os.path.join(save_dir, frame_index+'.txt'); save_file = open(save_file, 'w')
                sys.stdout.write('processing results for %s.txt: %d/%d\r' % (frame_index, count, num_frames))
                sys.stdout.flush()
                for result_tmp in dets:
                    token_tmp = result_tmp['sample_token']
                    assert token_tmp == sample_token, 'token is different'
                    
                    # create nuScenes box in world coordinate
                    xyz = result_tmp['translation']                 # center_x, center_y, center_z
                    wlh = result_tmp['size']                        # width, length, height
                    rotation = result_tmp['rotation']               # quaternion in the global frame: w, x, y, z
                    name = result_tmp['detection_name'].capitalize()
                    box = Box(xyz, wlh, Quaternion(rotation), name=name, token=sample_token)        # box in global frame

                    # convert to nuScenes lidar coordinate
                    box = nuScenes_world2lidar(box, cs_record_lid, pose_record)  

                    # Convert from nuScenes lidar to KITTI camera format.
                    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                        box, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect)

                    # Project 3d box to 2d box in image, ignore box if it does not fall inside.
                    bbox_2d = KittiDB.project_kitti_box_to_image(box_cam_kitti, p_left_kitti, imsize=(1600, 900))
                    if bbox_2d is None: bbox_2d = (-1, -1, -1, -1)          # add all anno to the converted version

                    truncated = 0.0
                    occluded = 0
                    box_cam_kitti.score = result_tmp['detection_score']

                    # KITTI format: type, trunc, occ, alpha, 2D bbox (x1, y1, x2, y2), 
                    # 3D bbox (h, w, l, x, y, z, ry), score
                    result_str = KittiDB.box_to_string(name=name, box=box_cam_kitti, bbox_2d=bbox_2d, truncation=truncated, occlusion=occluded)
                    save_file.write(result_str + '\n')

                save_file.close()
                count += 1

        print('Results saved to: %s' % save_dir)

if __name__ == '__main__':
    fire.Fire(KittiConverter)