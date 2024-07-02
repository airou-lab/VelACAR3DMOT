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
from typing import Tuple, List, Dict, Any
from shutil import copyfile
from pyquaternion import Quaternion
import argparse

import colorsys
import cv2

# load nuScenes libraries
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box, RadarPointCloud
from nuscenes.utils.splits import create_splits_logs, create_splits_scenes
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt,
    load_prediction,
)
from nuscenes.eval.tracking.algo import TrackingEvaluation
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP, LEGACY_METRICS
from nuscenes.eval.tracking.data_classes import TrackingMetrics, TrackingMetricDataList, TrackingConfig, TrackingBox, \
    TrackingMetricData
from nuscenes.eval.tracking.loaders import create_tracks
from nuscenes.eval.tracking.render import recall_metric_curve, summary_plot
from nuscenes.eval.tracking.utils import print_final_metrics


# load AN3DMOT model
from libs.utils import *
from libs.config import *
import mainfile
import evaluate


def exp_jobs_handler (args):
    print('starting exps')

    # cat_list = ['car', 'pedestrian', 'truck', 'bus', 'bicycle', 'motorcycle', 'trailer']
    cat_list = [args.cat]

    nusc = load_nusc(args.split,args.data_root)

    orig_score = get_score_thresh(args,args.cat)
    score = round(orig_score-0.1,3) # Avoid decimal errors

    best_amota = 0
    best_thresh = 0
    best_amotp = 2
    best_thresh_amotp = 0

    while score <=(orig_score+0.1):
        args.score_thresh = score
        print('thresh:',args.score_thresh)
        
        # CR3DMOT pipeline
        mainfile.tracking(args,
                            cat_list=cat_list,
                            nusc=nusc
                            )

        mainfile.concat_results(args,
                                data_dir='./results/logs/'+args.detection_method+'_mini' if 'mini' in args.data_root  else './results/logs/'+args.detection_method,
                                cat_list=cat_list,
                                nusc=nusc
                                )



        # nusc eval pipeline
        result_path_ = os.path.expanduser('output/track_output_CRN_mini/track_results_nusc.json')
        output_dir_ = os.path.expanduser('output/track_output_CRN_mini/')
        eval_set_ = 'val'
        dataroot_ = './data_mini/nuScenes'
        version_ = 'v1.0-trainval'
        config_path = ''
        render_curves_ = bool(0)
        verbose_ = bool(1)
        render_classes_ = ''

        
        cfg_ = config_factory('tracking_nips_2019')


        nusc_eval = evaluate.TrackingEval(config=cfg_, result_path=result_path_, eval_set=eval_set_, output_dir=output_dir_,
                             nusc_version=version_, nusc_dataroot=dataroot_, verbose=verbose_,
                             render_classes=render_classes_)


        summary = nusc_eval.main(render_curves=render_curves_)





        amota = summary['label_metrics']['amota'][args.cat]
        amotp = summary['label_metrics']['amotp'][args.cat]

        if amota > best_amota:
            best_amota = amota
            best_thresh_amota = score

        if amotp < best_amotp:
            best_amotp = amotp
            best_thresh_amotp = score


        # Saving output
        os.rename("./output/track_output_CRN_mini", "./output/exp_mini_v2/CRN_score_thresh/%s/%s"%(args.cat,str(score)))
        score = round(score+0.001,3) # Avoid decimal errors


    with open ('./output/exp_mini_v2/CRN_score_thresh/exp_output_'+args.cat+'.txt','w') as f:
        f.write('best AMOTA :\n')
        f.write(str(best_amota))
        f.write('\n')
        f.write('\n')
        f.write('det score threshold:\n')
        f.write(str(best_thresh_amota))
        f.write('\n')
        f.write('-----------------------------------------------------------------------------------------------------------')
        f.write('\n')
        f.write('best AMOTP :\n')
        f.write(str(best_amotp))
        f.write('\n')
        f.write('\n')
        f.write('det score threshold:\n')
        f.write(str(best_thresh_amotp))
        f.write('\n')


