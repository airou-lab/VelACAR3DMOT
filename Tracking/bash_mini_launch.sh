#!/bin/bash

# Backbone detection separation 
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/  --split mini_val --go_sep

# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only --use_R \
                    --detection_method Radiant --score_thresh 0.1

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_Radiant_mini/track_results_nusc.json --output_dir output/track_output_Radiant_mini/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 

mv output/track_output_Radiant_mini/ output/exp_mini/kf_R_vel_before_thresh_study






