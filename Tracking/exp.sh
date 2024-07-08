#!/bin/bash

python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --detection_method Radiant --score_thresh 0.1 --use_R

python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --detection_method Radiant --score_thresh 0.1 --concat --use_R


python evaluate.py --result_path output/track_output_Radiant/track_results_nusc.json --output_dir output/track_output_Radiant/ \
                    --eval_set val --dataroot ./data/nuScenes 

mv output/track_output_Radiant output/Radiant_exp/track_output_Radiant_hungar_R



python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --detection_method Radiant --score_thresh 0.1 --no-use_vel

python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --detection_method Radiant --score_thresh 0.1 --concat --no-use_vel


python evaluate.py --result_path output/track_output_Radiant/track_results_nusc.json --output_dir output/track_output_Radiant/ \
                    --eval_set val --dataroot ./data/nuScenes 

mv output/track_output_Radiant output/Radiant_exp/track_output_Radiant_hungar_no_R_no_vel





python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --detection_method Radiant --score_thresh 0.1 --no-use_vel --use_R

python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --detection_method Radiant --score_thresh 0.1 --concat --no-use_vel --use_R


python evaluate.py --result_path output/track_output_Radiant/track_results_nusc.json --output_dir output/track_output_Radiant/ \
                    --eval_set val --dataroot ./data/nuScenes 

mv output/track_output_Radiant output/Radiant_exp/track_output_Radiant_hungar_R_no_vel







# python workfile.py --gt_track --detection_method GT --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --no-use_vel

# python workfile.py --gt_track --detection_method GT --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --concat --no-use_vel


# python evaluate.py --result_path output/track_output_GT/track_results_nusc.json --output_dir output/track_output_GT/ \
#                     --eval_set val --dataroot ./data/nuScenes 

# mv output/track_output_GT output/GT_exp/track_output_GT_hungar_R_no_vel
