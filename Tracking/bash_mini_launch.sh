#!/bin/bash

# Backbone detection separation 
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/  --split mini_val --go_sep

# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only --use_R

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 

mv output/track_output_CRN_mini/ output/exp_mini_v2/kf_R_using_detvel_err



# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only --use_R

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 

mv output/track_output_CRN_mini/ output/exp_mini_v2/kf_R_vel_err_0





#-------------------------------------------------------------------------------------------------------------------------------
# EXPs
#-------------------------------------------------------------------------------------------------------------------------------
# # det_score thresh
# for cat in truck car pedestrian bus bicycle motorcycle trailer;
# do
#     rm -r output/exp_mini_v2/CRN_score_thresh/$cat/0.*

#     # CR3DMOT tracking pipeline
#     python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only\
#                         --cat $cat --exp

# done






