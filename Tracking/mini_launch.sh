#!/bin/bash

# Backbone detection separation 
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/  --split mini_val --go_sep

# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only --use_R

# concatenating results into formatted json
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/  --split mini_val --concat

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 

mv output/track_output_CRN_mini/ output/exp_mini_v2/global_test_R_kf_thresh_vel


# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only --use_R --no-use_vel

# concatenating results into formatted json
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/  --split mini_val --concat

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 

mv output/track_output_CRN_mini/ output/exp_mini_v2/global_test_R_kf_thresh_no_vel










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
































































































































# # keyframes with speed
# # CR3DMOT tracking pipeline
# python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only

# # concatenating results into formatted json
# python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/  --split mini_val --concat

# # nuscenes official evaluation file
# python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
#                     --eval_set val --dataroot ./data_mini/nuScenes 

# mv output/track_output_CRN_mini output/exp_mini_v2/CRN_vel_exp/kf_with_vel



# # keyframes without speed
# # CR3DMOT tracking pipeline
# python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only --no-use_vel

# # concatenating results into formatted json
# python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/  --split mini_val --concat

# # nuscenes official evaluation file
# python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
#                     --eval_set val --dataroot ./data_mini/nuScenes 

# mv output/track_output_CRN_mini output/exp_mini_v2/CRN_vel_exp/kf_without_vel




# # classic with speed
# # CR3DMOT tracking pipeline
# python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val

# # concatenating results into formatted json
# python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/  --split mini_val --concat

# # nuscenes official evaluation file
# python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
#                     --eval_set val --dataroot ./data_mini/nuScenes 

# mv output/track_output_CRN_mini output/exp_mini_v2/CRN_vel_exp/with_vel



# # classic without speed
# # CR3DMOT tracking pipeline
# python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --no-use_vel

# # concatenating results into formatted json
# python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/  --split mini_val --concat

# # nuscenes official evaluation file
# python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
#                     --eval_set val --dataroot ./data_mini/nuScenes 

# mv output/track_output_CRN_mini output/exp_mini_v2/CRN_vel_exp/without_vel



# # keyframes with custom R matrix
# # CR3DMOT tracking pipeline
# python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only --use_R

# # concatenating results into formatted json
# python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/  --split mini_val --concat

# # nuscenes official evaluation file
# python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
#                     --eval_set val --dataroot ./data_mini/nuScenes 

# mv output/track_output_CRN_mini output/exp_mini_v2/CRN_R_exp/kf_with_R_velacc_at_1