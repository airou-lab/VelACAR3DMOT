#!/bin/bash

# Backbone detection separation 
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/  --split val --go_sep




# #-------------------------------------------------------------------------------------------------------------------------------
# # Metrics
# #-------------------------------------------------------------------------------------------------------------------------------

# # metrics experiments (w/ most permissive tresh)
# for metric in dist_3d dist_2d m_dis iou_2d iou_3d giou_2d giou_3d;
# do
#     if [[ $metric == dist_3d ]] || [[ $metric == dist_2d ]]; then

#         # CR3DMOT tracking pipeline
#         python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --keyframes_only\
#                             --detection_method Radiant --split val\
#                             --run_hyper_exp --metric $metric --thresh 1

#     elif [[ $metric == m_dis ]]; then
#         # CR3DMOT tracking pipeline
#         python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --keyframes_only --no-use_vel\
#                             --detection_method Radiant --split val\
#                             --run_hyper_exp --metric $metric --thresh 1

#     elif [[ $metric == iou_2d ]] || [[ $metric == iou_3d ]]; then

#         # CR3DMOT tracking pipeline
#         python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --keyframes_only\
#                             --detection_method Radiant --split val\
#                             --run_hyper_exp --metric $metric --thresh 0.2

#     else    # giou 2d and 3d

#         # CR3DMOT tracking pipeline
#         python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --keyframes_only\
#                             --detection_method Radiant --split val\
#                             --run_hyper_exp --metric $metric --thresh -0.8
#     fi

#     # nuscenes official evaluation file
#     python evaluate.py --result_path output/track_output_Radiant/track_results_nusc.json --output_dir output/track_output_Radiant/ \
#                     --eval_set val --dataroot ./data/nuScenes 

#     # saving output
#     mv output/track_output_Radiant output/Radiant_hyper_exp/metrics/$metric
# done






# #-------------------------------------------------------------------------------------------------------------------------------
# # EXPs
# #-------------------------------------------------------------------------------------------------------------------------------
# # det_score thresh
# for cat in truck car pedestrian bus bicycle motorcycle trailer;
# do
#     # CR3DMOT tracking pipeline
#     python mainfile.py --data_root ./data_/nuScenes --cat_detection_root ./data_/cat_detection/ --split _val --keyframes_only\
#                        --detection_method Radiant 
#                        --cat $cat --exp
# done








