#!/bin/bash

# Backbone detection separation 
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/  --split mini_val --go_sep




# #-------------------------------------------------------------------------------------------------------------------------------
# # Metrics
# #-------------------------------------------------------------------------------------------------------------------------------

# # metrics experiments (w/ most permissive thresh)
# for metric in dist_3d dist_2d m_dis iou_2d iou_3d giou_2d giou_3d;
# do
#     if [[ $metric == dist_3d ]] || [[ $metric == dist_2d ]]; then

#         # CR3DMOT tracking pipeline
#         python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --keyframes_only\
#                             --detection_method Radiant --split mini_val\
#                             --run_hyper_exp --metric $metric --thresh 1

#     elif [[ $metric == m_dis ]]; then
#         # CR3DMOT tracking pipeline
#         python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --keyframes_only --no-use_vel\
#                             --detection_method Radiant --split mini_val\
#                             --run_hyper_exp --metric $metric --thresh 1


#     elif [[ $metric == iou_2d ]] || [[ $metric == iou_3d ]]; then

#         # CR3DMOT tracking pipeline
#         python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --keyframes_only\
#                             --detection_method Radiant --split mini_val\
#                             --run_hyper_exp --metric $metric --thresh 0.2

#     else    # giou 2d and 3d

#         # CR3DMOT tracking pipeline
#         python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --keyframes_only\
#                             --detection_method Radiant --split mini_val\
#                             --run_hyper_exp --metric $metric --thresh -0.8
#     fi

#     # nuscenes official evaluation file
#     python evaluate.py --result_path output/track_output_Radiant_mini/track_results_nusc.json --output_dir output/track_output_Radiant_mini/ \
#                     --eval_set val --dataroot ./data_mini/nuScenes 

#     # saving output
#     mv output/track_output_Radiant_mini output/Radiant_hyper_exp_mini/metrics/$metric
# done

#-------------------------------------------------------------------------------------------------------------------------------
# tracking thresh
#-------------------------------------------------------------------------------------------------------------------------------

# for track_thresh in 0 0.1 0.2 0.3 0.4 0.5;
# do

#     # CR3DMOT tracking pipeline
#     python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --keyframes_only\
#                         --detection_method Radiant --split mini_val\
#                         --thresh $track_thresh --cat car

#     # nuscenes official evaluation file
#     python evaluate.py --result_path output/track_output_Radiant_mini/track_results_nusc.json --output_dir output/track_output_Radiant_mini/ \
#                     --eval_set val --dataroot ./data_mini/nuScenes 

#     # saving output
#     mv output/track_output_Radiant_mini output/Radiant_hyper_exp_mini/track_thresh/$track_thresh
# done

# mkdir output/Radiant_hyper_exp_mini/track_thresh/iou
# mv output/Radiant_hyper_exp_mini/track_thresh/0* output/Radiant_hyper_exp_mini/track_thresh/iou

# for track_thresh in -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3;
# do

#     # CR3DMOT tracking pipeline
#     python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --keyframes_only\
#                         --detection_method Radiant --split mini_val\
#                         --thresh $track_thresh

#     # nuscenes official evaluation file
#     python evaluate.py --result_path output/track_output_Radiant_mini/track_results_nusc.json --output_dir output/track_output_Radiant_mini/ \
#                     --eval_set val --dataroot ./data_mini/nuScenes 

#     # saving output
#     mv output/track_output_Radiant_mini output/Radiant_hyper_exp_mini/track_thresh/$track_thresh
# done


#-------------------------------------------------------------------------------------------------------------------------------
# EXPs
#-------------------------------------------------------------------------------------------------------------------------------
# det_score thresh
for cat in truck car pedestrian bus bicycle motorcycle;# trailer;
do
    mkdir ./output/exp_mini/Radiant_score_thresh/$cat/pass1
    mv ./output/exp_mini/Radiant_score_thresh/$cat/0* ./output/exp_mini/Radiant_score_thresh/$cat/pass1

    # CR3DMOT tracking pipeline
    python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only\
                       --detection_method Radiant \
                       --cat $cat --exp
done








