#!/bin/bash

# data separation (only done once)
# python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.4 --go_sep



#-------------------------------------------------------------------------------------------------------------------------------
# Metrics
#-------------------------------------------------------------------------------------------------------------------------------

# # metrics experiments
# for metric in dist_3d dist_2d m_dis iou_2d iou_3d giou_2d giou_3d;
# do
#     if [[ $metric == dist_3d ]] || [[ $metric == dist_2d ]] || [[ $metric == m_dis ]]; then

#         # CR3DMOT tracking pipeline
#         python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.4\
#                             --run_hyper_exp --metric $metric --thresh 6

#     elif [[ $metric == iou_2d ]] || [[ $metric == iou_3d ]]; then

#         # CR3DMOT tracking pipeline
#         python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.4\
#                             --run_hyper_exp --metric $metric --thresh 0.5

#     else

#         # CR3DMOT tracking pipeline
#         python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.4\
#                             --run_hyper_exp --metric $metric --thresh -0.5
#     fi
    
#     # concatenating results into formatted json
#     python workfile.py --concat

#     # nuscenes official evaluation file
#     python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
#                     --eval_set val --dataroot ./data/nuScenes 

#     # saving output
#     mv output/track_output_CRN output/CRN_hyper_exp/metrics/$metric
# done

#-------------------------------------------------------------------------------------------------------------------------------
# Hyperparams
#-------------------------------------------------------------------------------------------------------------------------------

# threshold experiments
# for thresh in -0.4 -0.3 -0.6 -0.7 -0.8;
# do
#     # CR3DMOT tracking pipeline
#     python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.4\
#                         --run_hyper_exp --thresh $thresh
    
#     # concatenating results into formatted json
#     python workfile.py --concat

#     # nuscenes official evaluation file
#     python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
#                     --eval_set val --dataroot ./data/nuScenes 

#     # saving output
#     mv output/track_output_CRN output/CRN_hyper_exp/thresh/$thresh
# done

# for min_hit in 1 2 3;
# do
#     # CR3DMOT tracking pipeline
#     python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.4\
#                         --run_hyper_exp --min_hit $min_hit
    
#     # concatenating results into formatted json
#     python workfile.py --concat

#     # nuscenes official evaluation file
#     python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
#                     --eval_set val --dataroot ./data/nuScenes 

#     # saving output
#     mv output/track_output_CRN output/CRN_hyper_exp/min_hits/$min_hit
# done

# for max_age in 2 4;
# do
#     # CR3DMOT tracking pipeline
#     python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.4\
#                         --run_hyper_exp --max_age $max_age
    
#     # concatenating results into formatted json
#     python workfile.py --concat

#     # nuscenes official evaluation file
#     python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
#                     --eval_set val --dataroot ./data/nuScenes 

#     # saving output
#     mv output/track_output_CRN output/CRN_hyper_exp/max_age/$max_age
# done


#-------------------------------------------------------------------------------------------------------------------------------
# Velocity
#-------------------------------------------------------------------------------------------------------------------------------

# With velocity
# CR3DMOT tracking pipeline
python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --score_thresh 0.4 

# concatenating results into formatted json
python workfile.py --concat

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                --eval_set val --dataroot ./data/nuScenes 

# saving output
mv output/track_output_CRN output/CRN_vel_exp/with_vel



# Without velocity
# CR3DMOT tracking pipeline
python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --score_thresh 0.4 --no-use_vel

# concatenating results into formatted json
python workfile.py --concat

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                --eval_set val --dataroot ./data/nuScenes 

# saving output
mv output/track_output_CRN output/CRN_vel_exp/no_vel
