# #!/bin/bash

# # data separation (only done once)
# # python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --go_sep
# # python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --go_sep



# #-------------------------------------------------------------------------------------------------------------------------------
# # Metrics
# #-------------------------------------------------------------------------------------------------------------------------------

# # # metrics experiments (w/ most permissive tresh)
# # for metric in dist_3d dist_2d m_dis iou_2d iou_3d giou_2d giou_3d;
# # do
# #     if [[ $metric == dist_3d ]] || [[ $metric == dist_2d ]] || [[ $metric == m_dis ]]; then

# #         # CR3DMOT tracking pipeline
# #         python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ \
# #                             --run_hyper_exp --metric $metric --thresh 1

# #     elif [[ $metric == iou_2d ]] || [[ $metric == iou_3d ]]; then

# #         # CR3DMOT tracking pipeline
# #         python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ \
# #                             --run_hyper_exp --metric $metric --thresh 0.2

# #     else    # giou 2d and 3d

# #         # CR3DMOT tracking pipeline
# #         python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ \
# #                             --run_hyper_exp --metric $metric --thresh -0.8
# #     fi
    
# #     # concatenating results into formatted json
# #     python mainfile.py --concat

# #     # nuscenes official evaluation file
# #     python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
# #                     --eval_set val --dataroot ./data/nuScenes 

# #     # saving output
# #     mv output/track_output_CRN output/CRN_hyper_exp/metrics/$metric
# # done

# #-------------------------------------------------------------------------------------------------------------------------------
# # Hyperparams
# #-------------------------------------------------------------------------------------------------------------------------------

# # # threshold experiments giou
# # for thresh in -0.3 -0.4 -0.5 -0.6 -0.7 -0.8;
# # do
# #     # CR3DMOT tracking pipeline
# #     python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ \
# #                         --run_hyper_exp --thresh $thresh
    
# #     # concatenating results into formatted json
# #     python mainfile.py --concat

# #     # nuscenes official evaluation file
# #     python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
# #                     --eval_set val --dataroot ./data/nuScenes 

# #     # saving output
# #     mv output/track_output_CRN output/CRN_hyper_exp/thresh/$thresh
# # done

# # # threshold experiments dist
# # for thresh in 2 4 6 8 10;
# # do
# #     # CR3DMOT tracking pipeline
# #     python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ \
# #                         --run_hyper_exp --thresh $thresh
    
# #     # concatenating results into formatted json
# #     python mainfile.py --concat

# #     # nuscenes official evaluation file
# #     python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
# #                     --eval_set val --dataroot ./data/nuScenes 

# #     # saving output
# #     mv output/track_output_CRN output/CRN_hyper_exp/thresh/$thresh
# # done

# # for min_hit in 1 2 3;
# # do
# #     # CR3DMOT tracking pipeline
# #     python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ \
# #                         --run_hyper_exp --min_hit $min_hit
    
# #     # concatenating results into formatted json
# #     python mainfile.py --concat

# #     # nuscenes official evaluation file
# #     python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
# #                     --eval_set val --dataroot ./data/nuScenes 

# #     # saving output
# #     mv output/track_output_CRN output/CRN_hyper_exp/min_hits/$min_hit
# # done

# # for max_age in 2 4;
# # do
# #     # CR3DMOT tracking pipeline
# #     python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ \
# #                         --run_hyper_exp --max_age $max_age
    
# #     # concatenating results into formatted json
# #     python mainfile.py --concat

# #     # nuscenes official evaluation file
# #     python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
# #                     --eval_set val --dataroot ./data/nuScenes 

# #     # saving output
# #     mv output/track_output_CRN output/CRN_hyper_exp/max_age/$max_age
# # done


# #-------------------------------------------------------------------------------------------------------------------------------
# # Velocity
# #-------------------------------------------------------------------------------------------------------------------------------

# # # With velocity
# # # CR3DMOT tracking pipeline
# # python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val

# # # concatenating results into formatted json
# # python mainfile.py --concat

# # # nuscenes official evaluation file
# # python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
# #                 --eval_set val --dataroot ./data/nuScenes 

# # # saving output
# # mv output/track_output_CRN output/CRN_vel_exp_v2/with_vel



# # # Without velocity
# # # CR3DMOT tracking pipeline
# # python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --no-use_vel

# # # concatenating results into formatted json
# # python mainfile.py --concat

# # # nuscenes official evaluation file
# # python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
# #                 --eval_set val --dataroot ./data/nuScenes 

# # # saving output
# # mv output/track_output_CRN output/CRN_vel_exp_v2/without_vel


# #-------------------------------------------------------------------------------------------------------------------------------
# # Velocity keyframes
# #-------------------------------------------------------------------------------------------------------------------------------

# # # With velocity
# # # CR3DMOT tracking pipeline
# # python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --keyframes_only

# # # concatenating results into formatted json
# # python mainfile.py --concat

# # # nuscenes official evaluation file
# # python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
# #                 --eval_set val --dataroot ./data/nuScenes 

# # # saving output
# # mv output/track_output_CRN output/CRN_vel_exp_v2/with_vel_kf_only



# # # Without velocity
# # # CR3DMOT tracking pipeline
# # python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --no-use_vel --keyframes_only

# # # concatenating results into formatted json
# # python mainfile.py --concat

# # # nuscenes official evaluation file
# # python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
# #                 --eval_set val --dataroot ./data/nuScenes 

# # # saving output
# # mv output/track_output_CRN output/CRN_vel_exp_v2/without_vel_kf_only



# # With velocity and R matrix
# # CR3DMOT tracking pipeline
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --keyframes_only --use_R

# # concatenating results into formatted json
# # python mainfile.py --concat

# # nuscenes official evaluation file
# python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
#                 --eval_set val --dataroot ./data/nuScenes 

# # saving output
# mv output/track_output_CRN output/CRN_R_exp/kf_R_with_vel

# # Without velocity and R matrix
# # CR3DMOT tracking pipeline
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --no-use_vel --keyframes_only --use_R

# # concatenating results into formatted json
# # python mainfile.py --concat

# # nuscenes official evaluation file
# python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
#                 --eval_set val --dataroot ./data/nuScenes 

# # saving output
# mv output/track_output_CRN output/CRN_R_exp/kf_R_without_vel



# # With velocity and without R matrix
# # CR3DMOT tracking pipeline
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --keyframes_only

# # concatenating results into formatted json
# # python mainfile.py --concat

# # nuscenes official evaluation file
# python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
#                 --eval_set val --dataroot ./data/nuScenes 

# # saving output
# mv output/track_output_CRN output/CRN_R_exp/kf_no_R_with_vel

# # Without velocity and without R matrix
# # CR3DMOT tracking pipeline
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --no-use_vel --keyframes_only

# # concatenating results into formatted json
# # python mainfile.py --concat

# # nuscenes official evaluation file
# python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
#                 --eval_set val --dataroot ./data/nuScenes 

# # saving output
# mv output/track_output_CRN output/CRN_R_exp/kf_no_R_without_vel







# With velocity and R matrix
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --keyframes_only --use_R

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                --eval_set val --dataroot ./data/nuScenes 

# saving output
mv output/track_output_CRN output/CRN_R_exp/kf_R_using_detvel_err



# With velocity and R matrix
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --keyframes_only --use_R


# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                --eval_set val --dataroot ./data/nuScenes 

# saving output
mv output/track_output_CRN output/CRN_R_exp/kf_R_vel_err_0