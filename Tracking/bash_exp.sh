#!/bin/bash

# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --no-use_vel

# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --concat --no-use_vel


# python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
#                     --eval_set val --dataroot ./data/nuScenes 

# mv output/track_output_CRN output/CRN_exp/track_output_CRN_mix_R_no_vel


python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --detection_method CRN --use_R --go_sep
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --detection_method CRN --use_R --go_sep

#---------------------------------------------------------------------------------
# Using sweeps, with R
#---------------------------------------------------------------------------------

# Velocity ON
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --detection_method CRN --use_R

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                --eval_set val --dataroot ./data/nuScenes 
# saving output
mv output/track_output_CRN output/Paper_results/val/sweeps_R_vel_exps/sweeps_R_with_vel


# Velocity OFF
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --detection_method CRN --use_R --no-use_vel

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                --eval_set val --dataroot ./data/nuScenes 
# saving output
mv output/track_output_CRN output/Paper_results/val/sweeps_R_vel_exps/sweeps_R_without_vel


#---------------------------------------------------------------------------------
# Using sweeps, without R
#---------------------------------------------------------------------------------

# Velocity ON
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --detection_method CRN

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                --eval_set val --dataroot ./data/nuScenes 
# saving output
mv output/track_output_CRN output/Paper_results/val/sweeps_R_vel_exps/sweeps_without_R_with_vel


# Velocity OFF
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --detection_method CRN --no-use_vel

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                --eval_set val --dataroot ./data/nuScenes 
# saving output
mv output/track_output_CRN output/Paper_results/val/sweeps_R_vel_exps/sweeps_without_R_without_vel


#---------------------------------------------------------------------------------
# Using sweeps, with R, for mini dataset
#---------------------------------------------------------------------------------

# Velocity ON
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --detection_method CRN --use_R

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 
# saving output
mv output/track_output_CRN_mini/ output/Paper_results/mini/sweeps_R_vel_exps/sweeps_R_with_vel


# Velocity OFF
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --detection_method CRN --use_R --no-use_vel

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 
# saving output
mv output/track_output_CRN_mini/ output/Paper_results/mini/sweeps_R_vel_exps/sweeps_R_without_vel



#---------------------------------------------------------------------------------
# Using sweeps, without R, for mini dataset
#---------------------------------------------------------------------------------

# Velocity ON
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --detection_method CRN

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 
# saving output
mv output/track_output_CRN_mini/ output/Paper_results/mini/sweeps_R_vel_exps/kf_without_R_with_vel


# Velocity OFF
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --detection_method CRN --no-use_vel

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 
# saving output
mv output/track_output_CRN_mini/ output/Paper_results/mini/sweeps_R_vel_exps/kf_without_R_without_vel