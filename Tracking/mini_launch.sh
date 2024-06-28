#!/bin/bash

# Backbone detection separation 
# python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/  --split mini_val --go_sep

# CR3DMOT tracking pipeline
python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only

# concatenating results into formatted json
python workfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/  --split mini_val --concat

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN_mini/track_results_nusc.json --output_dir output/track_output_CRN_mini/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 

# saving output
mkdir output/exp_mini_v2
mv output/track_output_CRN_mini output/exp_mini_v2/CRN_vel_exp/
