#!/bin/bash

# --------------------------------------------------------------------------------
#               Start Velacar with mini validation dataset
# --------------------------------------------------------------------------------


# Backbone detection separation (only do this once)
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --go_sep 

# Velacar Tracking pipeline
python mainfile.py --data_root ./data_mini/nuScenes --cat_detection_root ./data_mini/cat_detection/ --split mini_val --keyframes_only --use_vel --use_R

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                    --eval_set val --dataroot ./data_mini/nuScenes 

# saving output
mv output/track_output_CRN_mini output/runs/run_mini