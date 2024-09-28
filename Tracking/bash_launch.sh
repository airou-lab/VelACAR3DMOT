#!/bin/bash

# --------------------------------------------------------------------------------
#               Start Velacar with mini validation dataset, Radiant backbone
# --------------------------------------------------------------------------------


# Backbone detection separation (only do this once)
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --go_sep

# Velacar tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --keyframes_only --use_vel --use_R \
                    --detection_method Radiant

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_Radiant_mini/track_results_nusc.json --output_dir output/track_output_Radiant_mini/ \
                    --eval_set val --dataroot ./data/nuScenes 

# saving output
mv output/track_output_Radiant/ output/runs/run_val






