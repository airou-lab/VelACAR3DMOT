#!/bin/bash

# --------------------------------------------------------------------------------
#               Start Velacar with validation dataset, CRN backbone
# --------------------------------------------------------------------------------


# Backbone detection separation (only do this once)
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --go_sep 

# Velacar tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --keyframes_only --use_vel --use_R

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                    --eval_set val --dataroot ./data/nuScenes 

# saving output
mv output/track_output_CRN output/runs/run_val
