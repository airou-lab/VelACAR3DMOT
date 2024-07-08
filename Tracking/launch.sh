#!/bin/bash

python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --detection_method Radiant --score_thresh 0.1 --go_sep

python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --detection_method Radiant --score_thresh 0.1

python workfile.py --detection_method Radiant --score_thresh 0.1 --concat


python evaluate.py --result_path output/track_output_Radiant/track_results_nusc.json --output_dir output/track_output_Radiant/ \
                    --eval_set val --dataroot ./data/nuScenes 
