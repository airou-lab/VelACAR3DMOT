#!/bin/bash

python workfile.py --gt_track --detection_method GT \
                    --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/

python workfile.py --concat --gt_track --detection_method GT \
                    --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/

python evaluate.py --result_path output/track_output_GT/track_results_nusc.json --output_dir output/track_output_GT/ \
                    --eval_set val --dataroot ./data/nuScenes 
