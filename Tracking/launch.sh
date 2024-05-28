#!/bin/bash

python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.4 --go_sep

python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.4

python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.4 --concat


python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                    --eval_set val --dataroot ./data/nuScenes 
