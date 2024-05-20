#!/bin/bash

python workfile.py --go_sep \
                    --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.58

python workfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.58

python workfile.py --concat \
                    --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --score_thresh 0.58

python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                    --eval_set val --dataroot ./data/nuScenes 
