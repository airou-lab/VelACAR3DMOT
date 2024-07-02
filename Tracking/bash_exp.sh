#!/bin/bash

python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --no-use_vel

python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --concat --no-use_vel


python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                    --eval_set val --dataroot ./data/nuScenes 

mv output/track_output_CRN output/CRN_exp/track_output_CRN_mix_R_no_vel








# python mainfile.py --gt_track --detection_method GT --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --no-use_vel

# python mainfile.py --gt_track --detection_method GT --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --concat --no-use_vel


# python evaluate.py --result_path output/track_output_GT/track_results_nusc.json --output_dir output/track_output_GT/ \
#                     --eval_set val --dataroot ./data/nuScenes 

# mv output/track_output_GT output/GT_exp/track_output_GT_hungar_R_no_vel
