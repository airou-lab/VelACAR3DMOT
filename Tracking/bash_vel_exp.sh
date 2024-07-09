#!/bin/bash

# data separation (only done once)
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --go_sep
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --go_sep



# With velocity
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --keyframes_only --use_R --detection_method Radiant

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_Radiant/track_results_nusc.json --output_dir output/track_output_Radiant/ \
                --eval_set val --dataroot ./data/nuScenes 

# saving output
mv output/track_output_Radiant output/Radiant_vel_exp/kf_R_with_vel





# Without velocity
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --no-use_vel --keyframes_only --use_R --detection_method Radiant

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_Radiant/track_results_nusc.json --output_dir output/track_output_Radiant/ \
                --eval_set val --dataroot ./data/nuScenes 

# saving output
mv output/track_output_Radiant output/Radiant_vel_exp/kf_R_without_vel