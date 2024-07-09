# #!/bin/bash

# # data separation (only done once)
# # python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --go_sep
# # python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --go_sep



# With velocity and R matrix
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --keyframes_only --use_R

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                --eval_set val --dataroot ./data/nuScenes 

# saving output
mv output/track_output_CRN output/CRN_vel_exp/kf_with_vel



# With velocity and R matrix
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split val --keyframes_only --use_R --no-use_vel


# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                --eval_set val --dataroot ./data/nuScenes 

# saving output
mv output/track_output_CRN output/CRN_vel_exp/kf_without_vel
