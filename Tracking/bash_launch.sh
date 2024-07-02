#!/bin/bash

# Backbone detection separation 
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --go_sep

# # CR3DMOT tracking pipeline
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --affi_pro

# # concatenating results into formatted json
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --concat

# # nuscenes official evaluation file
# python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
#                     --eval_set val --dataroot ./data/nuScenes 

# # saving output
# mv output/track_output_CRN output/CRN_exp/affi_pro


# Backbone detection separation 
# python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --go_sep

# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/

# concatenating results into formatted json
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --concat

# nuscenes official evaluation file
python evaluate.py --result_path output/track_output_CRN/track_results_nusc.json --output_dir output/track_output_CRN/ \
                    --eval_set val --dataroot ./data/nuScenes 

# saving output
# mv output/track_output_CRN output/CRN_exp/no_fix
