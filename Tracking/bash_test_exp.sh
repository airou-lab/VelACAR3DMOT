#!/bin/bash

python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split test --go_sep



# With velocity
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split test --keyframes_only --use_R

# saving output
mv output/track_output_CRN output/CRN_test_exp/kf_with_vel




# With velocity
# CR3DMOT tracking pipeline
python mainfile.py --data_root ./data/nuScenes --cat_detection_root ./data/cat_detection/ --split test --keyframes_only --use_R --no-use_vel

# saving output
mv output/track_output_CRN output/CRN_test_exp/kf_without_vel

