#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python scripts/train_radiant_fcos3d.py --dir_data data/nuScenes --dir_result results --do_eval \
                                    --load_pretrained_fcos3d --path_checkpoint_fcos3d checkpoints/FCOS3D/fcos3d.pth \
                                    --eval_set val --path_checkpoint_dwn checkpoints/FCOS3D/dwn_radiant_focs3d_val_checkpoint.tar
