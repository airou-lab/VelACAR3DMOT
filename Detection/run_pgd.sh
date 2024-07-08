#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python scripts/train_radiant_pgd.py --dir_data ./data/nuScenes --do_eval \
                                    --load_pretrained_pgd --path_checkpoint_pgd checkpoints/PGD/PGD_pretrained.pth \
                                    --eval_set test --path_checkpoint_dwn checkpoints/PGD/dwn_radiant_pgd_val.tar
