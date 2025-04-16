#!/bin/bash
source scripts/echo.sh
echo ""
echo "==========================================================================================================="


# [for KITTI]
CUDA_VISIBLE_DEVICES=2,3 python monoscene/scripts/generate_output.py \
    +output_path=$MONOSCENE_OUTPUT \
    dataset=kitti \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS \
    kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW \
    n_gpus=1 batch_size=1


echo "==========================================================================================================="
echo ""
source scripts/alarm.sh




## ============================
## ==========  MEMO ===========
## ============================
# > Node7: 0, 1, 2, 3, 4, 5 available.

# [for NYU]
# python monoscene/scripts/generate_output.py \
#     + output_path=$MONOSCENE_OUTPUT \
#     dataset=NYU \
#     NYU_root=$NYU_ROOT \
#     NYU_preprocess_root=$NYU_PREPROCESS \
#     n_gpus=1 batch_size=1

# [for KITTI]
# CUDA_VISIBLE_DEVICES=2,3 python monoscene/scripts/generate_output.py \
#     +output_path=$MONOSCENE_OUTPUT \
#     dataset=kitti \
#     kitti_root=$KITTI_ROOT \
#     kitti_preprocess_root=$KITTI_PREPROCESS \
#     kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW \
#     n_gpus=1 batch_size=1
