#!/bin/bash
sourch scripts/echo.sh
echo ""
echo "==========================================================================================================="


# [for KITTI]
CUDA_VISIBLE_DEVICES=1 python monoscene/scripts/train_monoscene.py \
    dataset=kitti \
    enable_log=true \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS\
    kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
    kitti_logdir=$KITTI_LOG \
    n_gpus=1 batch_size=1 \
    exp_prefix="Frame_test" \
    low_resolution=false \
    sequence_length=1 \
    use_event=false \


echo "==========================================================================================================="
echo ""
source scripts/alarm.sh



## ============================
## ==========  MEMO ===========
## ============================
# > Node7: 0, 1, 2, 3, 4, 5 available.

# [for NYU]
# CUDA_VISIBLE_DEVICES=3,4 python monoscene/scripts/train_monoscene.py \
#     dataset=NYU \
#     NYU_root=$NYU_ROOT \
#     NYU_preprocess_root=$NYU_PREPROCESS \
#     logdir=$NYU_LOG \
#     n_gpus=2 \
#     batch_size=2

# [for KITTI]
# TODO: saving other directory
# CUDA_VISIBLE_DEVICES=5 python monoscene/scripts/train_monoscene.py,1,2,3,4,5,6,7 \
# CUDA_VISIBLE_DEVICES=1 python monoscene/scripts/train_monoscene.py \
#     dataset=kitti \
#     enable_log=true \
#     kitti_root=$KITTI_ROOT \
#     kitti_preprocess_root=$KITTI_PREPROCESS\
#     kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
#     kitti_logdir=$KITTI_LOG \
#     n_gpus=1 batch_size=1 \
#     low_resolution=false \
#     sequence_length=1 \