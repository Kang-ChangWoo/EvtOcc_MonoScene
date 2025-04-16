#!/bin/bash

TARGET_NODE=$1

source scripts/echo.sh
echo ""
echo "==========================================================================================================="

if [ "$TARGET_NODE" == "default" ]; then
    # [for KITTI]
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS\
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
        kitti_logdir=$KITTI_LOG \
        n_gpus=4 batch_size=4 \
        exp_prefix="Reproduction_after0321_nodeTileTest_b5effitest_nLow_nSeq_" \
        low_resolution=false \
        sequence_length=1 \
        use_event=true \

    echo "==========================================================================================================="
    echo ""
    source scripts/alarm.sh


elif [ "$TARGET_NODE" == "n2" ]; then
    # [for KITTI]
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS\
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
        kitti_logdir=$KITTI_LOG \
        n_gpus=4 batch_size=4 \
        exp_prefix="EXP0324_A_SLICINGevt._n2" \
        low_resolution=false \
        sequence_length=1 \
        use_event=true \

    echo "==========================================================================================================="
    echo ""
    source scripts/alarm.sh

# elif [ "$TARGET_NODE" == "n4" ]; then
#     # [for KITTI]
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python monoscene/scripts/train_monoscene.py \
#         dataset=kitti \
#         enable_log=true \
#         kitti_root=$KITTI_ROOT \
#         kitti_evt_root=$KITTI_EVT_ROOT \
#         kitti_preprocess_root=$KITTI_PREPROCESS\
#         kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
#         kitti_logdir=$KITTI_LOG \
#         n_gpus=4 batch_size=4 \
#         exp_prefix="EXP0324_E_Pretrained2DUNET" \
#         low_resolution=false \
#         sequence_length=1 \
#         use_event=false \

#     echo "==========================================================================================================="
#     echo ""
#     source scripts/alarm.sh

elif [ "$TARGET_NODE" == "n4" ]; then
    # [for KITTI]
    CUDA_VISIBLE_DEVICES=0,1 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS\
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
        kitti_logdir=$KITTI_LOG \
        n_gpus=2 batch_size=2 \
        exp_prefix="EXP0414_evtTransformTkn" \
        low_resolution=false \
        sequence_length=1 \
        use_event=true \
        use_bulk=true \
        use_token=true \
        context_prior=false \
        relation_loss=false \
        CE_ssc_loss=false \
        sem_scal_loss=false \
        geo_scal_loss=false \
        
    echo "==========================================================================================================="
    echo ""
    source scripts/alarm.sh

# elif [ "$TARGET_NODE" == "n4" ]; then
#     # [for KITTI]
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python monoscene/scripts/train_monoscene.py \
#         dataset=kitti \
#         enable_log=true \
#         kitti_root=$KITTI_ROOT \
#         kitti_evt_root=$KITTI_EVT_ROOT \
#         kitti_preprocess_root=$KITTI_PREPROCESS\
#         kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
#         kitti_logdir=$KITTI_LOG \
#         n_gpus=2 batch_size=2 \
#         exp_prefix="EXP0414_test_singleResolution" \
#         low_resolution=false \
#         sequence_length=1 \
#         use_event=false \
#         use_bulk=false \
#         project_1_2=false \
#         project_1_4=false \
#         project_1_8=false \

#     echo "==========================================================================================================="
#     echo ""
#     source scripts/alarm.sh

elif [ "$TARGET_NODE" == "n6" ]; then
    # [for KITTI]
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS\
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
        kitti_logdir=$KITTI_LOG \
        n_gpus=4 batch_size=4 \
        exp_prefix="EXP0324_A_SLICINGevt." \
        low_resolution=false \
        sequence_length=1 \
        use_event=true \

    echo "==========================================================================================================="
    echo ""
    source scripts/alarm.sh


# elif [ "$TARGET_NODE" == "n7" ]; then
#     # [for KITTI]
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python monoscene/scripts/train_monoscene.py \
#         dataset=kitti \
#         enable_log=true \
#         kitti_root=$KITTI_ROOT \
#         kitti_evt_root=$KITTI_EVT_ROOT \
#         kitti_preprocess_root=$KITTI_PREPROCESS\
#         kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
#         kitti_logdir=$KITTI_LOG \
#         n_gpus=1 batch_size=1 \
#         exp_prefix="EXP0324_C_singleGB" \
#         low_resolution=false \
#         sequence_length=1 \
#         use_event=false \

#     echo "==========================================================================================================="
#     echo ""
#     source scripts/alarm.sh

# elif [ "$TARGET_NODE" == "n7" ]; then
#     # [for KITTI]
#     CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python monoscene/scripts/train_monoscene.py \
#         dataset=kitti \
#         enable_log=true \
#         kitti_root=$KITTI_ROOT \
#         kitti_evt_root=$KITTI_EVT_ROOT \
#         kitti_preprocess_root=$KITTI_PREPROCESS\
#         kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
#         kitti_logdir=$KITTI_LOG \
#         n_gpus=2 batch_size=2 \
#         exp_prefix="EXP0324_B_dualGB" \
#         low_resolution=false \
#         sequence_length=1 \
#         use_event=false \

#     echo "==========================================================================================================="
#     echo ""
#     source scripts/alarm.sh

elif [ "$TARGET_NODE" == "n7" ]; then
    # [for KITTI]
    CUDA_VISIBLE_DEVICES=3,4,5,6,7 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS\
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
        kitti_logdir=$KITTI_LOG \
        n_gpus=1 batch_size=1 \
        exp_prefix="EXP0324_D_accumulate" \
        low_resolution=false \
        sequence_length=1 \
        use_event=false \

    echo "==========================================================================================================="
    echo ""
    source scripts/alarm.sh


else
    echo "Invalid node argument. Usage: ./run.sh {node or workstation}"
    exit 1
fi



# source scripts/echo.sh
# echo ""
# echo "==========================================================================================================="

# # [for KITTI]
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python monoscene/scripts/train_monoscene.py \
#     dataset=kitti \
#     enable_log=true \
#     kitti_root=$KITTI_ROOT \
#     kitti_evt_root=$KITTI_EVT_ROOT \
#     kitti_preprocess_root=$KITTI_PREPROCESS\
#     kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
#     kitti_logdir=$KITTI_LOG \
#     n_gpus=4 batch_size=4 \
#     exp_prefix="Reproduction_after0321_nodeTileTest_b5effitest_nLow_nSeq_" \
#     low_resolution=false \
#     sequence_length=1 \
#     use_event=true \

# echo "==========================================================================================================="
# echo ""
# source scripts/alarm.sh



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