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
    CUDA_VISIBLE_DEVICES=3,4,5,6,7 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS\
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
        kitti_logdir=$KITTI_LOG \
        n_gpus=2 batch_size=2 \
        exp_prefix="EXP0430_evtTransform_MiDaS_Nadap60CLAMP_DepthEval" \
        input_mode="b" \
        model_type="VisionTransformer" \
        context_prior=false \
        relation_loss=false \
        CE_ssc_loss=false \
        sem_scal_loss=false \
        geo_scal_loss=false \
        depth_validation=true \

    echo "==========================================================================================================="
    echo ""
    source scripts/alarm.sh

 #EXP0514_evtTransform_Unet_DepthEval UNet VisionTransformer
elif [ "$TARGET_NODE" == "n3" ]; then
    # [for KITTI]
    CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS \
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW \
        kitti_logdir=$KITTI_LOG \
        n_gpus=2 batch_size=2 \
        exp_prefix="Testing_depth_anything" \
        input_mode="b" \
        model_type="DepthAnything" \
        context_prior=false \
        relation_loss=false \
        CE_ssc_loss=false \
        sem_scal_loss=false \
        geo_scal_loss=false \
        depth_validation=true \
        
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
    CUDA_VISIBLE_DEVICES=6,7 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS\
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
        kitti_logdir=$KITTI_LOG \
        n_gpus=2 batch_size=2 \
        exp_prefix="FROM0519_ViT_bigModel_preT" \
        input_mode="b" \
        model_type="VisionTransformer" \
        context_prior=false \
        relation_loss=false \
        CE_ssc_loss=false \
        sem_scal_loss=false \
        geo_scal_loss=false \
        depth_validation=true \
        
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

# For Unet Depth from Event dataset
elif [ "$TARGET_NODE" == "n6" ]; then
    # [for KITTI]
    CUDA_VISIBLE_DEVICES=2,3 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS\
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
        kitti_logdir=$KITTI_LOG \
        n_gpus=2 batch_size=2 \
        exp_prefix="FROM0519_ViT_fromRGB" \
        input_mode="a" \
        model_type="VisionTransformer" \
        context_prior=false \
        relation_loss=false \
        CE_ssc_loss=false \
        sem_scal_loss=false \
        geo_scal_loss=false \
        depth_validation=true \
        
    echo "==========================================================================================================="
    echo ""
    source scripts/alarm.sh

# For Unet Depth from Event dataset
elif [ "$TARGET_NODE" == "n7" ]; then
    CUDA_VISIBLE_DEVICES=0,1 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS \
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW \
        kitti_logdir=$KITTI_LOG \
        n_gpus=2 batch_size=2 \
        exp_prefix="FROM0519_depth_anything" \
        input_mode="b" \
        model_type="DepthAnything" \
        context_prior=false \
        relation_loss=false \
        CE_ssc_loss=false \
        sem_scal_loss=false \
        geo_scal_loss=false \
        depth_validation=true \

    echo "==========================================================================================================="
    echo ""
    source scripts/alarm.sh

elif [ "$TARGET_NODE" == "n14" ] || [ "$TARGET_NODE" == "n14" ] || [ "$TARGET_NODE" == "n15" ]; then
    # [for KITTI]
    CUDA_VISIBLE_DEVICES=4,5 python monoscene/scripts/train_monoscene.py \
        dataset=kitti \
        enable_log=true \
        kitti_root=$KITTI_ROOT \
        kitti_evt_root=$KITTI_EVT_ROOT \
        kitti_preprocess_root=$KITTI_PREPROCESS \
        kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW \
        kitti_logdir=$KITTI_LOG \
        n_gpus=2 batch_size=2 \
        exp_prefix="FROM0519_VisionTransformer" \
        input_mode="b" \
        model_type="VisionTransformer" \
        context_prior=false \
        relation_loss=false \
        CE_ssc_loss=false \
        sem_scal_loss=false \
        geo_scal_loss=false \
        depth_validation=true \

    echo "==========================================================================================================="
    echo ""
    source scripts/alarm.sh


else
    echo "Invalid node argument. Usage: ./run.sh {node or workstation}"

fi