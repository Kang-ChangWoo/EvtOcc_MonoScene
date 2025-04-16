#!/bin/bash

TARGET_NODE=$1

if [ "$TARGET_NODE" == "default" ]; then
    echo ""
    echo "==========================================================================================================="

    export NYU_PREPROCESS='/root/data/NYU_dataset/preprocess' # not used.
    export NYU_ROOT='/root/data/NYU_dataset/depthbin' # not used.
    export NYU_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/nyu_log' # not used.

    export KITTI_PREPROCESS='/root/dev/data/dataset/SemanticKITTI/preprocess_cw'
    export KITTI_PREPROCESS_LOW='/root/dev/data/dataset/SemanticKITTI/preprocess_cw_lowResolution' # for low resolution
    export KITTI_ROOT='/root/dev/data/dataset/SemanticKITTI'
    export KITTI_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/kitti_log'

    export MONOSCENE_OUTPUT='/root/storage/implementation/shared_evtOcc/MonoScene/result'

    export HYDRA_FULL_ERROR=1

    pip install -e ./

    echo "==========================================================================================================="
    echo ""
    source scripts/echo.sh

elif [ "$TARGET_NODE" == "storage" ]; then
    echo ""
    echo "==========================================================================================================="

    export NYU_PREPROCESS='/root/data/NYU_dataset/preprocess' # not used.
    export NYU_ROOT='/root/data/NYU_dataset/depthbin' # not used. 
    export NYU_LOG='/root/data/implementation/shared_evtOcc/MonoScene/nyu_log' # not used.

    export KITTI_PREPROCESS='/root/data0/dataset/SemanticKITTI/preprocess_cw'
    export KITTI_PREPROCESS_LOW='/root/data0/dataset/SemanticKITTI/preprocess_cw_lowResolution' # for low resolution
    export KITTI_ROOT='/root/data0/dataset/SemanticKITTI'
    export KITTI_LOG='/root/dev0/implementation/shared_evtOcc/MonoScene/kitti_log'

    export MONOSCENE_OUTPUT='/root/dev0/implementation/shared_evtOcc/MonoScene/result'
    # /root/data0/dataset/SemanticKITTI
    export HYDRA_FULL_ERROR=1

    pip install -e ./

    echo "==========================================================================================================="
    echo ""
    source scripts/echo.sh


elif [ "$TARGET_NODE" == "n2" ]; then
    echo ""
    echo "==========================================================================================================="

    export NYU_PREPROCESS='/root/data/NYU_dataset/preprocess' # not used.
    export NYU_ROOT='/root/data/NYU_dataset/depthbin' # not used.
    export NYU_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/nyu_log' # not used.

    export KITTI_PREPROCESS='/root/local1/changwoo/preprocess_cw'
    export KITTI_PREPROCESS_LOW='/root/dev/data/dataset/SemanticKITTI/preprocess_cw_lowResolution' # for low resolution
    export KITTI_ROOT='/root/local1/changwoo/'
    export KITTI_EVT_ROOT='/root/dev/data/dataset/SemanticKITTI/event_bin3_onoff'
    export KITTI_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/kitti_log'

    export MONOSCENE_OUTPUT='/root/storage/implementation/shared_evtOcc/MonoScene/result' 

    export HYDRA_FULL_ERROR=1

    pip install -e ./

    echo "==========================================================================================================="
    echo ""
    source scripts/echo.sh

elif [ "$TARGET_NODE" == "n4" ]; then
    echo ""
    echo "==========================================================================================================="

    export NYU_PREPROCESS='/root/data/NYU_dataset/preprocess' # not used.
    export NYU_ROOT='/root/data/NYU_dataset/depthbin' # not used.
    export NYU_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/nyu_log' # not used.

    export KITTI_PREPROCESS='/root/local1/changwoo/preprocess_cw'
    export KITTI_PREPROCESS_LOW='/root/dev/data/dataset/SemanticKITTI/preprocess_cw_lowResolution' # for low resolution
    export KITTI_ROOT='/root/local1/changwoo/SemanticKITTI'
    export KITTI_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/kitti_log'

    export MONOSCENE_OUTPUT='/root/storage/implementation/shared_evtOcc/MonoScene/result' 

    export HYDRA_FULL_ERROR=1

    pip install -e ./

    echo "==========================================================================================================="
    echo ""
    source scripts/echo.sh

elif [ "$TARGET_NODE" == "n6" ]; then
    echo ""
    echo "==========================================================================================================="

    #export NYU_PREPROCESS='/root/data/NYU_dataset/preprocess' # not used.
    #export NYU_ROOT='/root/data/NYU_dataset/depthbin' # not used.
    #export NYU_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/nyu_log' # not used.

    export KITTI_PREPROCESS='/root/data0/dataset/SemanticKITTI/preprocess_cw'
    export KITTI_PREPROCESS_LOW='/root/data0/dataset/SemanticKITTI/preprocess_cw_lowResolution' # for low resolution
    export KITTI_ROOT='/root/data0/dataset/SemanticKITTI'
    export KITTI_LOG='/root/dev0/implementation/shared_evtOcc/MonoScene/kitti_log'
    export KITTI_EVT_ROOT='/root/data0/dataset/SemanticKITTI/event_bin3_onoff'

    export MONOSCENE_OUTPUT='/root/storage/implementation/shared_evtOcc/MonoScene/result'

    export HYDRA_FULL_ERROR=1

    pip install -e ./

    echo "==========================================================================================================="
    echo ""
    source scripts/echo.sh

elif [ "$TARGET_NODE" == "n7" ]; then
    echo ""
    echo "==========================================================================================================="

    export NYU_PREPROCESS='/root/data/NYU_dataset/preprocess' # not used.
    export NYU_ROOT='/root/data/NYU_dataset/depthbin' # not used.
    export NYU_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/nyu_log' # not used.

    export KITTI_PREPROCESS='/root/dev/data/dataset/SemanticKITTI/preprocess_cw'
    export KITTI_PREPROCESS_LOW='/root/dev/data/dataset/SemanticKITTI/preprocess_cw_lowResolution' # for low resolution
    export KITTI_ROOT='/root/dev/data/dataset/SemanticKITTI'
    export KITTI_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/kitti_log'

    export MONOSCENE_OUTPUT='/root/storage/implementation/shared_evtOcc/MonoScene/result'

    export HYDRA_FULL_ERROR=1

    pip install -e ./

    echo "==========================================================================================================="
    echo ""
    source scripts/echo.sh


elif [ "$TARGET_NODE" == "w6" ]; then
    echo ""
    echo "==========================================================================================================="

    export NYU_PREPROCESS='/root/data/NYU_dataset/preprocess' # not used.
    export NYU_ROOT='/root/data/NYU_dataset/depthbin' # not used.
    export NYU_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/nyu_log' # not used.

    export KITTI_PREPROCESS='/root/data0/dataset/SemanticKITTI/preprocess_cw'
    export KITTI_PREPROCESS_LOW='/root/data0/dataset/SemanticKITTI/preprocess_cw_lowResolution' # for low resolution
    export KITTI_ROOT='/root/data0/dataset/SemanticKITTI'
    export KITTI_EVT_ROOT='/root/data0/dataset/SemanticKITTI/event_bin3_onoff'
    export KITTI_LOG='/root/dev0/implementation/shared_evtOcc/MonoScene/kitti_log'

    export MONOSCENE_OUTPUT='/root/storage/implementation/shared_evtOcc/MonoScene/result'

    export HYDRA_FULL_ERROR=1

    pip install -e ./

    echo "==========================================================================================================="
    echo ""
    source scripts/echo.sh



else
    echo "Invalid node argument. Usage: ./run.sh {node or workstation}"
    exit 1
fi






