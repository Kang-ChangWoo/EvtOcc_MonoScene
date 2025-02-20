#!/bin/bash
source scripts/echo.sh
echo ""
echo "==========================================================================================================="

export NYU_PREPROCESS='/root/data/NYU_dataset/preprocess'
export NYU_ROOT='/root/data/NYU_dataset/depthbin'
export NYU_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/nyu_log'

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



## ============================
## ==========  MEMO ===========
## ============================