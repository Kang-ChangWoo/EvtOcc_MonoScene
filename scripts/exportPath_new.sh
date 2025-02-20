#!/bin/bash
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
sourch scripts/echo.sh



## ============================
## ==========  MEMO ===========
## ============================