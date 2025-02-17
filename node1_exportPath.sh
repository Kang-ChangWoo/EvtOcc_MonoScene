echo ""
echo "==========================================================================================================="
export NYU_PREPROCESS='/root/data/NYU_dataset/preprocess'
export NYU_ROOT='/root/data/NYU_dataset/depthbin'
export NYU_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/nyu_log'

export KITTI_PREPROCESS='/root/dev/data/dataset/SemanticKITTI/preprocess_cw'
export KITTI_ROOT='/root/dev/data/dataset/SemanticKITTI'
export KITTI_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/kitti_log'

export MONOSCENE_OUTPUT='/root/storage/implementation/shared_evtOcc/MonoScene/result'

export HYDRA_FULL_ERROR=1

pip install -e ./

echo "================================================Export done.================================================"

echo "NYU_PREPROCESS: $NYU_PREPROCESS"
echo "NYU_ROOT: $NYU_ROOT"
echo "NYU_LOG: $NYU_LOG"

echo "KITTI_PREPROCESS: $KITTI_PREPROCESS"
echo "KITTI_ROOT: $KITTI_ROOT"
echo "KITTI_LOG: $KITTI_LOG"

echo "MONOSCENE_OUTPUT: $MONOSCENE_OUTPUT"

echo "==========================================================================================================="
echo ""