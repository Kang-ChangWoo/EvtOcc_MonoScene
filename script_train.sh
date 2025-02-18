echo ""
echo "==========================================================================================================="

# echo "NYU_PREPROCESS: $NYU_PREPROCESS"
# echo "NYU_ROOT: $NYU_ROOT"
# echo "NYU_LOG: $NYU_LOG"

echo "KITTI_PREPROCESS: $KITTI_PREPROCESS"
echo "KITTI_PREPROCESS(low_resolution): $KITTI_PREPROCESS_LOW"
echo "KITTI_ROOT: $KITTI_ROOT"
echo "KITTI_LOG: $KITTI_LOG"

echo "MONOSCENE_OUTPUT: $MONOSCENE_OUTPUT"

echo "================================================Training start.================================================"

# Node7: 0, 1, 2, 3, 4, 5 available.

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
CUDA_VISIBLE_DEVICES=0 python monoscene/scripts/train_monoscene.py \
    dataset=kitti \
    enable_log=true \
    kitti_root=$KITTI_ROOT \
    kitti_preprocess_root=$KITTI_PREPROCESS\
    kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW\
    kitti_logdir=$KITTI_LOG \
    n_gpus=1 batch_size=1 \
    low_resolution=true \
    sequence_length=1 \

echo "==========================================================================================================="
echo ""

# [for message to Slack]
WEBHOOK_URL="https://hooks.slack.com/services/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"

# "#36a64f"  # 초록색
# "#FFA500"  # 주황색
# "#FF0000"  # 빨간색
PAYLOAD=$(cat <<EOF
{
  "attachments": [
    {
      "color": "#36a64f", 
      "title": "Server Notification",
      "text": "Training done. \n $KITTI_ROOT \n $EVENT_ROOT",
      "footer": "from -",
      "ts": $(date +%s)
    }
  ]
}
EOF
)

curl -X POST -H 'Content-type: application/json' \
--data "$PAYLOAD" \
$WEBHOOK_URL