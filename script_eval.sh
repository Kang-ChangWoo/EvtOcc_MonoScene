echo ""
echo "==========================================================================================================="

# [for NYU]
# echo "NYU_PREPROCESS: $NYU_PREPROCESS"
# echo "NYU_ROOT: $NYU_ROOT"
# echo "NYU_LOG: $NYU_LOG"

# [for KITTI]
echo "KITTI_PREPROCESS: $KITTI_PREPROCESS"
echo "KITTI_PREPROCESS(low_resolution): $KITTI_PREPROCESS_LOW"
echo "KITTI_ROOT: $KITTI_ROOT"
echo "KITTI_LOG: $KITTI_LOG"

echo "MONOSCENE_OUTPUT: $MONOSCENE_OUTPUT"

echo "==============================================Start Evaluation.=============================================="

# Node7: 0, 1, 2, 3, 4, 5 available.

# [for NYU]
# python monoscene/scripts/eval_monoscene.py \
#     dataset=NYU \
#     NYU_root=$NYU_ROOT\
#     NYU_preprocess_root=$NYU_PREPROCESS \
#     n_gpus=1 batch_size=1

# [for KITTI]
CUDA_VISIBLE_DEVICES=2 python monoscene/scripts/eval_monoscene.py \
    dataset=kitti \
    kitti_root=$KITTI_ROOT\
    kitti_preprocess_root=$KITTI_PREPROCESS \
    kitti_preprocess_lowRes_root=$KITTI_PREPROCESS_LOW \
    n_gpus=1 batch_size=1

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
      "color": "#FFA500", 
      "title": "Server Notification",
      "text": "Evaluation done. \n $KITTI_ROOT \n $EVENT_ROOT",
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