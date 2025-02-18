#!/bin/bash

echo "Begin!"

# 기존 경로
export KITTI_PREPROCESS='/root/dev/data/dataset/SemanticKITTI/preprocess_cw'
export KITTI_ROOT='/root/dev/data/dataset/SemanticKITTI'
export KITTI_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/kitti_log'
export EVENT_ROOT='/root/dev/data/dataset/SemanticKITTI/event_bin3_onoff/'


# 새 경로 (Node3 local2 세팅)
NEW_BASE='/root/local1/changwoo'

# 복사 작업 수행
# echo "Copying $KITTI_PREPROCESS to $NEW_BASE/preprocess_cw"
# cp -r "$KITTI_PREPROCESS" "$NEW_BASE/preprocess_cw"

# echo "Copying $KITTI_ROOT to $NEW_BASE/SemanticKITTI"
mkdir -p "$NEW_BASE/SemanticKITTI/dataset/poses"
mkdir -p "$NEW_BASE/SemanticKITTI/dataset/sequences"

# cp -r "/root/dev/data/dataset/SemanticKITTI/dataset/poses" "$NEW_BASE/SemanticKITTI/dataset/"
# cp -r "/root/dev/data/dataset/SemanticKITTI/dataset/sequences" "$NEW_BASE/SemanticKITTI/dataset/"



# echo "Copying $KITTI_LOG to $NEW_BASE/kitti_log"
# cp -r "$KITTI_LOG" "$NEW_BASE/kitti_log"

echo "Copying $EVENT_ROOT to $NEW_BASE/SemanticKITTI/event_bin3_onoff"
mkdir -p "$NEW_BASE/SemanticKITTI/event_bin3_onoff"
# cp -r "$EVENT_ROOT" "$NEW_BASE/SemanticKITTI/"

# 환경 변수 업데이트
export KITTI_PREPROCESS="$NEW_BASE/preprocess_cw"
export KITTI_ROOT="$NEW_BASE/SemanticKITTI"
# export KITTI_LOG="$NEW_BASE/kitti_log"
export EVENT_ROOT="$NEW_BASE/SemanticKITTI/event_bin3_onoff"

# 결과 출력
echo "Environment variables updated:"
echo "KITTI_PREPROCESS=$KITTI_PREPROCESS"
echo "KITTI_ROOT=$KITTI_ROOT"
echo "KITTI_LOG=$KITTI_LOG"
echo "EVENT_ROOT=$EVENT_ROOT"


#!/bin/bash

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
      "text": "Data download done. \n $KITTI_ROOT \n $EVENT_ROOT",
      "footer": "from Node 4",
      "ts": $(date +%s)
    }
  ]
}
EOF
)

curl -X POST -H 'Content-type: application/json' \
--data "$PAYLOAD" \
$WEBHOOK_URL

