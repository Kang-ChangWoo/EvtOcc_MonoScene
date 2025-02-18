#!/bin/bash

# 기존 경로
# export KITTI_PREPROCESS='/root/dev/data/dataset/SemanticKITTI/preprocess_cw'
export KITTI_ROOT='/root/dev/data/dataset/SemanticKITTI'
# export KITTI_LOG='/root/storage/implementation/shared_evtOcc/MonoScene/kitti_log'
export EVENT_ROOT='/root/dev/data/dataset/SemanticKITTI/event_bin3_onoff/'


# 새 경로 (Node3 local2 세팅)
NEW_BASE='/root/local2/changwoo'

# 복사 작업 수행
# echo "Copying $KITTI_PREPROCESS to $NEW_BASE/preprocess_cw"
# cp -r "$KITTI_PREPROCESS" "$NEW_BASE/preprocess_cw"

# echo "Copying $KITTI_ROOT to $NEW_BASE/SemanticKITTI"
# mkdir -p "$NEW_BASE/SemanticKITTI/dataset/poses"
# mkdir -p "$NEW_BASE/SemanticKITTI/dataset/sequences"

# cp -r "/root/dev/data/dataset/SemanticKITTI/dataset/poses" "$NEW_BASE/SemanticKITTI/dataset/poses"
# cp -r "/root/dev/data/dataset/SemanticKITTI/dataset/sequences" "$NEW_BASE/SemanticKITTI/dataset/sequences"



# echo "Copying $KITTI_LOG to $NEW_BASE/kitti_log"
# cp -r "$KITTI_LOG" "$NEW_BASE/kitti_log"

echo "Copying $EVENT_ROOT to $NEW_BASE/SemanticKITTI/event_bin3_onoff"
# mkdir -p "$NEW_BASE/SemanticKITTI/event_bin3_onoff"
# cp -r "$EVENT_ROOT" "$NEW_BASE/SemanticKITTI/event_bin3_onoff"

# 환경 변수 업데이트
export KITTI_PREPROCESS="$NEW_BASE/preprocess_cw"
export KITTI_ROOT="$NEW_BASE/SemanticKITTI"
# export KITTI_LOG="$NEW_BASE/kitti_log"
export EVENT_ROOT="$NEW_BASE/SemanticKITTI/event_bin3_onoff"