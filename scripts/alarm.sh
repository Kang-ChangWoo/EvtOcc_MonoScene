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