#!/usr/bin/env bash
# 定期检查 openbee 图片上传进度，发现错误则打印并退出非 0
# 只检查日志最后 500 行，避免把历史错误当成当前错误

LOG="${1:-/root/LLaVAOV2.0-Caption2VQA_V2/upload_openbee_images.log}"
INTERVAL="${MONITOR_INTERVAL:-180}"
PATTERN="upload_openbee_images_to_hf.sh"
TAIL_LINES=500

while true; do
  if ! pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 上传进程已结束"
    if grep -q "全部完成" "$LOG" 2>/dev/null; then
      echo "  -> 正常完成"
      exit 0
    fi
    if tail -"$TAIL_LINES" "$LOG" 2>/dev/null | grep -qE "Error|Traceback|BadRequestError|SSLError|HTTPError|504|400"; then
      echo "  -> 检测到错误，最后 20 行:"
      tail -20 "$LOG" 2>/dev/null
      exit 1
    fi
    exit 0
  fi
  if tail -"$TAIL_LINES" "$LOG" 2>/dev/null | grep -qE "BadRequestError|400 Client Error|504 Server Error"; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 检测到错误（仅检查最近 ${TAIL_LINES} 行）"
    tail -30 "$LOG" 2>/dev/null
    exit 1
  fi
  echo "[$(date '+%H:%M:%S')] 运行中 | $(grep -E '已提交|上传:.*->|跳过' "$LOG" 2>/dev/null | tail -2)"
  sleep "$INTERVAL"
done
