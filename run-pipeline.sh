#!/usr/bin/env bash
set -euo pipefail

for argument in "$@"; do
  if [[ $argument == --help || $argument == -h ]]; then
    exec tracking-demo --help
  fi
done

if [[ $# -eq 0 || ! -f "$1" ]]; then
  echo "Usage: run-pipeline /input/video.mp4 [tracking options]" >&2
  exit 2
fi

video="$1"
shift
filename="$(basename -- "$video")"
clip="${filename%.*}"
segment="artifacts/$clip/segment_001"

tracking-demo "$video" --court-projection "$@"

pose-demo all "$segment" \
  --device cuda:0 \
  --pose2d-python /opt/pose2d/bin/python \
  --mmpose-config /opt/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py \
  --mmpose-checkpoint /opt/weights/rtmpose.pth \
  --pose3d-python /opt/pose3d/bin/python \
  --motionbert-repository /opt/MotionBERT \
  --motionbert-config /opt/MotionBERT/configs/pose3d/MB_ft_h36m_global_lite.yaml \
  --motionbert-checkpoint /opt/weights/motionbert.bin

exec export-scene "$segment" --output "artifacts/$clip/scene.json"
