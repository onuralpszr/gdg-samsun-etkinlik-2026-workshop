#!/usr/bin/env bash
# Step 3: High speed execution with the `ultralytics-inference` CLI
#
#   ./local_inference/run_cli.sh install     build the CLI with OpenVINO and video support
#   ./local_inference/run_cli.sh image       single image
#   ./local_inference/run_cli.sh batch       a whole folder in batches of 4
#   ./local_inference/run_cli.sh video       a video file, frame by frame with flat memory
#   ./local_inference/run_cli.sh webcam      live webcam window (CAMERA=50 for Intel MIPI laptop cameras)
#   ./local_inference/run_cli.sh tasks       all seven YOLO26 tasks
#   ./local_inference/run_cli.sh intel       OpenVINO on Intel CPU, GPU and NPU
#   ./local_inference/run_cli.sh rust        the Cargo project in this folder
#   ./local_inference/run_cli.sh all         everything except the webcam
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/.venv/bin/python"
YOLO="${ROOT}/.venv/bin/yolo"
CLI="ultralytics-inference"
VIDEO_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"
IMAGE_URL="https://ultralytics.com/images/bus.jpg"
CAMERA="${CAMERA:-0}"

# Let the OpenVINO native backend find the runtime that pip installed into .venv
if [ -x "$PY" ] && "$PY" -c "import openvino" 2>/dev/null
then
  OV_LIBS="$("$PY" -c "import openvino, os; print(os.path.dirname(openvino.__file__))")/libs"
  export LD_LIBRARY_PATH="${OV_LIBS}:${LD_LIBRARY_PATH:-}"
fi

say() {
  printf "\n\033[1;34m==> %s\033[0m\n" "$*"
}

need_assets() {
  mkdir -p videos
  [ -f bus.jpg ] || curl -sL -o bus.jpg "$IMAGE_URL"
  [ -f videos/solutions_ci_demo.mp4 ] || curl -sL -o videos/solutions_ci_demo.mp4 "$VIDEO_URL"
  [ -f yolo26n.onnx ] || "$YOLO" export model=yolo26n.pt format=onnx simplify=True
  [ -d yolo26n_openvino_model ] || "$YOLO" export model=yolo26n.pt format=openvino
}

install_cli() {
  say "Installing the Ultralytics Inference CLI with OpenVINO and video support"
  cargo install ultralytics-inference --features "openvino-native,video"
  "$CLI" --version
}

image() {
  say "Single image on CPU with ONNX Runtime"
  "$CLI" predict --model yolo26n.onnx --source bus.jpg
}

batch() {
  say "Folder of images in batches of 4"
  mkdir -p runs/cli_batch_input
  cp bus.jpg runs/cli_batch_input/
  [ -f zidane.jpg ] || curl -sL -o zidane.jpg https://ultralytics.com/images/zidane.jpg
  cp zidane.jpg runs/cli_batch_input/
  "$CLI" predict --model yolo26n.onnx --source runs/cli_batch_input --batch 4
}

video() {
  say "Video file, one frame in memory at a time, with saving turned off for minimal overhead"
  "$CLI" predict --model yolo26n_openvino_model --device intel:gpu --source videos/solutions_ci_demo.mp4 --save false --verbose false
  say "Same video with the annotated MP4 saved under runs/detect/predict"
  "$CLI" predict --model yolo26n_openvino_model --device intel:gpu --source videos/solutions_ci_demo.mp4
}

webcam() {
  say "Webcam $CAMERA in a live window, people only, press q or Esc to quit"
  "$CLI" predict --model yolo26n_openvino_model --device intel:gpu --source "$CAMERA" --show --save false --classes 0
}

tasks() {
  for task in detect segment semantic depth classify pose obb
  do
    say "Task: $task"
    "$CLI" predict --task "$task" --save true --verbose false
  done
}

intel() {
  for device in intel:cpu intel:gpu intel:npu
  do
    say "OpenVINO on $device"
    "$CLI" predict --model yolo26n_openvino_model --device "$device" --source bus.jpg --save false
  done
}

rust() {
  say "Building the workshop Cargo project with OpenVINO and video"
  cargo build --release --manifest-path local_inference/Cargo.toml --features "openvino,video"
  BIN=local_inference/target/release/gdg-inference
  "$BIN" image yolo26n.onnx bus.jpg
  DEVICE=intel:gpu "$BIN" bench yolo26n_openvino_model bus.jpg 200
  DEVICE=intel:gpu "$BIN" video yolo26n_openvino_model videos/solutions_ci_demo.mp4
}

MODE="${1:-all}"
case "$MODE" in
  install) install_cli ;;
  image | batch | video | webcam | intel | rust)
    need_assets
    "$MODE"
    ;;
  tasks) tasks ;;
  all)
    need_assets
    image
    batch
    video
    tasks
    intel
    rust
    ;;
  *)
    echo "Unknown mode '$MODE'. Pick install, image, batch, video, webcam, tasks, intel, rust or all."
    exit 1
    ;;
esac
