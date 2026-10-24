#!/usr/bin/env bash
# Build the workshop .venv with the right PyTorch for your machine
#
#   ./setup_env.sh          Intel GPU and NPU laptops (PyTorch XPU plus OpenVINO)
#   ./setup_env.sh cuda     NVIDIA GPU machines
#   ./setup_env.sh cpu      Apple silicon (MPS) and any other machine
set -euo pipefail

BACKEND="${1:-xpu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"
cd "$(dirname "$0")"

case "$BACKEND" in
  xpu) TORCH_INDEX="https://download.pytorch.org/whl/xpu" ;;
  cuda) TORCH_INDEX="https://download.pytorch.org/whl/cu130" ;;
  cpu) TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
  *)
    echo "Unknown backend '$BACKEND'. Pick xpu, cuda or cpu."
    exit 1
    ;;
esac

if command -v uv >/dev/null 2>&1
then
  echo "Using uv to create .venv with Python $PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" .venv
  PIP=(uv pip install --python .venv/bin/python)
else
  echo "uv not found, falling back to python3 venv"
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  PIP=(.venv/bin/python -m pip install)
fi

echo "Installing PyTorch for $BACKEND from $TORCH_INDEX"
"${PIP[@]}" torch torchvision --index-url "$TORCH_INDEX"

echo "Installing Ultralytics YOLO, OpenVINO, ONNX and friends"
"${PIP[@]}" -r requirements.txt

echo "Checking your hardware"
.venv/bin/python local_python/00_check_env.py

echo "All set. Activate with: source .venv/bin/activate"
