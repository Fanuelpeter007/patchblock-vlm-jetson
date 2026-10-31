#!/usr/bin/env bash
# setup.sh - environment for this project on a Jetson Orin Nano
# (JetPack 6.2 / L4T R36.4 / CUDA 12.6, Python 3.10). Run once:   bash setup.sh
#
# Versions are the ones the results were produced with. This script was
# reconstructed from that working environment; it has not been re-run on a
# freshly flashed board.
set -e

echo "== 1/4 system packages =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libopenblas-dev python3-opencv

echo "== 2/4 virtual environment (sees system OpenCV) =="
python3 -m venv --system-site-packages ~/patchblock-venv
source ~/patchblock-venv/bin/activate
python -m pip install --upgrade pip

echo "== 3/4 PyTorch for Jetson (must come from this index) =="
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

echo "== 4/4 model and analysis stack =="
# transformers is pinned: 5.x removed AutoModelForVision2Seq, used by these scripts.
# numpy is pinned so pip does not upgrade it past what torch was built against.
python -m pip install "transformers==4.49.0" "pillow>=10" accelerate httpx tqdm \
  scikit-learn "numpy==1.26.4"

python -c "import torch, numpy, transformers; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), '| numpy', numpy.__version__, '| transformers', transformers.__version__)"
echo "Done. Activate in new terminals with: source ~/patchblock-venv/bin/activate"
