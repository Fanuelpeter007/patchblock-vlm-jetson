#!/usr/bin/env python3
"""
check_env.py - Phase 1 gates 1 and 2.

Confirms PyTorch sees the Jetson GPU, then confirms the USB camera returns a
frame and saves it as test.jpg for the SmolVLM baseline to use.

Run with the venv active:
    source ~/patchblock-venv/bin/activate
    python check_env.py
"""

import sys

# ---- GATE 1: PyTorch + CUDA ----
try:
    import torch
except ImportError:
    print("FAIL: torch not installed. Run setup.sh first.")
    sys.exit(1)

print(f"torch version : {torch.__version__}")
print(f"CUDA visible  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name      : {torch.cuda.get_device_name(0)}")
else:
    print("FAIL: PyTorch does not see the GPU. Wrong wheel.")
    print("      Reinstall torch from https://pypi.jetson-ai-lab.io/jp6/cu126")
    sys.exit(1)

# ---- GATE 2: camera ----
try:
    import cv2
except ImportError:
    print("FAIL: cv2 not available. Install python3-opencv (setup.sh does this).")
    sys.exit(1)

cam = cv2.VideoCapture(0)          # 0 = first USB camera
ok, frame = cam.read()
cam.release()

if not ok:
    print("FAIL: camera 0 returned no frame. Is the USB webcam plugged in?")
    print("      Try index 1: edit VideoCapture(0) -> VideoCapture(1).")
    sys.exit(1)

cv2.imwrite("test.jpg", frame)
print(f"camera OK     : frame {frame.shape}, saved test.jpg")
print("\nGATE 1 and GATE 2 passed.")
