#!/usr/bin/env python3
"""
smolvlm_baseline.py

Phase 1 baseline: load SmolVLM in fp16 on the Jetson GPU, run a constrained
grounding inference on one image, and report cold and steady-state latency.
Greedy decoding, so the action output is deterministic.

Model: HuggingFaceTB/SmolVLM-500M-Instruct (drop to -256M- if latency hurts).

Note: the SmolVLM processor API varies slightly across transformers versions.
If processor(...) raises on the images argument, try images=image instead of
images=[image], and confirm transformers>=4.49.
"""

import argparse
import time

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Constrained action space. Model must reply with exactly one word.
ACTIONS = ["MOVE", "STOP", "FLAG", "REPORT"]
ACTION_PROMPT = (
    "You control a robot. Look at the image and choose exactly one action "
    f"from this list: {', '.join(ACTIONS)}. "
    "Answer with only the single word, nothing else."
)


def load(model_id=MODEL_ID):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    return processor, model


@torch.no_grad()
def infer(processor, model, image, prompt, max_new_tokens=8):
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    }]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat, images=[image], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # keep only the newly generated tokens
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def map_action(text):
    up = text.upper()
    for a in ACTIONS:
        if a in up:
            return a
    return "UNPARSED:" + text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to a test image")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "PyTorch does not see the GPU. Fix the wheel first."
    print(f"torch {torch.__version__} | CUDA {torch.cuda.is_available()}")

    print(f"Loading {args.model} ...")
    t0 = time.monotonic()
    processor, model = load(args.model)
    print(f"Load: {time.monotonic() - t0:.1f} s")

    image = Image.open(args.image).convert("RGB")

    torch.cuda.synchronize()
    t0 = time.monotonic()
    text = infer(processor, model, image, ACTION_PROMPT)
    torch.cuda.synchronize()
    print(f"Cold call: {time.monotonic() - t0:.2f} s -> {text!r} -> {map_action(text)}")

    lat = []
    for _ in range(args.runs):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        infer(processor, model, image, ACTION_PROMPT)
        torch.cuda.synchronize()
        lat.append(time.monotonic() - t0)
    lat.sort()
    print(f"Steady latency over {args.runs}: median {lat[len(lat)//2]:.2f} s | "
          f"min {lat[0]:.2f} s | max {lat[-1]:.2f} s")


if __name__ == "__main__":
    main()
