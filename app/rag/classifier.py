"""
LoRA classifier module.

Runs the ViT-B/16 + LoRA r=16 classifier in a subprocess to avoid
a known macOS conflict between faiss-cpu and peft/torch in the same process.

Model path: models/best_lora_r16/  (adapter_config.json + adapter_model.safetensors)
Base model:  timm vit_base_patch16_224 (downloaded from HuggingFace on first call ~346MB)

Preprocessing mirrors Notebook 02 exactly:
  64×64 float32 → bilinear resize to 224×224 → replicate to 3 channels
  → normalise with mean=0.5, std=0.5
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ADAPTER_PATH = Path(__file__).parent.parent.parent / "models" / "best_lora_r16"
_WORKER = Path(__file__).parent / "classify_worker.py"


def classify(gaf_image: np.ndarray) -> tuple[str, float]:
    """
    Classify a 64×64 GAF image.

    Runs in a subprocess so faiss (main process) and torch/peft (worker) never
    share the same process — avoids a segfault on macOS.

    Parameters
    ----------
    gaf_image : np.ndarray of shape (64, 64) or (1, 64, 64), dtype float32

    Returns
    -------
    label      : 'CONFIRMED' or 'FALSE POSITIVE'
    confidence : float 0–1
    """
    arr = gaf_image
    if arr.ndim != 2:
        arr = arr.squeeze()
    if arr.shape != (64, 64):
        raise ValueError(f"Expected (64, 64) GAF image, got {arr.shape}")

    arr = arr.astype(np.float32)
    encoded = base64.b64encode(arr.tobytes()).decode("ascii")

    result = subprocess.run(
        [sys.executable, str(_WORKER)],
        input=encoded,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Classifier worker failed:\n{result.stderr.strip()}"
        )

    data = json.loads(result.stdout.strip())
    if "error" in data:
        raise RuntimeError(data["error"])

    return data["label"], data["confidence"]


def is_available() -> bool:
    """Return True if adapter weights exist and torch/timm/peft are importable."""
    if not ADAPTER_PATH.exists():
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch, timm, peft"],
            capture_output=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False
