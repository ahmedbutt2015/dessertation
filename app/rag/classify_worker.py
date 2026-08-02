"""
Standalone classifier worker script.
Run by subprocess from classifier.py — imports torch/peft without faiss.
Reads a serialised numpy array from stdin, writes JSON result to stdout.

Usage (internal):
  echo '<base64-encoded-npy>' | python classify_worker.py
"""

import sys
import json
import base64
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import timm
from peft import PeftModel
from pathlib import Path
import torch.nn.functional as F

ADAPTER_PATH = Path(__file__).parent.parent.parent / "models" / "best_lora_r16"

MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    base = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    model = model.to(device)
    model.eval()
    return model, device


def preprocess(gaf_b64: str):
    raw = base64.b64decode(gaf_b64)
    arr = np.frombuffer(raw, dtype=np.float32).reshape(64, 64)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # (1,1,64,64)
    t = F.interpolate(t, size=224, mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)
    t = (t - MEAN) / STD
    return t


if __name__ == "__main__":
    try:
        gaf_b64 = sys.stdin.read().strip()
        model, device = load_model()
        tensor = preprocess(gaf_b64).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)
            pred   = logits.argmax(dim=1).item()
            conf   = probs[0, pred].item()

        label = "CONFIRMED" if pred == 1 else "FALSE POSITIVE"
        print(json.dumps({"label": label, "confidence": conf}))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
