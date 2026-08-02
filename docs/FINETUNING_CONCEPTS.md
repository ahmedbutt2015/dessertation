# Fine-Tuning Concepts — Plain English Reference

> This document covers pre-trained models, transfer learning, fine-tuning strategies,
> and LoRA in depth. Written for someone with a software background but no ML theory.
> Study this before presentations or the viva.

---

## What is a Pre-trained Model and Why Do We Use One?

**The analogy:** Imagine you want to hire someone to identify diseased leaves on crops. You have two choices:
1. Hire a person who has never seen anything and train them from scratch
2. Hire a biologist who already knows what healthy tissue looks like, then teach them specifically about crop disease

Option 2 is obviously faster and better. That is exactly what a pre-trained model is.

**ViT-B/16** was trained by Google on **ImageNet** — 14 million photos of everyday objects (cats, cars, chairs, etc.) over weeks on hundreds of GPUs. It learned general visual patterns: edges, textures, shapes, gradients. We take that already-trained model and point it at our problem.

**Why ViT specifically, not CNN?**
A CNN (like the old AstroNet paper) looks at tiny local patches and builds up understanding. A ViT (Vision Transformer) cuts the image into 16×16 pixel patches and treats them like words in a sentence — it learns which patches relate to each other across the whole image at once. For a GAF image where the transit dip could be subtle and spread across the image, the global attention of ViT is better suited than a CNN's local focus.

---

## The Four Evaluation Settings — Why We Test All Four

This is the original contribution of the dissertation. The question being asked is: **how much labelled exoplanet data does a ViT actually need?** That is a real research question nobody has answered for this domain.

---

### Setting 1 — Zero-Shot

**What it is:** Run the model exactly as it came from Google. No changes. No training. Just feed it a GAF image and ask it to classify.

**What "zero" means:** Zero labelled examples from the dataset were used to adapt the model.

**Why test this?** It tells you the baseline — how much does general image knowledge transfer to this very specific scientific problem?

**Result:** F1=0.505, AUC=0.536 — barely better than flipping a coin.

**Analogy:** Handing a fresh biology PhD graduate a GAF image with no context and asking them to classify it. They would guess.

---

### Setting 2 — One-Shot

**What it is:** Show the model exactly 1 labelled example per class (1 confirmed planet, 1 false positive), then evaluate.

**How it works technically:** All 85 million ViT weights stay completely frozen. Only a tiny linear classifier head on top is trained — 1,538 parameters total.

**Why test this?** In many scientific domains, labelled data is expensive. What if you only had 1 confirmed example?

**Result:** F1=0.456, AUC=0.464 — slightly worse than zero-shot.

**Why worse?** With 1 example per class, the head overfits immediately. It memorises those 2 examples and generalises to nothing. This is a known, expected finding in few-shot learning and is worth reporting.

---

### Setting 3 — Few-Shot (10 per class)

**What it is:** Show the model 10 labelled examples per class (20 total), freeze the ViT, train only the linear head.

**Why 10?** "Few-shot" has no fixed number in the literature — it means "very small". The dissertation sets it at 10 as a deliberate design choice.

**Result:** F1=0.731, AUC=0.797 — a big jump from one-shot.

**Key insight:** The ViT learned to "see" patterns from ImageNet. Those patterns (edges, gradients, shapes) happen to also be useful for reading GAF images. That is transfer learning working.

---

### Setting 4 — LoRA Fine-Tuning

**What it is:** Instead of training a tiny head on top of a frozen model, small trainable matrices are injected inside the model itself and those are trained instead.

**Why not just train the whole model?**
- The full ViT has 85 million parameters
- Training all of them needs huge amounts of data and GPU memory
- With 3,711 training examples, full fine-tuning would overfit badly and risk "catastrophic forgetting" — destroying the pre-trained knowledge

**Results:**

| Rank | Trainable Params | % of Model | F1 | AUC | Train Time |
|------|-----------------|------------|-----|-----|------------|
| r=4  | 147,456 | 0.17% | 0.807 | 0.909 | 16.2 min |
| r=8  | 294,912 | 0.34% | 0.820 | 0.910 | 16.3 min |
| r=16 | 589,824 | 0.68% | 0.834 | 0.911 | 16.3 min |

---

## LoRA Deep Dive — What the Numbers Actually Mean

### What is a Weight Matrix?

A neural network is a pile of matrices. Every layer is a grid of numbers. When an image passes through a layer, it gets multiplied by that matrix. The numbers in the matrix are what the model "learned" during pre-training.

For ViT-B/16, the attention layers have weight matrices of shape **768 × 768**.

Think of it like a spreadsheet:
- 768 rows, 768 columns
- 768 × 768 = **589,824 individual numbers** per matrix
- The model has many such matrices

To "fine-tune" means to update those numbers. Updating all of them across all layers = 85 million numbers changing = requires huge GPU memory and lots of data.

---

### What Does "Rank" Actually Mean?

Rank is a linear algebra concept. You can understand it without the maths.

**Simple example — a spreadsheet with hidden redundancy:**

```
2   4   6   8
3   6   9   12
1   2   3   4
5   10  15  20
```

Look at the columns. Column 2 = Column 1 × 2. Column 3 = Column 1 × 3. Column 4 = Column 1 × 4.

All four columns are just one column multiplied by different numbers. The whole 4×4 grid is really just 4 numbers (the first column) + 4 scaling factors. That is **rank 1** — one independent direction of information.

A rank-2 matrix needs two independent columns to reconstruct everything. Rank-4 needs four. And so on.

**A full-rank 768×768 matrix** has 768 independent directions — it can represent anything. But here is the key insight from the LoRA paper (Hu et al. 2022):

> When you adapt a large model to a new task, the *change* you need to make to the weights is low-rank. The original weights need all 768 directions to represent general image knowledge. But the update — the shift from "ImageNet model" to "exoplanet model" — lives in a much smaller subspace.

---

### The LoRA Trick

Instead of updating the full 768×768 matrix W directly, LoRA freezes W and adds two small matrices:

```
W_new = W + (A × B)
```

Where:
- **W** is frozen — 589,824 numbers, never touched
- **A** is shape 768 × r
- **B** is shape r × 768
- **A × B** produces a 768×768 matrix — same shape as W, but built from only 2 × 768 × r numbers

**The actual numbers:**

```
r = 4:
  A = 768 × 4  = 3,072 numbers
  B = 4 × 768  = 3,072 numbers
  Total = 6,144 numbers per matrix pair
  vs full matrix: 589,824 — that is 96× fewer numbers

r = 8:
  Total = 12,288 numbers per matrix pair (48× fewer)

r = 16:
  Total = 24,576 numbers per matrix pair (24× fewer)
```

---

### Where LoRA Is Injected

LoRA targets the **attention layers** — specifically Query and Value projection matrices. This is where the model decides "which patches of the image should I pay attention to?"

ViT-B/16 has 12 transformer blocks. Each block has Q and V projections. 12 blocks × 2 matrices = 24 matrix pairs adapted.

That is exactly where the 147,456 trainable parameters at r=4 come from:
- 24 matrix pairs × 6,144 numbers each = 147,456

---

### What Rank Controls

| Rank | What it means | Risk |
|------|---------------|------|
| r=1 | One adjustment direction — very rigid | Underfitting |
| r=4 | Four directions — conservative | Low overfitting risk |
| r=8 | Eight directions — moderate | Balanced |
| r=16 | Sixteen directions — expressive | Higher overfitting risk on small data |

**Higher rank = more expressive = more parameters = more risk of overfitting on small data.**

In the results, r=16 training loss dropped to 0.196 by epoch 10, while r=4 only reached 0.312. r=16 was learning harder and faster — showing it has more capacity but also more overfitting potential.

---

### Params Per Training Sample — Why This Matters

```
Your training set:    3,711 samples
Full fine-tuning:    85,800,194 params  →  ~23,110 params per sample (dangerous)
LoRA r=4:               147,456 params  →      ~40 params per sample (very safe)
LoRA r=8:               294,912 params  →      ~79 params per sample (safe)
LoRA r=16:              589,824 params  →     ~159 params per sample (borderline)
```

A rough rule: you want more training samples than parameters being trained. At r=4 you are extremely conservative. At r=16 you are approaching the edge — which is why overfitting signs appear at epoch 7+.

---

### At Inference Time — Zero Extra Cost

After training, A and B can be merged back:

```python
W_final = W + (A @ B)   # @ = matrix multiply
```

The result is just a normal 768×768 matrix. Zero extra latency. Zero extra memory. The model looks identical to the original from the outside. This is why LoRA is used in production — you keep the base model and swap in different A/B pairs for different tasks.

---

### LoRA — The Chef Analogy

Imagine a world-class chef (the pre-trained model). Knows how to cook everything. Someone asks them to cook only Bengali food (the new task).

The chef does not forget how to cook. They learn a small set of adjustments — more mustard oil, different spice balance, certain techniques. Those adjustments are low-rank — a small number of consistent changes applied across existing knowledge.

LoRA is those adjustments. Matrix A learns "which direction to push." Matrix B learns "how much to push in each layer."

---

## How to Choose Which Approach for Any ML Problem

```
Do you have labelled data?
├── No (or <5 examples)
│   └── Zero-shot or One-shot
│       Results will be weak unless task is very similar to pre-training
│
├── Very small (5–50 examples)
│   └── Few-shot: freeze backbone, train head only
│
└── Hundreds to thousands (your case: 3,711)
    └── Fine-tuning
        ├── Lots of GPU + data similar to pre-training
        │   └── Full fine-tuning
        └── Limited GPU + data different from pre-training  ← YOUR CASE
            └── LoRA
                ├── Task slightly different from pre-training  → r=4
                ├── Task moderately different                  → r=8 (default)
                └── Task very different                        → r=16
```

**Why LoRA was the right choice here:** 3,711 training samples, GAF images are very different from everyday ImageNet photos, free T4 GPU with 16GB VRAM. LoRA is the principled choice. Full fine-tuning would need tens of thousands of examples.

---

## The Key Dissertation Finding From Objective 3

> LoRA r=4 with 0.17% of parameters trained, in 16 minutes on a free GPU, achieves F1=0.807 — a 60% relative improvement over zero-shot (F1=0.505). Going to r=16 only adds +0.027 F1 at 4× the parameter count. The training time is flat across all ranks (~16 minutes) because the bottleneck is the forward pass through the frozen 85M parameter backbone, not the trainable portion.

This is an original, citable finding. No prior paper has published this LoRA rank comparison for Kepler exoplanet classification.
