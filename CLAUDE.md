# Dissertation — Claude Context & Instructions

**Module:** COM748 Masters Research Project  
**Student:** Ahmed Fayyaz Butt (20101228)  
**Supervisor:** Mubashir Ali Cheema  
**Title:** Explainable Exoplanet Transit Classification Using Vision Transformers, LoRA, and Retrieval-Augmented AI on NASA Kepler Light Curves  
**Task checklist:** `docs/TASKS.md`

---

## How to help me
- Be precise and concise
- Explain things step by step when I ask — I am learning as I build
- When I share errors, diagnose the root cause before suggesting fixes
- Keep `docs/TASKS.md` updated as items are completed
- **Any plain-English explanation of a concept, decision, or dataset that comes up in a session must be written into `docs/PROJECT_UNDERSTANDING.md`.** The user studies from that file. Never only put explanations in the chat — always persist them there.

---

## What this project does

The Kepler Space Telescope recorded brightness over time for 200,000 stars. When a planet crosses its star, it causes a faint dip in brightness — a transit. NASA flagged 5,302 suspicious cases (KOIs). This project classifies each KOI as a real planet or false positive using a ViT, and then explains the prediction by retrieving similar confirmed systems from the NASA Exoplanet Archive.

**Novel contribution:** No existing exoplanet system generates a scientifically grounded natural language explanation. The RAG module adds this. It runs after the classifier — it does NOT change predictions.

---

## Architecture

```
Raw FITS file (NASA Kepler / MAST)
        ↓
Lightkurve — download PDCSAP flux, stitch quarters, clean, normalize
        ↓
Phase-fold using koi_period + koi_time0bk → bin to 1024 points
        ↓
pyts — Gramian Angular Field (1D 1024 pts → 2D 64×64 image)
        ↓
ViT-B/16 — classify: CONFIRMED (1) / FALSE POSITIVE (0)
   Four evaluation settings:
   1. Zero-shot   — no adaptation at all
   2. One-shot    — 1 labelled example per class
   3. Few-shot    — small number of labelled examples per class
   4. LoRA        — low-rank matrices into frozen layers via PEFT (~1% params trainable, r=4/8/16)
        ↓
FAISS vector DB (NASA Exoplanet Archive — 6,128 confirmed systems)
        ↓
k=5 most similar confirmed cases by cosine similarity
        ↓
LangGraph agent → natural language explanation output
```

---

## Key decisions — do not change these without good reason

- No CNN models. ViT-B/16 only.
- No "full fine-tuning". The four paradigms above are the comparison.
- Labels: CONFIRMED = 1, FALSE POSITIVE = 0, CANDIDATE dropped (uncertain label).
- GAF size: 64×64. Phase bins: 1024 (Lightkurve path) / 2001 (Mendeley path — pyts downsamples to 64 internally either way).
- Split: 70% train / 15% val / 15% test, stratified on label.
- Weighted cross-entropy loss for class imbalance. Actual balance: 36.2% confirmed (KOI catalogue) / 41.4% confirmed (Mendeley). Compute weights from actual split counts in Notebook 02 — do not hardcode.
- Metrics: F1 and AUC-ROC — NOT raw accuracy (misleading under imbalance).
- CPU notebook for data acquisition. GPU notebook (P100) for training onwards.
- Kaggle Option 2 (manual upload) for now. Move to Kaggle API (Option 3) later.
- Data acquisition path: **Mendeley (`01b`) is now the active path.** `01_data_acquisition.ipynb` preserved as reference.

---

## Datasets

### Overview

| Dataset | Source | Role | Status |
|---------|--------|------|--------|
| NASA Kepler KOI Cumulative Table | NASA Exoplanet Archive | Labels + orbital params | Downloaded ✓ (9,564 rows) |
| Raw Kepler light curves | NASA MAST via Lightkurve | Original flux data per KOI | Cannot use — session timeout |
| Macedo & Zalewski 2024 (Mendeley) | Mendeley Data | Pre-processed light curves | **Active — used in 01b** |
| NASA Exoplanet Archive Stellar Params | Exoplanet Archive API | RAG knowledge base | Pending (Notebook 03+) |
| NASA TESS light curves | Lightkurve / MAST | Zero-shot generalisation only | Objective 5, never in training |

---

### Dataset 1 — NASA Kepler KOI Cumulative Table

**What it is:** The master list of every suspicious brightness dip Kepler detected. NASA analysts reviewed each one and labelled it CONFIRMED, FALSE POSITIVE, or CANDIDATE.

- **URL:** https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
- **Direct CSV:** https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv
- **Rows:** 9,564 total → 7,586 after dropping CANDIDATE
- **Actual class balance:** 36.2% CONFIRMED / 63.8% FALSE POSITIVE (not ~20% as originally estimated)
- **Used for:** Labels in `01_data_acquisition.ipynb`. Not needed in `01b` (Mendeley has labels built in).

Key columns used:

| Column | What it is | Used for |
|--------|-----------|----------|
| `kepid` | Star's Kepler ID | Lightkurve download key |
| `kepoi_name` | Human label e.g. K00001.01 | Logging / reference |
| `koi_disposition` | CONFIRMED / FALSE POSITIVE / CANDIDATE | **The label** |
| `koi_period` | Orbital period in days | Phase-folding |
| `koi_time0bk` | First transit epoch (BKJD) | Phase-folding |
| `koi_duration` | Transit duration in hours | RAG retrieval features |
| `koi_depth` | Brightness drop in ppm | RAG retrieval features |
| `koi_prad` | Planet radius in Earth radii | RAG retrieval features |
| `koi_srad` | Star radius in solar radii | RAG retrieval features |
| `koi_steff` | Star surface temperature in K | RAG retrieval features |

---

### Dataset 2 — Raw Kepler Light Curves (NASA MAST via Lightkurve)

**What it is:** 4 years of brightness measurements for each flagged star, stored as FITS files on NASA's MAST archive. Each star has ~17 quarterly files. Accessed via the `lightkurve` Python library which queries MAST automatically.

- **MAST URL:** https://mast.stsci.edu/
- **Lightkurve docs:** https://docs.lightkurve.org/
- **Used in:** `01_data_acquisition.ipynb` — download, stitch quarters, phase-fold, bin to 1024 points
- **Why it cannot be used on Kaggle:** ~7,586 KOIs × 17 quarters × ~7 sec/file ≈ 15–21 hours of downloading. Kaggle's CPU session limit is 9 hours. When the session ends, `/kaggle/working` is wiped. The batch was confirmed running for 9+ hours and got cut off with 0 files persisted.

---

### Dataset 3 — Macedo & Zalewski 2024 (Mendeley) ← ACTIVE

**What it is:** Pre-processed Kepler light curves — the same raw MAST data already downloaded, phase-folded, cleaned, and normalised by Macedo & Zalewski and published as a 200MB CSV. This replaces the live Lightkurve download for Kaggle.

- **DOI:** `10.17632/wctcv34962.3`
- **URL:** https://data.mendeley.com/datasets/wctcv34962/3
- **Citation:** Macedo, T. & Zalewski, M. (2024). *Dataset for Machine Learning Exoplanet Classification*. Mendeley Data, V3.
- **Size:** ~200 MB
- **Used in:** `01b_data_acquisition_mendeley.ipynb`

Files in `Dataset_Machine_Learning_Exoplanets_2024/`:

| File | Shape | What it contains |
|------|-------|-----------------|
| `all_global.csv` | 5,302 × 2,002 | Full phase-folded light curve — 2001 flux bins + `label` column. **This is what we use.** |
| `all_local.csv` | 5,300 × 202 | Zoomed-in transit window only — 201 flux bins + `label` |
| `lighkurve_KOI_dataset.csv` | 9,564 × 12 | KOI metadata (kepid, period, duration etc.) — no flux data |
| `q1_q17_dr25_sup_koi_*.csv` | — | Supplemental KOI catalogue snapshot |
| `TOI_*.csv` | — | TESS Objects of Interest (not used) |
| `TrES_*.csv` | — | TrES survey data (not used) |

**Why `all_global.csv` and not `all_local.csv`:**  
The global view (2001 bins) represents the full orbital phase — equivalent to what Lightkurve produces and what the ViT needs to see the transit shape in context. The local view (201 bins) is zoomed in on the transit only, which loses the baseline either side. GAF on the global view is the correct choice and matches Choudhary et al. 2025.

**Class balance in Mendeley:**  
2,195 CONFIRMED / 3,107 FALSE POSITIVE = **41.4% confirmed** (after no filtering — Mendeley already excludes candidates).

**Why 200MB when raw data is gigabytes:**  
Raw FITS files store ~70,000 timestamped readings per star. Phase-folding stacks all orbital periods together — the result needs only 2001 numbers to describe the transit shape. That is a 35× compression, stored as plain CSV floats rather than binary FITS.

**Academic legitimacy for dissertation:**  
Acceptable — same underlying NASA/MAST data source, citable published dataset, and the proposal lists it explicitly as a named fallback. 5,302 labelled examples is sufficient for ViT training and evaluation.

---

## KOI catalogue columns we use

| Column | What it is | Used for |
|--------|-----------|----------|
| `kepid` | Star's unique Kepler ID | Lightkurve download key |
| `kepoi_name` | Human label e.g. K00001.01 | Logging / reference |
| `koi_disposition` | CONFIRMED / FALSE POSITIVE / CANDIDATE | **The label** |
| `koi_period` | Orbital period in days | Phase-folding |
| `koi_time0bk` | First transit epoch (BKJD) | Phase-folding |
| `koi_duration` | Transit duration in hours | RAG retrieval features |
| `koi_depth` | Brightness drop in ppm | RAG retrieval features |
| `koi_prad` | Planet radius in Earth radii | RAG retrieval features |
| `koi_srad` | Star radius in solar radii | RAG retrieval features |
| `koi_steff` | Star surface temperature in K | RAG retrieval features |

---

## Tech stack

| Tool | Purpose |
|------|---------|
| Python + PyTorch | Core ML |
| HuggingFace Transformers | ViT-B/16 pretrained model (Apache 2.0) |
| PEFT | LoRA fine-tuning (MIT) |
| Lightkurve | Light curve download from MAST |
| pyts | Gramian Angular Field conversion |
| FAISS | Vector similarity search for RAG |
| LangGraph / LangChain | Agentic explanation pipeline |
| Kaggle P100 GPU (30hr/wk free) | Training compute |

---

## LoRA details

- Ranks tested: r = 4, 8, 16
- Freezes all original ViT weights; inserts small trainable low-rank matrices
- ~600K trainable params at r=8 vs 86M for the full model
- Training time: ~2 hours on P100 for this dataset size
- Library: HuggingFace PEFT

---

## RAG module details

- Runs after ViT classification — does not touch predictions
- FAISS index built from NASA Exoplanet Archive stellar/orbital parameters
- At inference: cosine similarity search, retrieve k=5 most similar confirmed systems
- Output: structured evidence report — stellar type, orbital period, transit depth, confidence
- Extended into LangGraph agent with 3 tools: classifier, retriever, explanation generator
- LLM fallback priority: Claude API free tier → Ollama/Llama-3 → template-based output

---

## Ablation plan (Objective 5)

- ViT zero-shot only (no RAG)
- ViT LoRA only (no RAG)
- ViT LoRA + static RAG (no agentic loop)
- Full system (LoRA + agentic pipeline)
- LoRA rank sweep: r=4 vs r=8 vs r=16
- Retrieval k sweep: k=3 vs k=5 vs k=10
- TESS zero-shot generalisation (no retraining)

---

## Key references

| Ref | Paper | Why it matters |
|-----|-------|---------------|
| [1] | Shallue & Vanderburg 2018 — AstroNet | Historical CNN baseline, cited for context |
| [2] | Dosovitskiy et al. 2021 — ViT | Core architecture |
| [3] | Hu et al. 2022 — LoRA | Parameter-efficient fine-tuning method |
| [4] | Lewis et al. 2020 — RAG | Foundation for the explanation module |
| [5] | Choudhary et al. 2025 — ViT + GAF on Kepler | Most directly relevant prior work (89.46% recall) |
| [6] | Macedo & Zalewski 2024 | Preprocessed Kepler dataset / fallback |

---

## File structure

```
dessertation/
  CLAUDE.md                          ← this file
  notebooks/
    01_data_acquisition.ipynb        ← original Lightkurve path (preserved, not active)
    01b_data_acquisition_mendeley.ipynb  ← Mendeley path (ACTIVE — upload this to Kaggle)
  data/
    Dataset_Machine_Learning_Exoplanets_2024/   ← Mendeley dataset (local copy)
      all_global.csv                 ← 5,302 × 2,002 — full phase-folded light curves (USED)
      all_local.csv                  ← 5,300 × 202  — zoomed transit window (not used)
      lighkurve_KOI_dataset.csv      ← KOI metadata with kepids (not used yet)
      q1_q17_dr25_sup_koi_*.csv      ← KOI catalogue snapshot (not used)
      TOI_*.csv                      ← TESS data (not used)
      TrES_*.csv                     ← TrES data (not used)
  docs/
    TASKS.md                         ← task checklist
    Exoplanet_Proposal_Ahmed_Fayyaz_Butt.docx   ← final accepted proposal
    Literature_Review_Explainable_Exoplanet_Classification.pptx
```

---

## Current state

- Proposal accepted and finalised
- `01_data_acquisition.ipynb`: confirmed Sections 1–4 pipeline is correct. Batch download hit Kaggle 9-hour session limit; cache wiped on restart. Preserved as reference — not the active path.
- `01b_data_acquisition_mendeley.ipynb`: created and ready. Uses Mendeley `all_global.csv` (5,302 KOIs, 2001 bins, labels built in). Sections 1–7 complete — needs Mendeley dataset added as Kaggle input then Run All.
- Mendeley dataset downloaded locally to `data/Dataset_Machine_Learning_Exoplanets_2024/`
- Next step: upload `01b` to Kaggle, add Mendeley as dataset input, update `MENDELEY_INPUT` path, Run All → produces `kepler_gaf_dataset.npz`
