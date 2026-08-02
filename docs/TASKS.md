# Task Checklist — Exoplanet Dissertation

> Update this file as work progresses. Mark items `[x]` when done.  
> Last updated: 2026-07-24

---

## Proposal
- [x] Write and submit proposal
- [x] Supervisor review
- [x] Proposal accepted (final version: `Exoplanet_Proposal_Ahmed_Fayyaz_Butt.docx`)

---

## Objective 1 — Literature Review

- [ ] Review: CNN approaches to exoplanet classification (AstroNet baseline)
- [ ] Review: Vision Transformers (ViT) and GAF image representations
- [ ] Review: LoRA and parameter-efficient fine-tuning methods
- [ ] Review: RAG applied to scientific/knowledge-intensive tasks
- [ ] Review: Choudhary et al. 2025 (ViT + GAF on Kepler — most relevant prior work)
- [ ] Compile literature review chapter / slides

---

## Objective 2 — Data Acquisition & Preprocessing ✓ COMPLETE

**Active Notebook:** `notebooks/01b_data_acquisition_mendeley.ipynb`

- [x] Notebook scaffolded (sections 1–8, Kaggle-ready)
- [x] Lightkurve path confirmed working end-to-end but too slow for Kaggle — switched to Mendeley
- [x] Load Mendeley `all_global.csv` → 5,302 KOIs, 2,001 flux bins, labels built in
- [x] Filter flat-line rows (safety net — 0 rows dropped)
- [x] Rescale each row to [-1, 1]
- [x] GAF conversion → X shape `(5302, 64, 64)`
- [x] Stratified 70/15/15 split (train 3711 / val 795 / test 796, ~41.4% confirmed each)
- [x] Save → `kepler_gaf_dataset.npz` + `valid_kois.csv`
- [x] Verify all splits load cleanly — confirmed

---

## Objective 3 — ViT-B/16 Evaluation Across Four Settings ✓ COMPLETE

**Notebook:** `notebooks/02_vit_evaluation.ipynb`

### Setup
- [x] Load pretrained ViT-B/16 from HuggingFace Transformers
- [x] Set up evaluation harness (F1, AUC-ROC, inference latency)

### Zero-Shot
- [x] Run ViT-B/16 on GAF images with no task-specific adaptation
- [x] Record metrics — F1=0.505, AUC=0.536

### One-Shot
- [x] Adapt ViT-B/16 with 1 labelled example per class
- [x] Record metrics — F1=0.456, AUC=0.464

### Few-Shot
- [x] Adapt ViT-B/16 with small number of labelled examples per class (n=10)
- [x] Record metrics — F1=0.731, AUC=0.797

### LoRA Fine-Tuning
- [x] Set up PEFT library with LoRA configuration
- [x] Train with r = 4 — F1=0.807, AUC=0.909, 16.2 min, 0.17% trainable params
- [x] Train with r = 8 — F1=0.820, AUC=0.910, 16.3 min, 0.34% trainable params
- [x] Train with r = 16 — F1=0.834, AUC=0.911, 16.3 min, 0.68% trainable params
- [x] Compare LoRA ranks — r=16 wins marginally; r=4 most parameter-efficient
- [x] Best LoRA model (r=16) saved → `models/best_lora_r16/` (adapter_config.json + adapter_model.safetensors)

### Results Summary
- [x] Compile comparison table — `models/best_lora_r16/metrics.csv`
- [ ] Write Objective 3 results section

---

## Objective 4 — RAG Module + Agentic Pipeline ✓ COMPLETE

### RAG Module
- [x] Download NASA Exoplanet Archive cumulative KOI table (9,564 rows, 50 cols) → `data/.../koi_cumulative.csv`
- [x] Extract 2,745 CONFIRMED planets with full stellar/orbital parameters
- [x] Scale 6 features (period, duration, depth, prad, steff, srad) with StandardScaler
- [x] L2-normalise feature vectors for cosine similarity via inner product
- [x] Build FAISS IndexFlatIP (6D, 2,745 vectors) → `models/rag/faiss_index.bin`
- [x] Save metadata → `models/rag/confirmed_planets.csv`
- [x] Save scaler → `models/rag/scaler.pkl`
- [x] Implement `app/rag/retriever.py` — retrieve(query_dict, k) and retrieve_by_koi(id, k)
- [x] Implement `app/rag/classifier.py` — loads LoRA r=16 adapter, preprocesses GAF, returns label + confidence
- [x] Test retrieval — K00001.01 → Kepler-718 b (similarity 0.999), scientifically sensible neighbours confirmed

### Agentic Pipeline (LangGraph)
- [x] Implement `app/rag/agent.py` — 3-node LangGraph StateGraph
    - Node 1: build_context (format retrieved neighbours into evidence string)
    - Node 2: generate_explanation (Claude claude-haiku-4-5 → template fallback)
    - Node 3: END
- [x] Wire Tool 1: ViT classifier (`app/rag/classifier.py`)
- [x] Wire Tool 2: FAISS knowledge retriever (`app/rag/retriever.py`)
- [x] Wire Tool 3: Explanation generator (Claude API → template fallback)
- [x] Gradio app (`app/app.py`) updated — Tab 1 search + Tab 2 full pipeline + Tab 3 results
- [x] End-to-end test: retrieval + LangGraph agent produces coherent explanation ✓
- [ ] End-to-end test with live ViT classifier (pending torch install)
- [ ] Validate explanation quality on 10+ known confirmed/FP cases
- [ ] Write Objective 4 results section

---

## Objective 5 — Ablation Study + TESS Generalisation

### Ablation
- [ ] Ablation 1: ViT zero-shot only (no RAG, no fine-tuning) — results already in metrics.csv
- [ ] Ablation 2: ViT LoRA only (no RAG) — results already in metrics.csv
- [ ] Ablation 3: ViT LoRA + static RAG (no agentic loop)
- [ ] Ablation 4: Full system (LoRA + agentic pipeline)
- [ ] Ablation 5: LoRA rank sweep — r=4 vs r=8 vs r=16 — results already in metrics.csv
- [ ] Ablation 6: Retrieval k sweep — k=3 vs k=5 vs k=10
- [ ] Compile ablation results table

### TESS Generalisation
- [x] Download NASA TESS light curves via Lightkurve (no changes to trained model)
- [x] Apply Kepler-trained pipeline zero-shot to TESS — F1=0.2222, AUC=0.6250 (14 targets, all predicted FP)
- [x] Record performance drop / transfer effectiveness — F1 drop -0.6116 (73.4%), AUC drop -0.2856
- [ ] Write Objective 5 results section

---

## Dissertation Write-Up

- [ ] Chapter 1: Introduction
- [ ] Chapter 2: Literature Review
- [ ] Chapter 3: Methodology
- [ ] Chapter 4: Implementation
- [ ] Chapter 5: Results & Evaluation
- [ ] Chapter 6: Discussion & Conclusion
- [ ] Appendix: Gantt chart
- [ ] Appendix: Ablation tables
- [ ] Final proofread and submission

---

## Ongoing / Admin

- [ ] Set up private GitHub repo for code (weekly commits)
- [x] Kaggle notebook environment — GPU P100 access confirmed
- [ ] Fortnightly supervisor meetings with Mubashir Ali Cheema
- [ ] Maintain Gantt chart in dissertation appendix
