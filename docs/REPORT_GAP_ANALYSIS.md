# Report Gap Analysis — What We Have vs What the Rubric Rewards

> Written 2026-08-02. Checked against `sample/COM748 Rubric.docx` and the highest-marked sample report (`sample/Zunair_Ahmad_Primary_Report-3.docx`).

---

## How COM748 is marked

The rubric has three parts:

| Part | Marks | Components |
|------|-------|-----------|
| **Research Paper** | 50 | Abstract (5), Existing Work (10), Methodology (10), Results (10), Discussion & Conclusions (10), References & Presentation (5) |
| **Supplementary Material** | 20 | Extended literature (5), Lifecycle/project management (5), Verification & Validation (5), Critical appraisal (5) |
| **Presentation/Viva** | 30 | Problem understanding (5), Depth & accuracy (10), Critical evaluation of own work (5), Demo + testing (5), Organisation (5) |

The sample report is an **IEEE-style research paper, ~5,200 words**: Abstract + Keywords → Introduction (Motivation, Research Objectives) → Literature Review (themed subsections ending with "Research Gaps and Study Rationale") → Methodology → Results and Analysis → Discussion and Implications → Conclusion and Future Work → References. Our report follows this shape.

---

## What is already strong

- **Complete quantitative story for the classifier.** All four settings run with clean, monotonic results (zero-shot 0.50 F1 → LoRA r=16 0.83 F1 / 0.91 AUC). The rank sweep (r=4/8/16) is done. This directly feeds Methodology + Results.
- **Working end-to-end system**, not just notebooks: FAISS index (2,745 confirmed planets), retriever, LangGraph agent, Gradio app. Rubric's "demonstration of solution" (viva, 5 marks) is well covered.
- **Genuine novelty claim** — retrieval-grounded natural language explanations for transit classification. No prior system does this; the sample report scored highly with a much weaker novelty claim.
- **Honest negative result** (TESS transfer failure) — examiners reward this under "awareness of limitations" if framed as a finding, not a failure.
- **Defensible data decisions** already documented: Mendeley over live MAST download (with the 9-hour Kaggle evidence), global vs local view, dropping CANDIDATEs, weighted loss, F1/AUC over accuracy.

---

## Gaps — ordered by marks at risk

### 1. The explainability claim has no evaluation (Results, 10 marks + Critical appraisal, 5 marks)
The novel contribution is the explanation module, but nothing measures explanation quality. Ablations 3 & 4 (static RAG vs agentic) and "validate explanations on 10+ known cases" are unticked in TASKS.md.
**Do:** run the pipeline on ~15 test KOIs (mix of confirmed/FP, correct/incorrect predictions). Score each explanation on a simple rubric — factual grounding (does every claim trace to a retrieved neighbour or KOI parameter?), completeness, hallucination count — comparing template output vs LLM output. Even a small human-scored table transforms the novelty claim from asserted to evidenced. **This is the single highest-value remaining task.**

### 2. No per-class metrics, confusion matrix, or ROC curves (Results, 10 marks)
We report macro F1 and AUC only. The rubric wants results "clearly presented and related to the broader field" — Choudhary et al. 2025 report 89.46% *recall*, so without our recall we cannot make the headline comparison to the closest prior work.
**Do:** re-run test-set evaluation for the saved LoRA r=16 adapter and dump: per-class precision/recall/F1, confusion matrix, ROC curve. Save figures as PNG for the report.

### 3. Only 6 references (References & Presentation, 5 marks + both literature criteria, 10 marks)
The sample report cites 20+. Ours needs Kepler/TESS mission papers (Borucki 2010, Ricker 2015), GAF encoding (Wang & Oates 2015), the ML-for-exoplanets line (Pearson 2018, Ansdell 2018, Armstrong 2021, Valizadegan 2022 ExoMiner), and explainability literature (Grad-CAM, Rudin 2019, hallucination surveys) to motivate the RAG module. Target 25–30 references. The report draft includes these — verify each before submission.

### 4. TESS sample is 14 targets (Methodology + Results)
n=14 with all-FP predictions is too thin to state a transfer result confidently; an examiner will ask.
**Do (if time):** expand to 50+ TESS targets. **Otherwise:** frame explicitly as a pilot probe and state the n=14 limitation in the same sentence as the result — never let the examiner say it first.

### 5. One-shot < zero-shot anomaly needs an explanation (Discussion, 10 marks)
0.456 vs 0.505 F1 looks like an error unless explained: a single example per class overfits the classification head to two arbitrary points, while zero-shot with a random head sits near chance. Say this in the report; it shows understanding rather than a bug.

### 6. Single-seed runs (Critical appraisal, 5 marks)
All results are one seed. Ideally re-run LoRA r=16 with 3 seeds and report mean ± std (~50 min total on P100). If not feasible, state it as a limitation explicitly.

### 7. Verification & Validation narrative missing (Supplementary, 5 marks)
The evidence exists but is scattered: stratified-split verification, flat-line filtering (0 rows dropped — the check itself is V&V), the retrieval sanity test (K00001.01 → Kepler-718 b, sim 0.999), LLM → template fallback testing, end-to-end pipeline test. **Do:** collect into a V&V section for the supplementary material — this is nearly free marks.

### 8. Remaining engineering items
- End-to-end test with the live ViT classifier (blocked on torch install locally) — needed before the viva demo.
- GitHub repo with commit history (lifecycle criterion, 5 marks).
- Gantt chart for the appendix (lifecycle criterion).
- Figures for the report: pipeline diagram, light-curve → GAF example, four-setting bar chart, ROC + confusion matrix, screenshot of the Gradio app with a generated explanation.

### 9. Retrieval k justification (Methodology)
The k sweep only recorded mean similarity (0.987/0.984/0.976 for k=3/5/10). That's fine — since RAG never alters predictions, classification metrics are the wrong yardstick — but the report must say *why* k=5: enough neighbours for a robust evidence base, before mean similarity dilutes below ~0.98.

---

## Priority order — status as of 2026-08-03

1. ~~Explanation-quality evaluation on 15 cases (Gap 1)~~ ✅ DONE — `results/explanation_eval.{csv,md}`, folded into report §4.4. **Caveat:** the `tmpl_hallucinations` column is inconsistent (identical deterministic template scored 0–5 across cases while grounding is 5/5 everywhere) — report wording avoids citing it; re-score or drop that column if an examiner asks.
2. ~~Per-class metrics + confusion matrix + ROC figures (Gap 2)~~ ✅ DONE — `results/lora_r16_*`, folded into report Table 2. Note: re-eval AUC is 0.920 vs 0.911 from the training run — verify the eval used identical preprocessing so the two numbers can be presented together.
3. ~~V&V write-up (Gap 7)~~ ✅ DONE — report Appendix A.
4. ~~Figures (Gap 8)~~ ✅ DONE except light-curve → GAF example image (see figure mapping note at end of report draft).
5. ~~TESS (Gap 4)~~ ✅ DECIDED — keep n=14; failure mode is cadence mismatch (physics), not sampling noise. One-line "n ≥ 50 is the immediate next step" added to §4.5.
6. Multi-seed LoRA run if time permits (Gap 6) — still open; currently framed as limitation #1 in §5.
7. GitHub + Gantt (admin, but marked) — still open.
8. Convert draft to final format (.docx), verify all references, complete Choudhary [11] citation.
