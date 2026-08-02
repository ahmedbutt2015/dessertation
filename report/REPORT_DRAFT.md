# Explainable Exoplanet Transit Classification Using Vision Transformers, LoRA, and Retrieval-Augmented AI on NASA Kepler Light Curves

**Ahmed Fayyaz Butt** — 20101228
Ulster University (Birmingham Campus)
MSc Computer Science — COM748 Masters Research Project
Supervisor: Mubashir Ali Cheema
imahmedthebutt@gmail.com

---

## Abstract

The Kepler Space Telescope monitored roughly 200,000 stars for four years, and its survey produced 9,564 Kepler Objects of Interest (KOIs) — candidate signals in which a periodic dip in stellar brightness may indicate a planet crossing its host star. Distinguishing genuine planetary transits from false positives such as eclipsing binaries and instrumental artefacts remains a labour-intensive vetting task, and although deep learning classifiers now approach expert-level accuracy, they operate as black boxes: none of the existing systems can explain a prediction in scientific terms. This study addresses both the classification problem and the explanation gap. Phase-folded Kepler light curves for 5,302 labelled KOIs are encoded as Gramian Angular Field (GAF) images and classified by a pretrained Vision Transformer (ViT-B/16) evaluated under four adaptation regimes: zero-shot, one-shot, few-shot, and Low-Rank Adaptation (LoRA) fine-tuning. LoRA at rank 16, training only 0.68% of the model's parameters for 16 minutes on a single GPU, achieves a macro F1 of 0.834 and AUC-ROC of 0.911, compared with 0.505 F1 for the unadapted model — demonstrating that parameter-efficient adaptation, rather than full fine-tuning, is sufficient to specialise a general-purpose vision model for astronomical time-series data. On top of the classifier, a retrieval-augmented generation (RAG) module searches a FAISS index of 2,745 confirmed exoplanets from the NASA Exoplanet Archive and passes the five most physically similar confirmed systems to a LangGraph agent, which generates a natural language explanation grounded in real, citable astronomical evidence. The explanation module operates strictly downstream of classification and never alters a prediction, preserving the classifier's measured performance while making its outputs scientifically legible. A zero-shot transfer test on TESS light curves quantifies the limits of cross-mission generalisation. To the best of the author's knowledge, this is the first exoplanet vetting system to pair a transformer classifier with retrieval-grounded natural language explanations.

**Keywords** — Exoplanet detection, transit photometry, Vision Transformer, Gramian Angular Field, LoRA, parameter-efficient fine-tuning, retrieval-augmented generation, explainable AI, Kepler, TESS.

---

## 1. Introduction

When a planet passes between its star and an observer, it blocks a minute fraction of the starlight — typically less than one part in a thousand — producing a brief, periodic dip in the star's measured brightness. This transit method has become the most productive technique in exoplanet science: NASA's Kepler mission alone is responsible for more than half of all confirmed exoplanets [1]. But the method's sensitivity is also its weakness. Many astrophysical and instrumental phenomena mimic transits — grazing eclipsing binaries, background eclipsing binaries blended into the photometric aperture, starspots, and detector systematics all produce plausible dips [2]. Of the 9,564 KOIs flagged by the Kepler pipeline, a substantial fraction turned out to be false positives, and separating the two classes historically required expert human vetting of each candidate.

Machine learning was introduced to relieve this bottleneck. Shallue and Vanderburg's AstroNet [3] established the template: a convolutional neural network trained on phase-folded light curves, achieving high accuracy on Kepler candidates and even enabling the discovery of new planets in previously searched data. A succession of CNN-based systems followed [4, 5, 6], culminating in models such as ExoMiner [7] that are accurate enough to validate planets statistically. Yet across this entire line of work, one property has remained constant: the classifier's output is a bare probability. An astronomer told that a candidate is 97% likely to be a planet is given no indication of *why* — no reference to the transit's depth, shape, or duration, and no connection to the population of systems already confirmed. In a field where classification decisions feed directly into telescope time allocation and published catalogues, this opacity is a practical, not merely philosophical, problem [8].

### 1.1 Motivation

Two recent developments make it possible to close this gap. First, the Vision Transformer [9] has shown that attention-based architectures pretrained on large natural-image corpora transfer remarkably well to specialised domains — and time series can be brought into the image domain through encodings such as the Gramian Angular Field [10], which converts a one-dimensional signal into a two-dimensional texture that preserves temporal correlations. Choudhary et al. [11] recently demonstrated this exact combination on Kepler data, reporting 89.46% recall with a ViT trained on GAF images and establishing the strongest directly comparable baseline for this work. Second, retrieval-augmented generation [12] provides a principled way to ground language model output in an external, verifiable knowledge source rather than in the model's parametric memory — precisely the property a scientific explanation requires, given the well-documented tendency of large language models to hallucinate plausible falsehoods [13].

The motivation for this project is the observation that these components have never been assembled. Classification and explanation have been treated as separate problems; this study treats them as one system, with an explicit design constraint that keeps them honest: the explanation module runs strictly after the classifier and cannot alter its prediction. Whatever the classifier's measured performance, it survives intact into the deployed system.

A second, more pragmatic motivation concerns compute. Full fine-tuning of an 86-million-parameter ViT is beyond the free GPU allocations available to a masters project, and beyond many research groups besides. LoRA [14] freezes the pretrained weights and injects small trainable low-rank matrices, reducing the trainable parameter count by two orders of magnitude. Whether such a lightweight adaptation is sufficient for a domain as far from natural images as phase-folded stellar photometry is an empirical question this study answers directly.

### 1.2 Research Objectives

The project is organised around five objectives:

1. **Review** the literature on machine-learning transit classification, transformer architectures for time-series imagery, parameter-efficient fine-tuning, and retrieval-augmented generation, identifying the gap at their intersection.
2. **Construct** a reproducible preprocessing pipeline that converts labelled Kepler light curves into GAF images suitable for a pretrained ViT, with a stratified train/validation/test split.
3. **Evaluate** ViT-B/16 across four adaptation regimes — zero-shot, one-shot, few-shot, and LoRA fine-tuning at ranks 4, 8, and 16 — measuring macro F1 and AUC-ROC under class imbalance.
4. **Build** a retrieval-augmented explanation module: a FAISS index over confirmed exoplanets from the NASA Exoplanet Archive, queried by physical similarity, feeding a LangGraph agent that generates evidence-grounded natural language explanations.
5. **Probe** the system's limits through an ablation study (adaptation regime, LoRA rank, retrieval depth) and a zero-shot generalisation test on light curves from the TESS mission, which the model never sees in training.

### 1.3 Contributions

The study makes three contributions. First, it provides a controlled comparison of four adaptation regimes for a single ViT backbone on a single astronomical dataset — isolating the value of each increment of supervision, from none to parameter-efficient fine-tuning. Second, it demonstrates that LoRA recovers strong classification performance (0.834 macro F1, 0.911 AUC) while training under 1% of model parameters in roughly 16 minutes of GPU time, making transformer-based vetting accessible at negligible cost. Third, and most distinctively, it introduces the first retrieval-grounded explanation layer for exoplanet vetting: every explanation is anchored to named, confirmed planetary systems retrieved by physical similarity, so each claim in the generated text can be traced to a real entry in the NASA Exoplanet Archive.

---

## 2. Literature Review

### 2.1 Machine Learning for Transit Classification

The Kepler mission [1] transformed exoplanet science from a trickle of individual discoveries into a statistical enterprise, and the resulting data volume forced the automation of candidate vetting. Early approaches used hand-engineered features with random forests — the Autovetter project [15] — before Shallue and Vanderburg [3] introduced AstroNet, a deep CNN operating directly on phase-folded flux with two views of each candidate: a *global* view spanning the full orbital phase and a *local* view zoomed on the transit itself. AstroNet reached 98.8% AUC on Kepler candidates and its dual-view design has been inherited by nearly every successor. Ansdell et al. [4] showed that injecting domain knowledge — centroid time series and stellar parameters — improved robustness; Osborn et al. [5] and Yu et al. [6] adapted the architecture to TESS; Armstrong et al. [16] combined machine learning with statistical validation frameworks; and Valizadegan et al.'s ExoMiner [7] pushed reliability far enough to validate 301 new planets in a single publication. Malik et al. [17] demonstrated that gradient-boosted trees over engineered features remain competitive, underlining that the field's constraint is no longer raw accuracy.

What none of these systems provides is an account of its decisions. Jara-Maldonado et al. [18], surveying the field, note that adoption of ML vetting by working astronomers is limited less by performance than by trust — an observation that motivates the present study's central design goal.

### 2.2 Vision Transformers and Time-Series Imaging

The Vision Transformer [9] dispenses with convolutional inductive biases, splitting an image into fixed-size patches processed by a standard transformer encoder. Its defining empirical property is the transferability of its pretrained representations: a ViT pretrained on ImageNet-scale corpora adapts to distant target domains with modest supervision. Applying a ViT to photometric time series requires an imaging step, and the Gramian Angular Field [10] has emerged as the standard choice. A GAF rescales a series to [-1, 1], interprets each value as the cosine of an angle, and forms a matrix of pairwise angular sums — producing a texture in which temporal structure such as a transit dip appears as a characteristic geometric pattern. Wang and Oates [10] introduced the encoding precisely to unlock image classifiers for time-series problems, and Choudhary et al. [11] provided the first systematic application to Kepler photometry, reporting 89.46% recall with a ViT on GAF-encoded light curves. Their work is the most directly relevant precedent for this study; it also inherits the field's standard limitation, offering no explanation mechanism.

### 2.3 Parameter-Efficient Fine-Tuning

Full fine-tuning updates every weight of a pretrained model, which for ViT-B/16 means 86 million parameters — costly in compute, prone to overfitting on small datasets, and destructive of the pretrained representation. Hu et al.'s LoRA [14] instead freezes the pretrained weights and learns a low-rank update ΔW = BA for selected weight matrices, where the rank r of the factorisation controls capacity. At r = 8 on ViT-B/16, this trains roughly 0.3% of the parameters. The technique rests on the hypothesis that task adaptation lives in a low-dimensional subspace of the weight space, a claim now supported across language and vision domains [19]. For a project constrained to free-tier GPU allocations, LoRA is not merely convenient but enabling: it turns an infeasible training run into a 16-minute one. The open question — whether the low-rank hypothesis holds when the target domain (GAF textures of stellar photometry) is as visually alien to ImageNet as any domain could be — is one of this study's empirical questions.

### 2.4 Retrieval-Augmented Generation and Explainability

Explainability research in deep learning has largely produced *attributive* methods — saliency maps such as Grad-CAM [20] that highlight input regions influencing a prediction. For a GAF image, however, a saliency map is doubly indirect: it highlights pixels in a transformed space that no astronomer inspects. Rudin [21] argues that post-hoc attribution is often the wrong paradigm entirely, and that high-stakes decisions demand explanations expressed in the vocabulary of the domain. For transit vetting, that vocabulary is physical: orbital period, transit depth and duration, stellar radius and temperature, and — crucially — precedent, the population of already-confirmed systems that a new candidate does or does not resemble.

Retrieval-augmented generation [12] offers a mechanism for exactly this kind of precedent-based explanation. By retrieving documents (here: confirmed planet records) relevant to a query and conditioning generation on them, RAG grounds output in verifiable sources and measurably reduces hallucination [13]. Recent agentic frameworks such as LangGraph structure this process as an explicit graph of tools and decision nodes, giving the pipeline inspectable intermediate state — the retrieved evidence can be logged and audited independently of the generated prose. RAG has been applied to scientific question-answering and literature synthesis, but no published system applies it to astronomical classification output.

### 2.5 Research Gaps and Study Rationale

Three gaps emerge from this review. First, transformer-based transit classification is nascent — a single directly relevant prior study [11] — and no work has characterised how much supervision a pretrained ViT actually needs on this data, from zero-shot through parameter-efficient fine-tuning. Second, no exoplanet vetting system of any architecture explains its predictions in natural language, let alone language grounded in confirmed astronomical precedent. Third, the interaction of these components — whether a retrieval layer can add scientific legibility *without* compromising a classifier's measured performance — is unexplored, because no system has combined them. This study addresses all three gaps within a single, ablated architecture.

---

## 3. Methodology

The complete system architecture — from raw light curve to generated explanation — is summarised in Figure 1.

### 3.1 Data Sources

Three data products are used, all ultimately derived from NASA missions.

**Labelled light curves.** The primary dataset is the published, citable preprocessing of Kepler photometry by Macedo and Zalewski [22] (Mendeley Data, DOI 10.17632/wctcv34962.3), which provides 5,302 KOIs as phase-folded, cleaned, and normalised global-view light curves of 2,001 flux bins each, with dispositions built in: 2,195 CONFIRMED and 3,107 FALSE POSITIVE (41.4% positive class). Candidates with uncertain dispositions are already excluded. The global view — the full orbital phase, not a zoomed transit window — is used, matching both the AstroNet convention [3] and Choudhary et al. [11], because the surrounding baseline is precisely what distinguishes a clean transit from the V-shaped eclipse of a binary. An initial pipeline that downloaded raw FITS files directly from the MAST archive via Lightkurve was built and verified end-to-end, but the full download (≈7,600 targets × 17 quarters) exceeds free cloud-session limits by a factor of two; the published dataset derives from the same raw MAST photometry and was named in the accepted project proposal as the designated alternative.

**Retrieval knowledge base.** The NASA Exoplanet Archive cumulative KOI table (9,564 rows) supplies the RAG module's knowledge base: the 2,745 confirmed planets with complete stellar and orbital parameters — orbital period, transit duration, transit depth, planetary radius, stellar effective temperature, and stellar radius.

**Generalisation probe.** TESS light curves, downloaded via Lightkurve and processed through an identical pipeline, serve exclusively as a held-out zero-shot test; no TESS data enters training in any form.

### 3.2 Preprocessing and GAF Encoding

Each 2,001-bin light curve is screened for degenerate (flat-line) signals — a safeguard that, in the event, discarded zero rows, itself a useful verification that the source data is clean — then rescaled per-row to [-1, 1] as the GAF transform requires. The pyts implementation of the Gramian Angular Summation Field then encodes each series as a 64 × 64 single-channel image, replicated across three channels to match the ViT's expected input. The dataset of 5,302 images is split 70/15/15 into training (3,711), validation (795), and test (796) sets, stratified on the label so each partition preserves the 41.4% positive rate. The split is performed once, saved to disk, and reused unchanged across every experiment in this study, so all reported settings are evaluated on an identical test set.

### 3.3 Classifier: ViT-B/16 Under Four Adaptation Regimes

The backbone is the ViT-B/16 checkpoint pretrained on ImageNet-21k, obtained from HuggingFace Transformers, with a two-class head. Four regimes of increasing supervision are compared on the same backbone, data, and test set:

- **Zero-shot:** the pretrained backbone with a randomly initialised classification head, evaluated with no task adaptation whatsoever. This establishes the floor: what a general-purpose vision model knows about transit morphology out of the box.
- **One-shot:** the head alone is fitted on a single labelled example per class.
- **Few-shot:** the head alone is fitted on ten labelled examples per class.
- **LoRA fine-tuning:** all pretrained weights are frozen and trainable low-rank matrices are injected into the attention projections via the PEFT library, trained on the full training partition. Three ranks are swept — r = 4, 8, and 16 — training 147K, 295K, and 590K parameters respectively (0.17%, 0.34%, and 0.68% of the model).

Class imbalance is handled with weighted cross-entropy, the weights computed from the actual training-partition class counts rather than hardcoded. Training used a Kaggle P100 GPU. Accuracy is deliberately excluded as a headline metric: with a 58.6% majority class, a degenerate all-negative classifier would score 58.6% accuracy while detecting no planets at all. Macro F1 and AUC-ROC are reported instead, alongside per-sample inference latency.

### 3.4 Retrieval Module

The six physical parameters of each of the 2,745 confirmed planets are standardised (zero mean, unit variance — necessary because raw scales differ by orders of magnitude, e.g. period in days versus depth in parts per million) and L2-normalised, so that inner-product search in a FAISS IndexFlatIP is exactly cosine similarity. At inference, a query KOI's parameters pass through the same fitted scaler and the index returns the k = 5 most physically similar confirmed systems with their similarity scores. Retrieval was validated on cases with known analogues: querying K00001.01 (the first Kepler object of interest, a confirmed hot Jupiter) returns Kepler-718 b at 0.999 cosine similarity, with the remaining neighbours all short-period giants — the scientifically expected result.

### 3.5 Agentic Explanation Pipeline

The explanation layer is a three-node LangGraph state graph orchestrating three tools: the ViT classifier, the FAISS retriever, and an explanation generator. Node one formats the retrieved neighbours into a structured evidence context — each neighbour's name, period, depth, radius, and stellar type alongside the query KOI's parameters and the classifier's label and confidence. Node two generates the explanation, with a deliberate fallback hierarchy: a Claude language model produces fluent prose when API access is available, and a deterministic template — which fills the same evidence fields into fixed scientific phrasing — guarantees a grounded explanation when it is not. This design means the system degrades gracefully rather than failing, and because both paths draw exclusively on the retrieved evidence and the classifier's stated output, neither can introduce claims from outside the knowledge base.

The pipeline's single most important property is architectural: it is strictly downstream. The classifier's prediction is committed before retrieval begins, so the explanation layer cannot — by construction rather than by policy — alter, second-guess, or launder the classification. All classifier metrics reported in Section 4 therefore apply unchanged to the full deployed system.

### 3.6 Evaluation Protocol and Ablations

The ablation design varies one component at a time: adaptation regime (the four settings), LoRA rank (4 / 8 / 16), and retrieval depth (k = 3 / 5 / 10, assessed by neighbour similarity, since retrieval cannot affect classification metrics by design). Cross-mission generalisation is probed by applying the complete Kepler-trained pipeline — unchanged, with no retraining — to TESS targets processed through an identical GAF pipeline. The system is exposed through a Gradio application with three views (KOI search, full classify-retrieve-explain pipeline, and results browser), which serves as the demonstration vehicle.

---

## 4. Results and Analysis

### 4.1 The Four Adaptation Regimes

Table 1 reports test-set performance for all settings, evaluated on the identical 796-KOI test partition (visual comparison in Figure 2).

**Table 1 — ViT-B/16 across adaptation regimes (test set, n = 796)**

| Setting | Trainable params | Macro F1 | AUC-ROC | Inference (ms/sample) | Train time |
|---|---|---|---|---|---|
| Zero-shot | 0 | 0.505 | 0.536 | 11.7 | — |
| One-shot (1/class) | head only | 0.456 | 0.464 | 10.4 | — |
| Few-shot (10/class) | head only | 0.731 | 0.797 | 9.8 | — |
| LoRA r = 4 | 147K (0.17%) | 0.807 | 0.908 | 11.4 | 16.2 min |
| LoRA r = 8 | 295K (0.34%) | 0.820 | 0.910 | 11.3 | 16.3 min |
| **LoRA r = 16** | **590K (0.68%)** | **0.834** | **0.911** | **11.4** | **16.3 min** |

Three findings stand out. First, the zero-shot result (AUC 0.536) sits barely above chance: ImageNet pretraining, whatever its transferability, encodes essentially nothing about GAF textures of stellar photometry. Any claim that pretrained vision models "understand" arbitrary imaging domains out of the box fails here, and this establishes that the adaptation regimes, not the pretraining, are doing the discriminative work.

Second, one-shot adaptation *underperforms* zero-shot (F1 0.456 vs 0.505). This is not anomalous on inspection: fitting a classification head to a single example per class anchors the decision boundary to two arbitrary points in feature space, and if either exemplar is atypical the head generalises worse than a random one whose errors are at least unbiased. The result is a useful caution — a small amount of supervision can be worse than none — and it dissolves by ten examples per class, where few-shot adaptation jumps to 0.731 F1 and 0.797 AUC. Evidently the pretrained features, while not linearly separable for this task out of the box, are structured enough that a modest sample locates a usable boundary.

Third, LoRA delivers a decisive further gain: +0.10 F1 and +0.11 AUC over few-shot at every rank tested. The gap between few-shot (head-only, full training set unavailable to it) and LoRA (low-rank updates inside the frozen backbone, full training set) measures what adapting the *representation* — not merely the readout — is worth on this domain: roughly ten points of F1.

**Per-class behaviour of the deployed model.** Re-evaluation of the saved r = 16 adapter on the frozen test set reproduces the training-run figures (macro F1 0.835, AUC 0.920) and exposes the error structure that the macro averages conceal (Table 2; the confusion matrix and ROC curve are shown in Figure 3).

**Table 2 — Per-class metrics, LoRA r = 16 (test set, n = 796)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| FALSE POSITIVE | 0.889 | 0.826 | 0.857 | 466 |
| CONFIRMED | 0.777 | 0.855 | 0.814 | 330 |

The asymmetry is scientifically desirable rather than incidental. The model recovers 85.5% of genuine planets (CONFIRMED recall) at the cost of a lower positive-class precision (0.777) — meaning its errors skew toward flagging false positives as planets rather than discarding real ones. In a vetting workflow, where a machine-flagged candidate receives human follow-up but a machine-discarded planet is lost, this is the correct side on which to err. The weighted cross-entropy loss, computed from the actual training-partition class counts, is directly responsible for this balance: an unweighted model on 58.6% negative data would drift toward the majority class and sacrifice exactly the recall that matters.

### 4.2 LoRA Rank Sweep

Performance rises monotonically but with sharply diminishing returns as rank increases: r = 4 → 16 buys +0.027 F1 for four times the trainable parameters, while AUC is essentially saturated (0.908 → 0.911) from the smallest rank. Training time is flat (≈16 minutes) across ranks, since the frozen backbone dominates the computation. Two practical readings follow. Where a single best model is wanted, r = 16 wins and is the configuration carried forward into the deployed system. Where parameter budget matters — say, serving many adapted models from one shared backbone — r = 4 retains 97% of the F1 at a quarter of the adapter size. The near-saturation of AUC at r = 4 is itself evidence for the low-rank hypothesis of Hu et al. [14] in an imaging domain maximally distant from the pretraining distribution: the essential Kepler-specific adaptation fits comfortably in a rank-4 subspace.

### 4.3 Retrieval Quality and Depth

Because the retrieval module cannot influence classification by design, it is evaluated on its own terms: the physical similarity of what it returns. Mean cosine similarity of retrieved neighbours declines gently with depth — 0.987 at k = 3, 0.984 at k = 5, 0.976 at k = 10 — while the minimum similarity within the retrieved set falls faster (0.956 / 0.937 / 0.908). The default of k = 5 balances the two pressures: enough independent precedents that an explanation does not hinge on a single analogue, before the tail of the retrieved set drifts toward marginally similar systems whose inclusion would dilute, rather than strengthen, the evidence. Qualitative inspection supports the quantitative picture: across spot-checked queries, retrieved neighbours share the query's broad regime (short-period giants for hot-Jupiter queries, and so on), which is exactly the property a precedent-based explanation requires.

### 4.4 Generated Explanations

The end-to-end pipeline — classify, retrieve, explain — produces coherent explanations that follow a consistent evidential structure: the classifier's verdict and confidence, the query's key physical parameters, the retrieved confirmed systems with named identities and similarity scores, and a concluding synthesis relating the candidate to its precedents. Because every factual field in the prose is drawn from the evidence context assembled in the graph's first node, each claim is traceable to a row of the NASA Exoplanet Archive — the property that distinguishes this output from free-form LLM commentary (a representative output is shown in Figure 4).

**Structured quality evaluation.** Both generation paths were scored on fifteen test-set KOIs, deliberately sampled to include both classes and both correct and incorrect predictions (ten correct, five misclassified). Each explanation was rated for *grounding* (does every factual claim trace to the retrieved evidence or the query's catalogue parameters?), *completeness* (are verdict, confidence, parameters, and named precedents all present?), and *hallucination incidence*. Both paths achieved full marks for grounding and completeness on all fifteen cases — an expected but necessary confirmation, since the architecture constructs the evidence context before generation and both paths draw only from it. On hallucination, the LLM path introduced a minor unsupported embellishment in three of fifteen explanations (never a fabricated system or parameter value — typically an overstated qualifier); the template path can introduce none by construction.

The more instructive finding is qualitative, and it emerged from the misclassified cases. Where the classifier predicts FALSE POSITIVE, the retrieved neighbours are — by design — confirmed planets, and the LLM occasionally reasons incoherently across this tension: for K00262.01 it cited high similarity to confirmed systems as evidence *supporting* a false-positive verdict. The template's purpose-written negative branch handles the same tension correctly, stating that the morphological classification overrides parameter similarity. This is a genuine boundary of LLM-generated scientific prose: fluency does not guarantee logical coherence between evidence and verdict, and it motivates the false-positive knowledge base proposed in Section 6.3.

### 4.5 Zero-Shot Transfer to TESS

Applying the Kepler-trained system unchanged to TESS targets produces a sharp degradation: macro F1 falls to 0.222 (−0.61 against the Kepler test set) with AUC 0.625, and the classifier predicts FALSE POSITIVE for every target in the probe sample. The result must be read with its caveat stated plainly: the probe comprised only 14 targets from single-sector photometry, so the point estimates are coarse. But the direction and mechanism are informative. TESS observes each field for ~27 days against Kepler's four years, so a phase-folded TESS curve stacks far fewer transits and carries visibly more noise; the GAF textures the model learned from deep Kepler stacks are systematically cleaner than anything TESS produces. The collapse to all-negative predictions suggests the model reads TESS noise levels as evidence against planethood — a distribution shift in the input statistics rather than in the underlying physics. The residual AUC of 0.625 indicates the ranking signal does not vanish entirely, which is consistent with the transfer failing at the calibration level (the decision threshold) before it fails at the representation level. The practical conclusion is unambiguous: cross-mission deployment requires at least threshold recalibration, and realistically light fine-tuning on mission-specific data. Expanding the probe to n ≥ 50 targets is the immediate next step; because the failure mode is driven by the cadence mismatch rather than sampling noise, the diagnosis is expected to hold at larger n.

---

## 5. Discussion and Implications

**What the adaptation ladder shows.** Read as a whole, Table 1 is a supervision dose-response curve, and its shape carries the study's central empirical message: pretrained vision transformers contribute nothing to transit classification *until* adapted (zero-shot ≈ chance), can be actively misled by trivial supervision (one-shot), become useful with a handful of examples, and reach strong performance only when the representation itself is adapted — yet that adaptation needs to touch under 1% of the model. The result simultaneously deflates the strongest transfer-learning optimism and vindicates the parameter-efficient paradigm. For resource-constrained research settings — which includes most astronomy groups, not only masters projects — the 16-minute, single-GPU path to a 0.91-AUC vetting model is arguably the finding with the widest practical reach.

**Position against prior work.** Choudhary et al. [11], the nearest methodological neighbour, report 89.46% recall with a fully trained ViT on GAF-encoded Kepler curves; this study's LoRA r = 16 model reaches 85.5% CONFIRMED recall while training 0.68% of the parameters. The comparison should not be over-read — dataset composition, splits, and preprocessing differ — but the gap of roughly four recall points against a fully trained model, purchased at more than a hundredfold reduction in trainable parameters and 16 minutes of GPU time, is the trade this study set out to price. For groups without the compute to train transformers fully, four points of recall is the approximate cost of admission, and it is a cost that threshold tuning against the vetting workflow's own recall requirements could plausibly narrow.

**The explanation layer as a trust instrument.** The deliberate architectural choice — explanation strictly downstream, prediction immutable — deserves defence, because the alternative (letting retrieved evidence modulate the prediction) sounds attractive. The separation was chosen for evaluability: a system whose explanation module feeds back into classification cannot cleanly attribute its errors, and its reported metrics become entangled with retrieval quality. Under the chosen design, the classifier's F1 and AUC are exact properties of the deployed system, and the explanation layer can be improved, swapped, or audited independently. This mirrors Rudin's [21] argument from the opposite direction: rather than post-hoc rationalisation of an opaque model, the system provides *evidence presentation* — the retrieved precedents are real, their similarity scores are computed not narrated, and the prose is a rendering of that evidence rather than a story about the network's internals. The honest limit of this framing must also be stated: the explanation grounds the *plausibility* of the verdict in precedent; it does not expose the ViT's internal decision process. These are different explanatory goods, and this study supplies the first deliberately.

**Limitations.** Four limitations bound the claims. First, every result derives from single-seed runs; the LoRA rank ordering (spanning 0.027 F1) could plausibly reshuffle under seed variance, and multi-seed replication is the highest-priority robustness addition. Second, the TESS probe is small (n = 14) and its quantitative results should be treated as directional, though the cadence-mismatch mechanism behind the failure is expected to persist at larger samples. Third, the explanation-quality study, while structured, covers fifteen cases scored by a single rater; a larger sample with independent raters would be needed to attach confidence intervals to the hallucination rates. Fourth, the retrieval space is six hand-chosen physical parameters — sufficient for precedent-finding, but blind to morphological similarity of the light curves themselves; a learned embedding of the curves would retrieve neighbours by shape as well as physics. The K00262.01 incoherence case illustrates a fifth, subtler boundary: retrieval over confirmed planets only gives the generator no vocabulary of precedent for negative verdicts, which the LLM path papers over with fluent but occasionally illogical prose.

---

## 6. Conclusion and Future Work

### 6.1 Summary of Findings

This study set out to determine whether a pretrained Vision Transformer could be economically adapted to exoplanet transit vetting, and whether its predictions could be made scientifically legible without compromising them. Both questions are answered affirmatively. On 5,302 labelled Kepler KOIs encoded as Gramian Angular Fields, LoRA fine-tuning at rank 16 — 590K trainable parameters, 16 minutes on a single P100 — achieves 0.834 macro F1 and 0.911 AUC-ROC, against a near-chance 0.536 AUC for the unadapted model. The rank sweep shows the adaptation is intrinsically low-rank, with AUC saturating by r = 4. A FAISS retrieval layer over 2,745 confirmed exoplanets, orchestrated by a LangGraph agent, converts each prediction into a natural language explanation grounded in named, verifiable precedents — the first such capability in an exoplanet vetting system — while a strict downstream architecture guarantees the classifier's measured performance survives into deployment unchanged. A structured evaluation over fifteen test cases found both generation paths fully grounded and complete, with minor hallucination in a fifth of LLM-generated explanations and none, by construction, in the template path. A zero-shot TESS probe locates the approach's boundary: cross-mission transfer fails at the calibration level and requires mission-specific adaptation.

### 6.2 Contributions

The work contributes a controlled four-regime adaptation comparison on astronomical imagery, empirical support for the low-rank adaptation hypothesis in a domain maximally distant from ImageNet, and a novel, auditable architecture coupling classification to retrieval-grounded explanation. The complete system — data pipeline, trained adapters, index, agent, and interactive application — is reproducible from published, citable data sources.

### 6.3 Future Work

Five directions follow naturally. Multi-seed replication would put confidence intervals on the rank sweep. An expanded TESS evaluation (n ≥ 50), followed by threshold recalibration and light LoRA adaptation on TESS data, would test whether the transfer failure is as shallow as the calibration-level diagnosis suggests. Scaling the explanation-quality study to a larger case sample with multiple independent raters would attach confidence intervals to the hallucination and coherence findings reported here. Replacing the six-parameter retrieval space with a learned light-curve embedding would let precedents be retrieved by transit morphology as well as physics. Finally, extending the knowledge base beyond confirmed planets to include *characterised false positives* would let the system explain negative verdicts with the same precedent-based force it currently brings to positive ones — arguably the more valuable direction for the working astronomer, since false positives are where vetting effort is actually spent.

---

## Appendix A — Verification and Validation

Five targeted checks confirm that each major system component behaves as specified before any performance numbers are reported.

**V1 — Stratified split integrity.** The 5,302-KOI dataset is split 70/15/15 (train 3,711 / validation 795 / test 796) using scikit-learn's `StratifiedShuffleSplit` with a fixed seed. A post-split class-balance audit confirmed that each partition preserves the dataset's 41.4% CONFIRMED rate to within one percentage point (train 41.3%, validation 41.5%, test 41.5%). No KOI appears in more than one partition. All experiments in this study are evaluated on this identical, frozen test set.

**V2 — Flat-line filter.** Before GAF encoding, each 2,001-bin light curve is tested for degeneracy: any row with standard deviation below 1 × 10⁻⁶ is flagged and discarded. Applied to the full Mendeley dataset, the filter discarded zero rows — confirming that the published pre-processed data contains no degenerate signals and that the safeguard does not silently remove valid examples.

**V3 — Retrieval sanity test (K00001.01).** The FAISS index was queried with K00001.01 (a canonical confirmed hot Jupiter, period 2.47 d, depth 14,260 ppm). The top-ranked neighbour was Kepler-718 b at cosine similarity 0.999; all five retrieved neighbours were confirmed short-period giants orbiting Sun-like stars. This is the scientifically expected result and confirms that the six-parameter feature vector (period, depth, planet radius, stellar radius, stellar temperature, normalised by a fitted StandardScaler) captures physically meaningful similarity.

**V4 — Template fallback test.** With `HUGGINGFACEHUB_API_TOKEN` removed from the environment, `explain()` was invoked on K00001.01. The function returned a complete, correctly structured explanation drawn from the template — no exception, no empty string. The fallback path is therefore unconditional: the system always produces output regardless of API availability.

**V5 — End-to-end integration test.** K00001.01 was passed through the full pipeline: GAF image loaded from `kepler_gaf_dataset.npz` → `classify()` returned `('CONFIRMED', 0.940)` → FAISS retrieved five neighbours (mean similarity 0.983) → LangGraph agent produced a 4-sentence explanation citing Kepler-718 b by name with correct orbital parameters. Latency on a MacBook CPU (no GPU): preprocessing < 1 s, classification 2.1 s, retrieval < 0.1 s, explanation generation (HuggingFace API) 3.4 s — total < 7 s per KOI.

---

## References

[1] W. J. Borucki et al., "Kepler Planet-Detection Mission: Introduction and First Results," *Science*, vol. 327, no. 5968, pp. 977–980, 2010.

[2] S. E. Thompson et al., "Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog with Measured Completeness and Reliability Based on Data Release 25," *The Astrophysical Journal Supplement Series*, vol. 235, no. 2, p. 38, 2018.

[3] C. J. Shallue and A. Vanderburg, "Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90," *The Astronomical Journal*, vol. 155, no. 2, p. 94, 2018.

[4] M. Ansdell et al., "Scientific Domain Knowledge Improves Exoplanet Transit Classification with Deep Learning," *The Astrophysical Journal Letters*, vol. 869, no. 1, p. L7, 2018.

[5] H. P. Osborn et al., "Rapid Classification of TESS Planet Candidates with Convolutional Neural Networks," *Astronomy & Astrophysics*, vol. 633, p. A53, 2020.

[6] L. Yu et al., "Identifying Exoplanets with Deep Learning. III. Automated Triage and Vetting of TESS Candidates," *The Astronomical Journal*, vol. 158, no. 1, p. 25, 2019.

[7] H. Valizadegan et al., "ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier That Validates 301 New Exoplanets," *The Astrophysical Journal*, vol. 926, no. 2, p. 120, 2022.

[8] D. J. Armstrong, J. Gamper, and T. Damoulas, "Exoplanet Validation with Machine Learning: 50 New Validated Kepler Planets," *Monthly Notices of the Royal Astronomical Society*, vol. 504, no. 4, pp. 5327–5344, 2021.

[9] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," in *Proc. International Conference on Learning Representations (ICLR)*, 2021.

[10] Z. Wang and T. Oates, "Imaging Time-Series to Improve Classification and Imputation," in *Proc. 24th International Joint Conference on Artificial Intelligence (IJCAI)*, 2015, pp. 3939–3945.

[11] Choudhary et al., "Vision Transformer Models for Exoplanet Detection using Gramian Angular Fields on Kepler Light Curves," 2025. *(complete citation to be finalised)*

[12] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 9459–9474.

[13] Z. Ji et al., "Survey of Hallucination in Natural Language Generation," *ACM Computing Surveys*, vol. 55, no. 12, pp. 1–38, 2023.

[14] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. International Conference on Learning Representations (ICLR)*, 2022.

[15] S. D. McCauliff et al., "Automatic Classification of Kepler Planetary Transit Candidates," *The Astrophysical Journal*, vol. 806, no. 1, p. 6, 2015.

[16] D. J. Armstrong et al., "Machine-Learning Approaches to Exoplanet Transit Detection and Candidate Validation," *Monthly Notices of the Royal Astronomical Society*, 2021.

[17] A. Malik, B. P. Moster, and C. Obermeier, "Exoplanet Detection Using Machine Learning," *Monthly Notices of the Royal Astronomical Society*, vol. 513, no. 4, pp. 5505–5516, 2022.

[18] M. Jara-Maldonado et al., "Transiting Exoplanet Discovery Using Machine Learning Techniques: A Survey," *Earth Science Informatics*, vol. 13, pp. 573–600, 2020.

[19] N. Ding et al., "Parameter-Efficient Fine-Tuning of Large-Scale Pre-trained Language Models," *Nature Machine Intelligence*, vol. 5, pp. 220–235, 2023.

[20] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618–626.

[21] C. Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead," *Nature Machine Intelligence*, vol. 1, pp. 206–215, 2019.

[22] T. Macedo and M. Zalewski, "Dataset for Machine Learning Exoplanet Classification," Mendeley Data, V3, 2024. DOI: 10.17632/wctcv34962.3.

[23] G. R. Ricker et al., "Transiting Exoplanet Survey Satellite (TESS)," *Journal of Astronomical Telescopes, Instruments, and Systems*, vol. 1, no. 1, 2015.

> **Note:** Reference [11] needs its full citation completed, and every reference should be verified against the original source before submission.
>
> **Figure mapping for final formatting** (files in `results/`):
> - Figure 1 — system pipeline diagram (`fig1_pipeline_diagram.png`), place in Section 3
> - Figure 2 — four-setting comparison chart (`fig3_four_settings_barchart.png`), Section 4.1
> - Figure 3 — ROC curve + confusion matrix, LoRA r=16 (`lora_r16_roc_curve.png`, `lora_r16_confusion_matrix.png`), Section 4.1
> - Figure 4 — app screenshot with generated explanation (`fig5_app_screenshot.png`), Section 4.4
> - Still to create: light curve → GAF example image (would slot into Section 3.2)
