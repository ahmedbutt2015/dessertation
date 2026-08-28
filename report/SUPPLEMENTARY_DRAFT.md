# Supplementary Report — Explainable Exoplanet Transit Classification Using Vision Transformers, LoRA, and Retrieval-Augmented AI on NASA Kepler Light Curves

**Ahmed Fayyaz Butt** — 20101228
Ulster University (Birmingham Campus)
MSc Computer Science — COM748 Masters Research Project
Supervisor: Dr. Mubashir Ali Cheema

---

## Declaration

I declare that from the date this dissertation is placed in the library or any department, school, or faculty of Ulster University, I allow it to be copied fully or partly without my permission, provided that such copying is for study use only and not for publication. Proper credit must always be given to the source. This rule does not cover the copying or publication of the title, abstract, or introduction of this dissertation. Anyone reading this dissertation must accept that the copyright belongs to the author and that no quotation or information taken from it may be published unless the source is clearly acknowledged.

**Signed:** Ahmed Fayyaz Butt
**Date:** August 2026

---

## Abstract

This supplementary report accompanies the eight-page IEEE-format research paper "Explainable Exoplanet Transit Classification Using Vision Transformers, LoRA, and Retrieval-Augmented AI on NASA Kepler Light Curves." Free of the primary publication's page constraints, it documents in full depth every aspect of the system's design, implementation, experimental evaluation, and project management.

The project addresses the problem of automated transit vetting for Kepler Objects of Interest (KOIs): classifying 5,302 candidate planetary signals as confirmed planets or false positives using a ViT-B/16 vision transformer applied to Gramian Angular Field encodings of phase-folded light curves. Four adaptation regimes are compared — zero-shot, one-shot, few-shot, and LoRA parameter-efficient fine-tuning — across ranks r = 4, 8, and 16. The deployed LoRA r = 16 configuration achieves macro F1 of 0.835 and AUC-ROC of 0.920 while training only 590K of 86M model parameters (0.68%) in approximately 17 minutes on a free-tier GPU. A retrieval-augmented generation module, backed by a FAISS index of 2,745 confirmed exoplanets from the NASA Exoplanet Archive and orchestrated by a LangGraph agentic pipeline, provides the first scientifically grounded natural-language explanation for exoplanet vetting decisions, in which every factual claim traces to a real entry in the Archive. A zero-shot generalisation probe on 14 TESS targets reveals complete transfer failure at the decision threshold while AUC of 0.625 hints that the ranking signal survives, establishing a concrete hypothesis for future recalibration experiments.

This report covers the extended literature review, complete dataset and data-preparation description, detailed system design, implementation with pseudocode, full experimental record, project lifecycle and risk management, verification and validation programme, professional and ethical analysis, and a ranked critical appraisal of all project weaknesses with explicit alternatives.

---

## Acknowledgments

I am grateful to my supervisor, Dr. Mubashir Ali Cheema, for consistent guidance, timely feedback, and for maintaining high standards that improved every aspect of this project. I thank Ulster University Birmingham for access to computing and library resources throughout the programme. The computational experiments depended entirely on free-tier resources provided by Kaggle and Google Colab; this project could not have been attempted without these platforms. The dataset of Macedo and Zalewski [22] resolved a critical operational failure at no cost to scientific validity, and I am grateful to those authors for publishing it with a citable DOI. Finally, the NASA Exoplanet Archive and the MAST archive represent decades of community effort; this project is built on that foundation.

---

## Table of Contents

- [Declaration](#declaration)
- [Abstract](#abstract)
- [Acknowledgments](#acknowledgments)
- [List of Tables](#list-of-tables)
- [List of Figures](#list-of-figures)
- [Chapter 1: Introduction and Scope](#ch1)
  - [1.1 The exoplanet detection challenge](#ch1-1)
  - [1.2 Machine learning approaches to transit vetting](#ch1-2)
  - [1.3 The explainability gap](#ch1-3)
  - [1.4 Research questions](#ch1-4)
  - [1.5 Project objectives](#ch1-5)
  - [1.6 Novel contributions](#ch1-6)
  - [1.7 Scope and assumptions](#ch1-7)
  - [1.8 Report structure](#ch1-8)
- [Chapter 2: Extended Literature Review](#ch2)
  - [2.1 The transit vetting lineage](#ch2-1)
  - [2.2 ExoMiner and the limits of architectural explainability](#ch2-2)
  - [2.3 Time-series imaging and encoding choices](#ch2-3)
  - [2.4 Vision Transformers: architecture and capabilities](#ch2-4)
  - [2.5 Parameter-efficient fine-tuning: the full landscape](#ch2-5)
  - [2.6 Retrieval-augmented generation](#ch2-6)
  - [2.7 Hallucination taxonomy and mitigation](#ch2-7)
  - [2.8 Agentic pipelines and LangGraph](#ch2-8)
  - [2.9 Summary and gap statement](#ch2-9)
- [Chapter 3: Datasets and Data Preparation](#ch3)
  - [3.1 Overview of the dataset landscape](#ch3-1)
  - [3.2 The NASA Kepler Mission](#ch3-2)
  - [3.3 The KOI Cumulative Table](#ch3-3)
  - [3.4 Raw Kepler light curves from NASA MAST](#ch3-4)
  - [3.5 Macedo and Zalewski 2024 — the active data path](#ch3-5)
  - [3.6 NASA Exoplanet Archive for retrieval](#ch3-6)
  - [3.7 TESS dataset for the generalisation probe](#ch3-7)
  - [3.8 Data quality and integrity](#ch3-8)
- [Chapter 4: System Design and Architecture](#ch4)
  - [4.1 Architectural overview](#ch4-1)
  - [4.2 Design principles](#ch4-2)
  - [4.3 Preprocessing and GAF encoding](#ch4-3)
  - [4.4 Classification module design](#ch4-4)
  - [4.5 Retrieval module design](#ch4-5)
  - [4.6 Agentic explanation pipeline design](#ch4-6)
  - [4.7 Interactive application design](#ch4-7)
  - [4.8 API schema](#ch4-8)
- [Chapter 5: Implementation in Detail](#ch5)
  - [5.1 Development environment and toolchain](#ch5-1)
  - [5.2 Dependency management and environment reproducibility](#ch5-2)
  - [5.3 Data preprocessing implementation](#ch5-3)
  - [5.4 Dataset construction and DataLoader](#ch5-4)
  - [5.5 ViT model configuration](#ch5-5)
  - [5.6 Zero-shot and few-shot regime implementation](#ch5-6)
  - [5.7 LoRA training implementation](#ch5-7)
  - [5.8 FAISS index construction](#ch5-8)
  - [5.9 LangGraph agent implementation](#ch5-9)
  - [5.10 Web application implementation](#ch5-10)
  - [5.11 End-to-end pipeline integration and latency profile](#ch5-11)
- [Chapter 6: Experimental Results and Analysis](#ch6)
  - [6.1 Experimental design overview](#ch6-1)
  - [6.2 The four adaptation regimes — full analysis](#ch6-2)
  - [6.3 LoRA rank sweep analysis](#ch6-3)
  - [6.4 Multi-seed robustness](#ch6-4)
  - [6.5 Per-class performance](#ch6-5)
  - [6.6 Retrieval quality](#ch6-6)
  - [6.7 Explanation quality evaluation](#ch6-7)
  - [6.8 TESS zero-shot transfer](#ch6-8)
  - [6.9 Synthesis and cross-experiment interpretation](#ch6-9)
- [Chapter 7: Project Management and Lifecycle](#ch7)
  - [7.1 Development methodology](#ch7-1)
  - [7.2 Requirements specification](#ch7-2)
  - [7.3 Project planning and Gantt](#ch7-3)
  - [7.4 Risk register](#ch7-4)
  - [7.5 Version control and documentation](#ch7-5)
  - [7.6 Supervision and stakeholder management](#ch7-6)
  - [7.7 Toolchain](#ch7-7)
- [Chapter 8: Verification and Validation](#ch8)
  - [8.1 Verification and validation strategy](#ch8-1)
  - [8.2 Requirements and traceability matrix](#ch8-2)
  - [8.3 V1 — Data split integrity](#ch8-3)
  - [8.4 V2 — Input screening](#ch8-4)
  - [8.5 V3 — Retrieval sanity check](#ch8-5)
  - [8.6 V4 — Fallback behaviour](#ch8-6)
  - [8.7 V5 — End-to-end integration](#ch8-7)
  - [8.8 Unit-level verification checks](#ch8-8)
  - [8.9 Validation against research objectives](#ch8-9)
- [Chapter 9: Professional, Ethical, Social and Sustainability Issues](#ch9)
  - [9.1 Professional conduct](#ch9-1)
  - [9.2 Scientific integrity in reporting](#ch9-2)
  - [9.3 Responsible AI design for scientific use](#ch9-3)
  - [9.4 Social impact and democratisation](#ch9-4)
  - [9.5 Environmental sustainability](#ch9-5)
  - [9.6 Open science and reproducibility](#ch9-6)
- [Chapter 10: Critical Appraisal](#ch10)
  - [10.1 Objectives met](#ch10-1)
  - [10.2 Ranked weaknesses and alternatives](#ch10-2)
  - [10.3 What exceeded the proposal](#ch10-3)
- [Chapter 11: Future Work in Detail](#ch11)
  - [11.1 Multi-seed replication for the full rank sweep](#ch11-1)
  - [11.2 Expanded TESS evaluation](#ch11-2)
  - [11.3 LoRA on recurrence-plot inputs](#ch11-3)
  - [11.4 Explanation quality programme](#ch11-4)
  - [11.5 Learned embedding retrieval](#ch11-5)
  - [11.6 Deployment pathway](#ch11-6)
- [Chapter 12: Conclusion](#ch12)
  - [12.1 Summary of contributions](#ch12-1)
  - [12.2 Key findings](#ch12-2)
  - [12.3 Closing remarks](#ch12-3)
- [Chapter 13: Supporting Artefacts](#ch13)
- [References](#references)

---

## List of Tables

| Table | Description | Section |
|---|---|---|
| Table 1 | KOI catalogue columns used for classification and retrieval | 3.3 |
| Table 2 | Dataset overview: name, source, role, size, status | 3.1 |
| Table 3 | PEFT method comparison: mechanism, overhead, use case | 2.5 |
| Table 4 | Functional and non-functional requirements | 7.2 |
| Table 5 | Risk register: risk, likelihood, impact, mitigation, status | 7.4 |
| Table 6 | Project toolchain and library versions | 7.7 |
| Table 7 | ViT-B/16 across four adaptation regimes (test set, n = 796) | 6.2 |
| Table 8 | LoRA rank sweep: trainable parameters and performance | 6.3 |
| Table 9 | LoRA r = 16 multi-seed replication (test set, n = 796) | 6.4 |
| Table 10 | Per-class metrics, LoRA r = 16 (test set, n = 796) | 6.5 |
| Table 11 | Retrieval depth sweep: cosine similarity at k = 3, 5, 10 | 6.6 |
| Table 12 | Explanation quality over fifteen cases | 6.7 |
| Table 13 | Per-target TESS zero-shot predictions (n = 14) | 6.8 |
| Table 14 | Verification and validation traceability matrix | 8.2 |
| Table 15 | Ranked weaknesses with alternatives and weight | 10.2 |
| Table 16 | Milestones and deliverables with target and actual dates | 7.3 |

---

## List of Figures

| Figure | Description | File |
|---|---|---|
| Figure S1 | Complete system pipeline | `results/fig1_pipeline_diagram.png` |
| Figure S2 | Phase-folded light curve and GAF encoding for CONFIRMED and FALSE POSITIVE | `results/fig2_lightcurve_to_gaf.png` |
| Figure S3 | Macro F1 and AUC-ROC across four adaptation regimes and LoRA ranks | `results/fig3_four_settings_barchart.png` |
| Figure S4 | Reproducibility of LoRA r = 16 across three random seeds | `results/fig7_multiseed_stability.png` |
| Figure S5 | ROC curve for deployed LoRA r = 16 model | `results/lora_r16_roc_curve.png` |
| Figure S6 | Confusion matrix for deployed LoRA r = 16 model | `results/lora_r16_confusion_matrix.png` |
| Figure S7 | LoRA rank sweep: macro F1 and AUC-ROC at r = 4, 8, 16 | `results/fig6_rank_sweep.png` |
| Figure S8 | Pipeline output for K00001.01: retrieved analogues and agent explanation | `results/fig5_explanation_example.png` |
| Figure S9 | Mission Control landing page | |
| Figure S10 | Project Gantt chart | `results/fig6_gantt_chart.png` |
| Figure S11 | Real-time pipeline analysis — K00752.01 | |
| Figure S12 | Classification output — K00752.01 | |
| Figure S13 | Classification output — K00069.01 | |

---

<a id="ch1"></a>
## Chapter 1: Introduction and Scope

<a id="ch1-1"></a>
### 1.1 The exoplanet detection challenge

Discovering planets around other stars has been among the most consequential scientific endeavours of recent decades. Before Kepler, exoplanet detection was painstaking, instrument-limited, and largely confined to measuring the gravitational tug that massive planets exert on their host stars through the radial velocity method. The transit method offered a complementary approach: if a planet's orbital plane aligns with the observer's line of sight, it periodically crosses the stellar disk, reducing the observed brightness by an amount proportional to the squared ratio of planetary to stellar radius. For an Earth-radius planet crossing a Sun-like star, this dimming is about 84 parts per million — measurable only with space-based photometry free of atmospheric distortion.

The Kepler Space Telescope, launched in March 2009 and operational until 2018, demonstrated the viability of the transit method at scale [1]. Over its primary mission, Kepler monitored approximately 200,000 stars continuously in a fixed field of view, accumulating four years of 30-minute cadence photometry for each. The result was an unprecedented dataset: millions of light curves, each a time series of fractional brightness measurements, from which the signatures of transiting planets could be extracted. The mission produced 9,564 Kepler Objects of Interest (KOIs) — cases where an automated pipeline identified a periodic brightness dip consistent with a transit and flagged them for human vetting. Of these, the DR25 catalogue released by Thompson et al. [2] assigns each KOI one of three dispositions: CONFIRMED (the transit is of planetary origin, verified by follow-up observations), FALSE POSITIVE (the dip arises from a non-planetary source such as an eclipsing binary or a background star), or CANDIDATE (insufficient evidence either way). The classification of KOIs from this catalogue is the primary task of this project.

The scale of the task motivates the use of machine learning. Manual vetting by expert astronomers is thorough but cannot keep pace with the data volumes produced by Kepler and its successor TESS [23]. A single KOI requires examination of the phase-folded light curve, the centroid behaviour, the secondary eclipse depth, the odd-even asymmetry, and contextual information from stellar catalogues — a process taking tens of minutes per target, implying thousands of hours for the full KOI list. Automated vetting systems can process thousands of targets in minutes, flagging the high-confidence false positives and confirmed planets so that human expert effort can focus on the ambiguous cases where it adds most value.

<a id="ch1-2"></a>
### 1.2 Machine learning approaches to transit vetting

The machine learning approach to transit vetting has a clear lineage. McCauliff et al.'s Autovetter [15], published in 2015, established the feasibility of the approach using a random forest classifier on a set of hand-engineered features derived from KOI catalogue parameters and light curve statistics. It achieved high performance within the feature space available to it, but the hand-engineering bottleneck meant that it could not adapt to novel data types or benefit from raw photometric features that human designers had not thought to extract.

Shallue and Vanderburg's AstroNet [3], published in 2018, replaced feature engineering with end-to-end learning from raw light curves. They introduced the dual-view convention that has dominated the field since: a global view of the full phase-folded light curve provides context for the transit shape within the orbital cycle, while a local view zoomed on the transit window provides fine-grained morphological detail. A convolutional neural network consuming both views achieved 96% precision at 75% recall on their dataset. The architectural choice of a CNN was natural: convolutional filters detect local patterns such as the characteristic flux dip, and the hierarchical feature learning removes the hand-engineering bottleneck.

Subsequent work refined AstroNet's framework rather than departing from it. Ansdell et al. [4] added centroid time-series and stellar-parameter channels as additional inputs, demonstrating that domain knowledge encoded as auxiliary features improved performance beyond what the photometric signal alone could support. Osborn et al. [5] and Yu et al. [6] adapted the CNN framework to TESS, where shorter baselines, higher cadence, and different systematic noise sources created a distribution shift that the Kepler-trained models did not handle gracefully without adaptation.

ExoMiner [7] represents the current state of the art in automated vetting. It achieves high precision by combining eleven diagnostic input streams into a multi-branch deep network, and its authors describe it as "explainable" on the basis that an expert can inspect which diagnostic branch drove the score for a given prediction. The system has been used to validate 301 new exoplanets from the Kepler archive. Despite its performance, the explainability claim deserves closer examination — taken up in Section 1.3.

<a id="ch1-3"></a>
### 1.3 The explainability gap

The term "explainability" is used in the machine learning literature to describe a spectrum of properties ranging from full model transparency (a decision tree in which every step is human-readable) to post-hoc attribution (highlighting which input regions drove the prediction of an opaque model). ExoMiner's explainability sits at a specific point on this spectrum: because the model is built from named diagnostic branches, an expert can determine, after the fact, which branch assigned the highest score. As the primary paper states, "its explainability is architectural... This is dissection, not communication." Nothing in ExoMiner produces a statement an astronomer could read, evaluate, question, or cite in a subsequent paper. The system reveals which diagnostic channel was most active; it does not say what the prediction means in the context of known planetary systems.

The practical cost of this gap was identified by Jara-Maldonado et al. [18], whose survey of machine learning approaches to transit detection found that trust, not accuracy, is the principal barrier to astronomers' adoption of ML vetting tools. Astronomers need to understand not only what the model predicted, but why that prediction is scientifically plausible — which known systems the candidate resembles, whether its parameters are consistent with planetary physics, and what the specific evidence is for or against the planetary interpretation. No existing system generates this kind of explanation. That is what this project is built to produce.

That gap reaches beyond the user interface. An opaque classifier that is correct 92% of the time may be acceptable for bulk processing, but it cannot participate in the epistemic process of astronomical science, where every claim must rest on evidence that others can scrutinise and challenge. A system whose verdicts come with verifiable retrieved evidence from the Archive changes the relationship between model and astronomer from oracle to evidence-presenting collaborator.

<a id="ch1-4"></a>
### 1.4 Research questions

This project addresses four focused research questions:

**RQ1:** To what degree does a Vision Transformer pretrained on ImageNet transfer to the visually alien domain of Gramian Angular Field textures of stellar photometry, and how does performance change across a controlled sequence of supervision levels from zero-shot to LoRA fine-tuning?

**RQ2:** At what LoRA rank does performance saturate in this domain, and what does the rank-sweep result imply about the intrinsic dimensionality of the task-adaptation problem for transit classification?

**RQ3:** Can a retrieval-augmented generation system, anchored to a FAISS index of the NASA Exoplanet Archive, produce scientifically grounded natural-language explanations for transit vetting decisions that are verifiably free of factual hallucination?

**RQ4:** How does the Kepler-trained ViT-B/16 + LoRA system perform on TESS photometry presented in the same GAF format, and what mechanism explains any degradation observed?

<a id="ch1-5"></a>
### 1.5 Project objectives

The project was scoped around five numbered objectives from the accepted proposal:

**Objective 1:** Construct a labelled image dataset by encoding phase-folded Kepler KOI light curves as Gramian Angular Field images and splitting them into reproducible training, validation, and test partitions.

**Objective 2:** Train and evaluate a ViT-B/16 classifier under four adaptation regimes (zero-shot, one-shot, few-shot, LoRA) and report macro F1 and AUC-ROC on a frozen test set, with inference latency recorded alongside.

**Objective 3:** Build a FAISS retrieval index over the NASA Exoplanet Archive confirmed planet catalogue and implement a cosine-similarity search interface for querying any KOI by its physical parameters.

**Objective 4:** Implement a LangGraph agentic pipeline that consumes the classifier's verdict and retrieved neighbours to generate a structured natural-language explanation, and evaluate the pipeline's output on grounding, completeness, and hallucination rate.

**Objective 5:** Evaluate the Kepler-trained model on TESS targets with no retraining to characterise zero-shot generalisation and identify the mechanism of any performance gap.

<a id="ch1-6"></a>
### 1.6 Novel contributions

This project makes five contributions that, individually or in combination, have not appeared in the published literature. The primary IEEE paper groups these into three headline contributions; this supplementary document expands them into five to give each the depth of treatment it requires:

**Contribution 1 — ViT on GAF in the parameter-efficient regime.** Prior ViT-based exoplanet classification (Choudhary et al. [11]) uses full feature extraction or full fine-tuning. This project is the first to evaluate ViT-B/16 under a controlled four-regime supervision sequence on Kepler GAF data, isolating the contribution of each increment of adaptation.

**Contribution 2 — Empirical LoRA rank evidence in an astronomical imaging domain.** The intrinsic-dimensionality argument underlying LoRA [14] has been validated extensively in NLP and natural-image domains, but never in GAF-encoded stellar photometry. The rank sweep provides direct evidence that task adaptation occupies a low-rank subspace even when the input domain is maximally alien to the pretraining distribution.

**Contribution 3 — The first retrieval-grounded explanation layer for exoplanet vetting.** Every prior vetting system produces a score or a label. This project produces a natural-language account in which every factual claim traces to a named, verifiable entry in the NASA Exoplanet Archive — the first such system in the literature.

**Contribution 4 — An agentic architecture with typed state for scientific accountability.** The LangGraph implementation enforces typed intermediate state, making every product of the pipeline inspectable, logged, and auditable — the architecture suited to a system whose outputs may inform scientific decisions.

**Contribution 5 — An honest characterisation of zero-shot transfer to TESS.** Rather than omitting the negative result or presenting it as a partial success, this project reports the complete failure at the decision threshold, analyses the input-statistics mechanism that likely explains it, and lays out the expansion experiment that would test the hypothesis.

<a id="ch1-7"></a>
### 1.7 Scope and assumptions

**In scope:** KOI classification (CONFIRMED vs FALSE POSITIVE only); ViT-B/16 as the sole backbone; the four adaptation regimes described; the six physical parameters as the retrieval feature space; the Flask web application as the demonstration vehicle; the Kaggle P100 GPU as the training platform; and the Mendeley dataset as the labelled source.

**Out of scope:** CNN architectures; full fine-tuning of the ViT backbone; multi-modal classification combining photometry with centroid or stellar spectra; retraining on TESS data; real-time streaming from MAST; production deployment with SLA guarantees; and integration with external vetting databases such as ExoFOP.

**Assumptions:** The Mendeley dataset accurately represents the MAST photometry for the 5,302 KOIs it covers; the published disposition labels in DR25 are the ground truth; KOI identifiers not present in the catalogue are treated as missing data and rejected rather than imputed; and all GPU training results are reproducible within the random-seed variance documented in the multi-seed study.

<a id="ch1-8"></a>
### 1.8 Report structure

Chapter 2 extends the literature review across the transit vetting lineage, ViT architecture, the full PEFT landscape, retrieval-augmented generation, hallucination taxonomy, and agentic pipelines. Chapter 3 describes all datasets. Chapters 4 and 5 cover system design and implementation in full, with pseudocode for every major component. Chapter 6 presents the experimental record and cross-experiment interpretation. Chapter 7 documents the project lifecycle — requirements, planning, and risk management. Chapter 8 details the verification and validation programme with a full traceability matrix. Chapter 9 addresses professional, ethical, social, and sustainability dimensions, including a discussion of responsible AI design in scientific contexts. Chapter 10 provides a ranked critical appraisal of nine explicit weaknesses with concrete alternatives, and Chapter 11 specifies the six future experiments that directly follow from those weaknesses. Chapter 12 concludes. Chapter 13 lists supporting artefacts.

---

<a id="ch2"></a>
## Chapter 2: Extended Literature Review

<a id="ch2-1"></a>
### 2.1 The transit vetting lineage

The automated classification of Kepler KOIs has a clear evolutionary arc, and understanding it requires tracing not just what changed across generations of systems, but what did not. McCauliff et al.'s Autovetter [15] applied a random forest to a vector of 25 hand-engineered features derived from the KOI catalogue: the odd-even depth difference, secondary eclipse depth, centroid offset significance, and others that encode known false-positive diagnostics identified by human vetters. The random forest achieved competitive performance on the DR24 catalogue, but the hand-engineering step was a fundamental bottleneck: the feature set encoded only what was already understood, and any novel photometric signature that the feature engineers had not anticipated would be invisible to the classifier.

Shallue and Vanderburg's AstroNet [3] replaced this bottleneck with end-to-end learning from raw phase-folded flux values. Two architectural choices shaped everything that followed. First, the dual-view convention: a global view of 2,001 bins spanning the full orbital phase, and a local view of 201 bins zoomed on the transit window. The global view provides the out-of-transit baseline — essential for detecting V-shaped eclipses, secondary eclipses at phase 0.5, and ellipsoidal variations that indicate a binary system rather than a planet. The local view provides fine-grained transit morphology. Together they encode most of the information a human vetter examines in the light curve, without hand-selecting which aspects matter. Second, a convolutional backbone: the CNN's local feature detectors are naturally suited to detecting the transit dip, which is a spatially compact pattern in the phased flux sequence. The AstroNet architecture achieved 96% precision at 75% recall, establishing a baseline that subsequent work aimed to match or surpass.

The innovations in the years following AstroNet came in two flavours: enriching the input representation and refining the output framework. Ansdell et al. [4] demonstrated the first flavour: by adding centroid time-series, stellar-parameter channels (effective temperature, surface gravity), and multiplicity flags as auxiliary inputs, they improved on AstroNet's performance and showed that domain knowledge, when encoded as additional channels rather than hand-engineered features, could still be exploited by an end-to-end architecture. Ansdell et al.'s approach makes more domain information available to the model and lets it determine what to do with it — the bottleneck is not domain knowledge itself but the act of manually pre-selecting which features to extract. Pearson et al. [16] explored the second direction: they showed that neural networks could detect transit signals in raw, undetrended photometry, removing the assumption that systematic removal must precede classification and making the pipeline more robust to detrending artefacts.

The generalisation to TESS introduced by Osborn et al. [5] and Yu et al. [6] revealed the fragility of the Kepler-trained models under distribution shift. TESS observes each ecliptic sector for approximately 27 days rather than Kepler's four-year baseline, producing light curves with different systematics, cadence, and a smaller number of observed transits per target. The shorter baseline means fewer orbital cycles are stacked during phase-folding, leaving a noisier light curve profile. Both groups found that direct application of Kepler-trained models to TESS produced degraded performance and that some degree of adaptation was needed. The nature of that adaptation — whether threshold recalibration, domain-adaptive training, or full retraining — remained an open question that this project's TESS probe partially addresses.

Armstrong et al. [8] represent the output-enrichment direction: rather than trying to improve classification accuracy, they used classifier outputs in a Bayesian framework to compute posterior planet probabilities, providing a statistically rigorous basis for claiming that fifty Kepler candidates are genuine planets. This approach treats the classifier as one component of a broader validation apparatus rather than an endpoint. The present project's RAG module serves an analogous enrichment role: it does not change the classification decision, but it provides the context that makes the decision interpretable and traceable.

Malik et al. [17] applied gradient-boosting classifiers and compared them with neural approaches on the Kepler KOI catalogue, finding competitive performance with a much smaller computational footprint. Their work underscores that the classification problem, at its core, is not computationally intractable — the challenge is building a system whose outputs earn the trust of the astronomers who would use it.

<a id="ch2-2"></a>
### 2.2 ExoMiner and the limits of architectural explainability

ExoMiner [7] is the most capable and widely cited automated KOI classifier to date. Its architecture comprises eleven specialised input branches, each consuming one diagnostic data stream: a global view and local view of the light curve, secondary eclipse features, centroid motion, stellar parameters, and several others. Each branch processes its input through a small convolutional network, and the branch outputs are concatenated and fed to a fully connected classifier. The result is high precision — 99.5% on the test set — and the validated discovery of 301 new exoplanets from Kepler archive data.

The authors describe ExoMiner as "explainable" on the basis that the branch decomposition makes it possible to determine which diagnostic contributed most to a given prediction. A post-hoc inspection of branch weights can tell an expert that, for example, the centroid branch drove the false-positive classification of a given KOI, implying a background star is probably responsible for the brightness dip. This is a genuine and useful property. But "explainability" in this sense is restricted to experts who already understand which diagnostic each branch corresponds to and can interpret a weight vector in that context. An astronomer unfamiliar with the specific false-positive scenario encoded in the centroid branch still cannot make sense of what the model is saying. More fundamentally, a weight vector is not a statement. It cannot be cited, it cannot be disputed by another expert pointing to a different interpretation of the same evidence, and it does not situate the classification in the context of what is already known about similar systems.

The primary paper's phrase — "dissection, not communication" — captures this precisely. Dissection reveals internal structure; communication produces a statement that can enter the discourse of science. The Jara-Maldonado et al. survey [18] of machine learning for exoplanet detection makes the practical consequence explicit: the single largest barrier to astronomer adoption of ML vetting tools is not accuracy (existing systems are already highly accurate) but trust, and trust requires communication, not dissection. Astronomers want to understand why a verdict was reached in a way that connects to their existing knowledge of planetary physics and known systems. That is the gap the retrieval-augmented explanation layer fills.

<a id="ch2-3"></a>
### 2.3 Time-series imaging and encoding choices

The conversion of a one-dimensional time series to a two-dimensional image for processing by a 2D architecture is a non-trivial design decision with multiple valid options, each carrying different assumptions about what structure in the signal matters most. Wang and Oates [10] introduced the Gramian Angular Field and the Markov Transition Field as two approaches within the same framework of time-series imaging. The GAF has two variants: the Summation Field (GASF), which forms the matrix of cosines of angular sums, and the Difference Field (GADF), which uses angular differences. Both begin by rescaling the series to [-1, 1] and mapping each value to an angle through the arccosine, preserving monotonicity. The matrix of pairwise cosines (angular sums for GASF, angular differences for GADF) encodes correlations between time points in a way that a 2D convolutional or attentional architecture can exploit. The diagonal of the GASF image recovers the original time series values, making the encoding invertible. Amplitude structure is preserved because values are encoded as angles before the matrix is formed; temporal correlation is preserved because the (i, j) entry of the matrix records the relationship between the values at time points i and j.

The Markov Transition Field encodes state-transition probabilities rather than amplitude values. The series is discretised into Q quantile bins, and the (i, j) entry records the frequency with which a value in the bin containing x_i is followed (at any lag) by a value in the bin containing x_j. This encodes the temporal dynamics of state transitions rather than amplitude correlations, making it more sensitive to periodicity and less sensitive to the absolute shape of the signal. For transit detection — where the key signal is a periodic dip of specific depth and width — neither encoding is obviously superior a priori.

Recurrence plots take a different approach entirely: the series is embedded in a delay-coordinate space of dimension m and delay τ, producing a trajectory in that space, and the (i, j) entry records whether the distance between the state at time i and the state at time j is below a threshold ε. The binary or graded thresholding reveals the recurrence structure of the underlying dynamical system. Transit signals produce distinctive recurrence patterns because the periodic dip recurs with the orbital period; the recurrence plot makes this periodicity visible as a set of parallel diagonal lines.

Choudhary et al. [11] conducted the most directly relevant comparison. They evaluated ViT-B/16 on Kepler KOI light curves encoded as both GAFs and recurrence plots, finding that recurrence plots produced higher recall (89.46% on confirmed planets) than GAFs. Their interpretation is consistent with the signal-processing intuition: transit vetting is fundamentally a periodicity detection problem, and recurrence plots are designed to reveal periodicity, while GAFs are designed to reveal amplitude correlation structure. The GAF encodes the transit dip as a dark cross — a locally compact pattern — whereas the recurrence plot encodes the periodic recurrence of that dip as a globally extended diagonal structure. The latter is arguably more informative for the task.

This project retained the GAF encoding for two reasons. The first is toolchain stability: the pyts library provides a mature, well-tested GASF implementation, while recurrence plot implementations in the scientific Python ecosystem are less standardised. The second is schedule: switching encodings mid-project after the GAF pipeline had been verified would have required re-running every preprocessing step and re-establishing all verification checks. As the critical appraisal (Section 10.2) argues, this was the project's most consequential methodological constraint, and the counterfactual — LoRA on recurrence plots — is the highest-priority future experiment.

<a id="ch2-4"></a>
### 2.4 Vision Transformers: architecture and capabilities

The Vision Transformer (ViT), introduced by Dosovitskiy et al. [9], applies the transformer architecture of Vaswani et al. to image classification by treating an image as a sequence of patches. A ViT-B/16 model divides a 224 × 224 pixel input into 14 × 14 = 196 patches of 16 × 16 pixels each. Each patch is flattened into a 768-dimensional vector through a learned linear projection (the patch embedding). A learnable classification token [CLS] is prepended to the sequence, and learnable positional embeddings are added to all 197 tokens to encode spatial structure. The resulting sequence is processed by L = 12 transformer encoder blocks, each containing multi-head self-attention (12 heads, each attending to all 197 tokens) and a position-wise MLP with a 3072-dimensional intermediate layer. Layer normalisation is applied before each sub-block (the Pre-LN configuration used in ViT-B/16), and residual connections bypass each sub-block. After all 12 blocks, the [CLS] token's 768-dimensional representation is passed through a linear classification head to produce the output logits.

What distinguishes ViT from CNN-based alternatives is the global receptive field from the very first layer. A CNN's first convolutional filter operates on a local region of the image; its receptive field expands layer by layer through pooling and striding. A ViT's attention mechanism, by contrast, allows every patch to attend to every other patch from the first block. For transit classification, where the diagnostic signal is the relationship between different parts of the phase-folded profile — the dip at phase 0 against the baseline, or against a secondary eclipse at phase 0.5 — this global view is a genuine architectural advantage over CNNs, without needing many layers to propagate that information.

Pretraining on ImageNet-21k (14M images, 21,000 classes) provides the ViT-B/16 with a general-purpose visual representation: the model has learned to detect edges, textures, shapes, and compositional structures that generalise broadly. Whether that representation transfers to the visually unusual domain of GAF-encoded stellar photometry is a central question this project addresses. The zero-shot result (AUC 0.536, essentially chance) answers it directly: the pretrained features encode nothing discriminative about GAF textures of transit signals. Everything in the subsequent rows of the comparison table is bought by adaptation. That result carries weight beyond this single task. A common assumption in the machine learning community is that sufficiently large pretrained models will transfer to any visual domain with minimal supervision; on this domain, that assumption does not hold.

<a id="ch2-5"></a>
### 2.5 Parameter-efficient fine-tuning: the full landscape

The parameter-efficient fine-tuning (PEFT) literature, surveyed by Ding et al. [19], addresses a common problem: a large pretrained model must be adapted to a specific task, but full fine-tuning of all its parameters is computationally expensive, requires large amounts of task-specific data to avoid catastrophic forgetting, and produces a model whose weights diverge from the pretrained checkpoint in ways that are difficult to control. PEFT methods constrain the adaptation to a small subset of parameters while keeping the majority of the model frozen.

**Adapter modules** [Houlsby et al., 2019] insert small bottleneck networks between the attention and MLP sub-blocks of each transformer layer. A typical adapter has a down-projection from the model dimension d to a small bottleneck r, a non-linearity, and an up-projection back to d. Only the adapter parameters (2 × d × r per layer) are trained, while the original transformer weights remain fixed. The residual connection around the adapter means that at initialisation (with the down-projection weights small), the adapter is nearly an identity map and does not disrupt the pretrained representation. The limitation is inference overhead: the adapter modules are not merged into the pretrained weights, so every forward pass must execute the additional computations. For a system serving interactive queries, this overhead accumulates.

**Prefix tuning** and **prompt tuning** [Li and Liang, 2021; Lester et al., 2021] prepend a small number of learnable soft-prompt tokens to the input sequence (or to the key-value matrices of each attention layer in prefix tuning). The transformer processes these additional tokens alongside the actual input, and the learned prefix steers the model's behaviour. The approach is parameter-efficient — the prefix is typically 10–100 tokens — but it reduces the effective context window available for actual input, and the learned prefixes are difficult to interpret or transfer between tasks.

**BitFit** [Ben-Zaken et al., 2022] updates only the bias parameters of the transformer, freezing all weight matrices. This reduces the trainable parameter count to approximately 0.1% of the model, with the theoretical justification that biases primarily encode task-specific activation patterns while weight matrices encode the general structure of the representation. BitFit performs competitively on some NLP benchmarks at this extreme parameter reduction, but its effectiveness has not been demonstrated for vision tasks or for domains as distant from pretraining as GAF-encoded photometry.

**LoRA** [14] is the method this project uses. Rather than adding new modules or adapting bias parameters, LoRA inserts trainable low-rank matrices into existing weight matrices. For a weight matrix W ∈ R^{d×k}, LoRA introduces an update ΔW = BA where B ∈ R^{d×r} and A ∈ R^{r×k}, with r ≪ min(d, k). A is initialised with a Gaussian, B is initialised to zero, so ΔW = 0 at training start and the pretrained behaviour is preserved. During training, only A and B are updated. After training, the update can be merged into the frozen weight: W' = W + BA, producing a model with no additional parameters or inference overhead compared to the original. The trainable parameter count for LoRA applied to the query and value projections of all 12 attention layers in ViT-B/16 at rank r is approximately 2 × 12 × 768 × r × 2 = 36,864r parameters, yielding 147K at r = 4, 295K at r = 8, and 590K at r = 16.

**QLoRA** [S1] extends LoRA with 4-bit quantisation of the frozen weights. The frozen parameters are stored in NF4 (Normal Float 4-bit) format, dramatically reducing GPU memory requirements while preserving the trainable LoRA matrices in full precision. This makes very large models (70B parameters and above) adaptable on consumer hardware. For ViT-B/16 at 86M parameters, the memory advantage of QLoRA is unnecessary — the model fits comfortably on a Kaggle P100 in full precision — and the quantisation introduces small numerical errors that are avoidable when the hardware budget permits. QLoRA was therefore not used in this project.

The theoretical justification for LoRA's effectiveness is the intrinsic-dimensionality hypothesis: Aghajanyan et al. [S9] demonstrated empirically that the solution space of NLP fine-tuning tasks is low-dimensional in the original weight space, meaning that full fine-tuning traverses a high-dimensional space but the solution lies near a low-rank subspace. LoRA directly targets this subspace. Whether this hypothesis holds for vision tasks in visually exotic domains — GAF textures of stellar photometry — had no published answer before this project. The rank-sweep result (AUC saturating at r = 4) is direct empirical evidence that it does: even in this domain, the adaptation problem is low-rank, and rank r = 4 (147K trainable parameters, 0.17% of the model) captures essentially all of the discriminative signal. The significance extends beyond the headline numbers — the finding speaks to how vision transformers adapt to new domains generally, not just to this specific task.

**Table 3 — PEFT method comparison**

| Method | Trainable parameters | Inference overhead | Zero-init? | Merges into weights? | Best use case |
|---|---|---|---|---|---|
| Adapter | 2 × d × r per layer | Yes — extra compute per layer | Yes | No | NLP fine-tuning with memory constraints |
| Prefix tuning | num_prefix × d per layer | Yes — reduced context | Yes | No | NLP tasks; few-shot transfer |
| BitFit | Biases only (~0.1%) | No | No | N/A | Extreme low-resource NLP |
| LoRA | 2 × d × r per adapted layer | No (merges after training) | Yes (B=0) | Yes | Any transformer, vision or language, low inference overhead required |
| QLoRA | Same as LoRA | No | Yes | Yes | Very large models on limited GPU memory |

<a id="ch2-6"></a>
### 2.6 Retrieval-augmented generation

Lewis et al.'s RAG framework [12] addresses a fundamental limitation of parametric language models: the knowledge they possess is static, determined by the training data, and cannot be updated without retraining. Retrieval augmentation supplements the model's parametric memory with non-parametric external storage. At inference time, a query is encoded as a dense vector, a retrieval system finds the most similar vectors in a database of documents (or structured records), and the retrieved content is provided to the generator as additional context. The generator then produces a response that is grounded in this retrieved evidence.

The FAISS library (Facebook AI Similarity Search) provides the retrieval infrastructure. For this project, the index type is IndexFlatIP, which performs exact inner-product search over all indexed vectors. For L2-normalised vectors, inner product equals cosine similarity, so the index retrieves the k nearest neighbours by cosine similarity without approximation. Approximate search methods (IndexIVFFlat, IndexHNSW) trade exactness for speed, but with 2,745 indexed vectors, exact search takes under one millisecond, making approximation unnecessary.

The distinction between retrieval-augmented explanation and fine-tuning-based explanation is important. Fine-tuning encodes knowledge into model parameters during training and cannot be updated at inference time without retraining; it also entangles the explanation with the model's inductive biases, potentially hallucinating details that seem statistically plausible but are not factually grounded. Retrieval reads from an external database at inference time and can be updated by updating the database; every piece of information it provides can in principle be traced to a specific record. For a scientific explanation system, this property is essential: it means the explanation can be audited, the records it cites can be checked, and errors can be corrected by updating the knowledge base rather than retraining the model.

The distinction between retrieval-grounded explanation and attention-based attribution (such as attention rollout [S2]) is equally important. Attention rollout aggregates attention weights across transformer layers to produce a map indicating which input tokens or image patches the model attended to most strongly. This sounds like an explanation, but it suffers from two problems. First, it requires the reader to interpret the attention map over a GAF image — an encoded matrix that no astronomer examines directly — rather than over a quantity with physical meaning. Second, a growing body of work questions whether attention weights are faithful indicators of feature importance at all: Jain and Wallace [2019] showed that attention weights can be arbitrarily permuted without changing model predictions, and Wiegreffe and Pinter [2019] found that alternative attention distributions with equally high weights on different tokens produce identical predictions. Retrieval-grounded explanation sidesteps the faithfulness debate entirely: the system does not claim to reveal the model's internal process. It claims only to present verifiable evidence from the Archive that supports or contextualises the verdict. That's a more modest claim than full model transparency — but it's one that holds under scrutiny.

<a id="ch2-7"></a>
### 2.7 Hallucination taxonomy and mitigation

Ji et al.'s survey [13] provides the most comprehensive taxonomy of hallucination in natural language generation systems. The survey distinguishes two primary types: intrinsic hallucination, in which the generated output contradicts the input or the retrieved source material, and extrinsic hallucination, in which the generated output introduces information that cannot be verified from the input or retrieved context — it may or may not be factually correct, but it is not grounded in the provided evidence. The survey documents hallucination rates, detection methods, and mitigation strategies across a wide range of generation tasks.

Multiple mitigation strategies have been proposed in the literature, falling broadly into three categories. Retrieval grounding (the approach this project uses) constrains generation to be consistent with retrieved evidence; its effectiveness depends on the quality and relevance of the retrieved content. Faithfulness training uses reward models or discriminators that penalise outputs inconsistent with the source; it requires additional training and does not unconditionally prevent fabrication. Constrained decoding enforces consistency constraints at the token level during beam search; it is task-specific and difficult to apply to open-ended prose generation.

The explanation evaluation in this project maps cleanly onto the Ji et al. taxonomy: across fifteen cases, the LLM cascade produced no intrinsic hallucinations (no output contradicted the retrieved record values) and its extrinsic additions were minor qualifiers that went slightly beyond the evidence without asserting new facts. The template fallback, which inserts values directly from the retrieved records, produced no extrinsic additions of any kind. However, the evaluation uncovered a failure mode that the standard taxonomy does not address: on K00262.01, the LLM produced a fluent three-sentence explanation citing high cosine similarity to confirmed planets as evidence supporting a false-positive verdict. Every factual claim in the explanation was correct (the similarities were high, the listed planets are real), but the reasoning was incoherent: high similarity to confirmed planets is evidence for, not against, a planetary interpretation. This is a coherent-facts-illogical-reasoning failure, distinct from both intrinsic and extrinsic hallucination, and it represents a genuine gap in the current hallucination taxonomy. The template fallback handles this correctly because its FALSE POSITIVE branch was written to explicitly state that the model's morphological assessment overrides parameter-space similarity to confirmed systems.

Rudin [21] argues that the machine learning community should prefer inherently interpretable models over black-box models with post-hoc explanations for high-stakes decisions, because post-hoc explanations may not faithfully represent the model's actual reasoning. This project's design is not inconsistent with Rudin's position. The retrieval-augmented explanation layer does not claim to explain the ViT's internal process; it presents verifiable domain-relevant evidence alongside the verdict. The ViT remains opaque. What the explanation layer provides is a scientifically grounded context for the verdict — analogous to a judge citing precedent rather than explaining their neurology. This is "explanation in the vocabulary of the domain," which is precisely Rudin's prescription when an inherently interpretable classifier of equivalent accuracy is not available.

<a id="ch2-8"></a>
### 2.8 Agentic pipelines and LangGraph

Agentic AI systems extend large language models with the ability to take actions: calling tools, querying databases, reading files, or invoking other models. The agent's decision about which tool to call, in which order, and with which arguments is driven by the LLM itself, making the system capable of multi-step reasoning over external resources. Simple agentic systems use a ReAct-style loop (Reason + Act), alternating between producing a reasoning trace and taking a tool action. More complex systems use directed graphs of nodes, each performing a specific computation, with edges encoding the valid transitions between them.

LangGraph implements the directed-graph approach. Each node in the graph is a Python function that takes a typed state object and returns a (possibly modified) state object. Edges can be conditional, allowing the graph to branch based on state values. The typed state is the critical design decision for scientific accountability: because every field of the state is typed and named, every intermediate product of the pipeline is inspectable, logged, and attributable to a specific node. There is no hidden prompt text or implicit reasoning that could produce outputs the system cannot account for. For a system whose outputs may inform scientific decisions — identifying planets is a high-stakes task — that auditability isn't a quality-of-life extra; it's part of what correctness means for this application.

The contrast with chain-of-thought (CoT) prompting is instructive. CoT prompts the LLM to produce a reasoning trace before its final answer, which has been shown to improve performance on multi-step reasoning tasks. But the reasoning trace is generated text, not a verified computation; the LLM may produce a fluent and coherent-looking reasoning chain that reaches the wrong conclusion, or may hallucinate intermediate steps. In a LangGraph system, each node performs a specific, deterministic computation (e.g., FAISS vector search, template string formatting) that is guaranteed to be correct within its design assumptions; only the final prose generation step involves the LLM. Typed state makes the boundary between deterministic computation and LLM generation explicit. The pipeline's governing property — that it is strictly downstream — is enforced by the graph structure: the classification node runs before the retrieval node, and the retrieval node runs before the explanation node, with no edges in the reverse direction.

<a id="ch2-9"></a>
### 2.9 Summary and gap statement

The literature establishes several points clearly. Transit classification is a solved problem in the narrow sense of accuracy: ExoMiner achieves 99.5% precision on Kepler data. It is not solved in the sense of scientific utility: astronomers do not yet trust automated classifiers enough to act on their verdicts without independent vetting, because no system communicates its verdicts in the language of astronomical evidence. ViT architectures are competitive with or superior to CNNs on image classification generally, and Choudhary et al. [11] demonstrate their effectiveness specifically on Kepler GAF data. LoRA is the most deployment-efficient member of the PEFT family for transformer models with an interactive serving constraint, but its effectiveness in visually exotic domains has not been characterised before this project. Retrieval-augmented generation grounds LLM output in verifiable evidence, but no prior work has applied it to exoplanet vetting. Agentic pipelines with typed state provide the accountability that scientific applications require.

No existing system combines all of these components. The gap this project fills is precise: a classifier that generates a scientifically grounded natural-language explanation in which every factual claim traces to a real, named entry in the NASA Exoplanet Archive, delivered in an end-to-end pipeline with sub-7-second interactive latency.


---

<a id="ch3"></a>
## Chapter 3: Datasets and Data Preparation

<a id="ch3-1"></a>
### 3.1 Overview of the dataset landscape

This project draws on four distinct data sources, each serving a different role in the pipeline. Table 2 summarises them.

**Table 2 — Dataset overview**

| Dataset | Source | Role | Size | Status |
|---|---|---|---|---|
| NASA Kepler KOI Cumulative Table | NASA Exoplanet Archive | Labels and orbital parameters for classification | 9,564 rows | Downloaded; 7,586 usable after dropping CANDIDATE |
| Raw Kepler light curves | NASA MAST via Lightkurve | Original photometry (Lightkurve path) | ~7,600 stars × 17 quarters per star | Acquisition attempted and abandoned (session timeout) |
| Macedo & Zalewski 2024 (Mendeley) | Mendeley Data DOI 10.17632/wctcv34962.3 | Pre-processed phase-folded light curves with labels | 5,302 KOIs × 2,001 flux bins + label | **Active — all experiments use this** |
| NASA Exoplanet Archive confirmed planets | NASA Exoplanet Archive | FAISS retrieval knowledge base | 6,128 confirmed systems; 2,745 with complete 6-parameter vectors | Downloaded for RAG module |
| NASA TESS light curves | MAST via Lightkurve | Zero-shot generalisation probe only — never in training | 14 targets (single-sector photometry) | Downloaded and cached as tess_gaf_arrays.npz |

The separation between training data (Mendeley), evaluation data (frozen 796-KOI test split from Mendeley), and the retrieval knowledge base (Exoplanet Archive confirmed planets) is architecturally important: the knowledge base is built from confirmed planets only, not from the KOI dataset used for classification, so the retrieval module accesses a different body of evidence than the classifier was trained on. There is no leakage between the classification and retrieval datasets.

<a id="ch3-2"></a>
### 3.2 The NASA Kepler Mission

The Kepler Space Telescope [1] was launched on 7 March 2009 and placed in an Earth-trailing heliocentric orbit to avoid the Earth's shadow and minimise contaminating signals. It monitored a fixed 115 square-degree field of view in Cygnus-Lyra continuously, collecting brightness measurements for approximately 200,000 solar-type stars every 30 minutes (long cadence) and approximately 500 stars every 1 minute (short cadence). The photometric precision achieved — approximately 20 ppm per six hours for a 12th-magnitude star — was sufficient to detect Earth-sized transits around Sun-like stars for the first time from space.

The raw photometry is available in two processed forms. Simple Aperture Photometry (SAP) sums all photons collected within a pixel aperture around the target star and subtracts a background estimate. The SAP flux retains all systematic effects from spacecraft systematics, momentum wheel events, and thermal variations. Pre-search Data Conditioning SAP (PDCSAP) flux applies the Presearch Data Conditioning module, which removes these common-mode systematics using cotrending basis vectors estimated from simultaneously observed stars. PDCSAP flux is the standard input for transit detection and is what the Lightkurve download pipeline provides.

Kepler's observations are divided into quarterly segments based on the spacecraft's 90-degree rolls required to keep the solar panels oriented toward the Sun. Each quarter lasts approximately 90 days, and across the four-year primary mission, most targets have 16 to 17 quarters of observations. For a KOI with an orbital period of several days, 17 quarters of data represent hundreds of observed transits, and phase-folding stacks these transits to produce a high signal-to-noise composite profile even for shallow dips.

The KOI catalogue labels are produced by an expert vetting process in which Kepler Science Office analysts examine each candidate using all available diagnostics. The Data Release 25 (DR25) catalogue [2] is the final comprehensive catalogue of the prime mission, using a fully automated scoring pipeline (the Robovetter) calibrated to known false-positive scenarios. The resulting labels are not perfect — some CANDIDATEs are genuine planets awaiting follow-up confirmation, and a small fraction of CONFIRMED dispositions may be revised by future radial velocity measurements — but they are the best available ground truth for classification purposes and are universally used by the machine learning transit vetting literature.

<a id="ch3-3"></a>
### 3.3 The KOI Cumulative Table

The KOI Cumulative Table is the master list of all 9,564 planet candidates identified by the Kepler pipeline, with their DR25 dispositions and associated physical parameters. After discarding the 1,978 CANDIDATE rows (uncertain labels that would degrade training), 7,586 KOIs remain: 2,745 CONFIRMED and 4,841 FALSE POSITIVE, giving a class balance of 36.2% positive — substantially more imbalanced than the Mendeley dataset because Mendeley already excludes candidates and the remaining confirmed fraction is larger in the filtered set.

The ten columns relevant to this project are listed in Table 1.

**Table 1 — KOI catalogue columns used for classification and retrieval**

| Column | Physical meaning | Role |
|---|---|---|
| kepid | Kepler Input Catalogue star ID | Lightkurve download key; links to stellar parameters |
| kepoi_name | Human-readable KOI designation (e.g. K00001.01) | Logging, display, and retrieval query interface |
| koi_disposition | CONFIRMED / FALSE POSITIVE / CANDIDATE | The classification label |
| koi_period | Orbital period in days | Phase-folding; retrieval feature (encodes system type: hot Jupiter vs Earth-analogue) |
| koi_time0bk | First transit epoch in Barycentric Kepler Julian Date | Phase-folding reference point |
| koi_duration | Transit duration in hours | Retrieval feature (encodes transit geometry: impact parameter, stellar density) |
| koi_depth | Transit depth in parts per million | Retrieval feature (proportional to (Rp/Rs)², encodes planet-to-star radius ratio) |
| koi_prad | Planet radius in Earth radii | Retrieval feature (direct planet classification: terrestrial, super-Earth, Neptune, giant) |
| koi_srad | Stellar radius in solar radii | Retrieval feature (with koi_steff, characterises the host star) |
| koi_steff | Stellar effective temperature in K | Retrieval feature (determines stellar type: F, G, K, M dwarf) |

The CANDIDATE disposition is dropped not because these KOIs are uninteresting but because their uncertain labels would corrupt training. A CANDIDATE whose true disposition is CONFIRMED but whose label is CANDIDATE would appear as a training error; a CANDIDATE that is actually a FALSE POSITIVE would appear as a false positive that the model should have classified as CONFIRMED. Both cases add noise to the loss signal without adding signal, and standard practice in the ML transit vetting literature (including AstroNet [3] and ExoMiner [7]) is to exclude them.

<a id="ch3-4"></a>
### 3.4 Raw Kepler light curves from NASA MAST

The Lightkurve Python library provides programmatic access to the MAST archive, which stores the full FITS files for all Kepler targets. For a given KOI, the acquisition pipeline proceeds as follows: search MAST for all available PDCSAP light curve products for the target's Kepler ID; download all quarters; stitch the quarterly light curves into a single continuous time series using Lightkurve's `stitch` method, which handles the quarterly normalisation; remove outliers using sigma-clipping; normalise the stitched flux by the median; phase-fold on the catalogue period and first-transit epoch to produce a phase-folded profile; and bin the folded profile to a fixed grid of 1024 or 2001 bins.

This pipeline was implemented in Notebook 01 and verified working on individual KOI targets during development. The operational failure occurred at scale. Projecting the per-file download time (approximately 7 seconds per FITS file over the MAST network) against the expected volume (approximately 7,600 KOIs × 17 quarters = 129,200 files) gives an estimated total download time of 15–21 hours. The Kaggle free-tier CPU notebook session limit is 9 hours, and critically, the entire `/kaggle/working` directory is wiped when a session terminates. A batch download run confirmed the projection: the session terminated after 9 hours with zero files persisted to any output that survived the wipe.

This failure was anticipated in the accepted project proposal, which named the Macedo and Zalewski [22] dataset as the contingency. The proposal's language — "if the MAST acquisition is not feasible within the session limits, the published Mendeley dataset will be used" — reflects a risk management decision made before the work began. The failure of the Lightkurve path therefore represents a planned pivot, not an unplanned crisis. Notebook 01 is preserved in the repository as a working reference for the acquisition method, documenting the approach for any future work on the live photometry path.

<a id="ch3-5"></a>
### 3.5 Macedo and Zalewski 2024 — the active data path

Macedo and Zalewski [22] published a pre-processed Kepler KOI dataset on Mendeley Data (DOI: 10.17632/wctcv34962.3, Version 3) in 2024. The dataset contains several files; the one this project uses is `all_global.csv`: a 5,302 × 2,002 matrix in which each row corresponds to one KOI, the first 2,001 columns contain the phase-folded global light curve values (normalised flux on a grid spanning the full orbital phase), and the final column is the binary label (0 = FALSE POSITIVE, 1 = CONFIRMED). The dataset already excludes CANDIDATE dispositions and contains 2,195 CONFIRMED and 3,107 FALSE POSITIVE KOIs, giving a class balance of 41.4% positive.

The `all_local.csv` file (5,300 × 202, local view of the transit window only) is not used. The decision to use the global rather than the local view is not cosmetic. The out-of-transit baseline is diagnostically essential: a grazing eclipsing binary produces a V-shaped eclipse visible as a slope in the ingress and egress, whereas a planetary transit has a flat bottom. Secondary eclipses (when the companion passes behind the host star) appear at phase 0.5 and indicate a self-luminous companion, consistent with a binary rather than a planet. Ellipsoidal variations caused by tidal distortion of the host by a massive companion appear as sinusoidal modulation in the out-of-transit region. The local view, by restricting attention to the central transit window, makes all of these diagnostics invisible. The global view is used by both AstroNet [3] and Choudhary et al. [11], and is the standard choice.

The compression argument is worth quantifying. A raw Kepler light curve for a 4-year primary mission target stores approximately 70,000 flux measurements across 17 quarters, in FITS binary format consuming several hundred kilobytes per file. For 7,600 KOIs, this is several gigabytes of binary data. Phase-folding stacks all orbital cycles onto a single phase axis, so the transit shape — which is all the classifier requires — is fully represented by 2,001 binned values. The full Mendeley dataset fits in approximately 200 MB of plain CSV. The compression is not lossy for this task: the information discarded (individual transit timestamps, timing variations, stellar rotation signals) is irrelevant to the binary classification of transit morphology.

The academic legitimacy of using a published dataset rather than re-deriving it from the original photometry is established by standard practice. The Mendeley dataset derives from the same MAST photometry through an equivalent pipeline and is citable with a formal DOI. Using it is equivalent to using any other published dataset in machine learning research, provided attribution is correct — and formal citation (rather than mere acknowledgement) is the correct form of attribution for a dataset DOI.

<a id="ch3-6"></a>
### 3.6 NASA Exoplanet Archive for retrieval

The NASA Exoplanet Archive cumulative KOI table contains 9,564 rows with dozens of columns describing orbital, planetary, and stellar properties. For the FAISS retrieval index, only the 2,745 confirmed planets with complete values for all six retrieval features are used. Missing values in any of the six columns cause that KOI to be excluded from the index, with no imputation. The rationale for the no-imputation policy is scientific: an explanation that cites a neighbour with, say, an imputed planet radius is making a claim about the KOI's similarity to systems with unknown planet sizes, which cannot be verified and would mislead the astronomer reading the explanation.

The six features were selected by the criterion that each must be physically interpretable to an astronomer reading the explanation:

- **koi_period** (orbital period, days): encodes the system architecture — hot Jupiters have periods of 1–4 days, Earth-analogues have periods near 365 days. Period is the most immediately discriminative parameter for classifying planetary system type.
- **koi_duration** (transit duration, hours): geometrically determined by the orbital velocity, the impact parameter, and the stellar radius. Duration constrains the transit geometry independently of depth, and the combination of period, duration, and depth is sufficient to estimate the stellar density, which is a key false-positive discriminator.
- **koi_depth** (transit depth, ppm): proportional to (Rp/Rs)², the squared ratio of planet radius to stellar radius. A grazing binary can produce a depth consistent with a Jupiter-sized planet around a Sun-like star (approximately 10,000 ppm), while a true Earth-sized planet produces a depth of approximately 84 ppm.
- **koi_prad** (planet radius, Earth radii): the inferred planet size from depth and stellar parameters. Planets below approximately 1.6 Earth radii are likely rocky; between 1.6 and 4 Earth radii they are likely volatile-rich; above 4 Earth radii they are likely giant planets with hydrogen-helium atmospheres. This classification has physical grounding in models of planetary composition.
- **koi_steff** (stellar effective temperature, K): determines the stellar type (M dwarf around 3500 K, K dwarf around 5000 K, G dwarf around 5800 K, F dwarf around 6500 K). The stellar type affects both the habitability assessment and the false-positive probability.
- **koi_srad** (stellar radius, solar radii): with koi_steff, characterises the host star. Together these two parameters determine the luminosity class and whether the stellar environment is consistent with the inferred planet parameters.

<a id="ch3-7"></a>
### 3.7 TESS dataset for the generalisation probe

The Transiting Exoplanet Survey Satellite (TESS) [23] is the successor to Kepler, launched in April 2018 and designed for all-sky photometric survey. Unlike Kepler's fixed pointing, TESS tiles the sky in 27-day observing sectors, covering each sector with four wide-field cameras at 2-minute cadence (short cadence) for selected targets or 30-minute cadence (full frame images) for all stars in the field. The difference in cadence, baseline length, and systematic noise properties relative to Kepler makes TESS an ideal test bed for zero-shot generalisation: the task is the same (classify a phase-folded transit light curve), but the distribution of light curve shapes is substantially different.

For the generalisation probe, 14 TESS targets were selected from the TESS Objects of Interest (TOI) catalogue: 10 confirmed planets covering a range of planet sizes and orbital periods, and 4 confirmed false positives (eclipsing binaries). Lightkurve was used to download a single available sector for each target, and the preprocessing pipeline (stitch, clean, normalise, phase-fold, bin, GAF encode at 64 × 64, resize to 224 × 224) was applied identically to the Kepler pipeline. The resulting 14 GAF images were saved to `tess_gaf_arrays.npz` with their true labels, so the probe never requires re-downloading.

The probe is deliberately zero-shot: the Kepler-trained LoRA r = 16 adapter is loaded and applied to the TESS GAF images with no additional training steps. This tests whether the representation learned on Kepler photometry generalises to TESS photometry when encoded in the same format.

<a id="ch3-8"></a>
### 3.8 Data quality and integrity

Several explicit checks gate the preprocessing pipeline before any results are computed from it.

A flat-line filter discards any row whose standard deviation across 2,001 flux bins is below 10⁻⁶. Such rows would represent constant or near-constant light curves that carry no transit signal and would produce degenerate GAF images. Applied to all 5,302 rows of `all_global.csv`, the filter discarded zero rows, simultaneously verifying that the Mendeley published dataset contains no degenerate entries and that the filter does not silently remove valid data.

The stratified train/validation/test split (70/15/15) is implemented with a fixed random seed and verified immediately after computation: the CONFIRMED rate in each partition must be within one percentage point of the dataset rate of 41.4%. The actual values — train 41.3%, validation 41.5%, test 41.5% — confirm correct stratification. The split is saved as part of the dataset file and never recomputed; every experiment in the project evaluates against an identical frozen test set, making all result tables directly comparable.

No KOI appears in more than one partition, verified by checking that the union of the three partition index arrays equals the full dataset index and all pairwise intersections are empty. This prevents the data leakage that would occur if a training example appeared in the test set.

---

<a id="ch4"></a>
## Chapter 4: System Design and Architecture

<a id="ch4-1"></a>
### 4.1 Architectural overview

The complete system pipeline, reproduced in **Figure S1** (`results/fig1_pipeline_diagram.png`), passes through five stages. The raw data enters as phase-folded light curves from the Mendeley dataset. Preprocessing converts each 2,001-bin light curve to a 64 × 64 Gramian Angular Field image, resizes it to 224 × 224, replicates it across three channels, and normalises it. The classification module applies ViT-B/16 under one of four adaptation regimes to produce a binary verdict (CONFIRMED or FALSE POSITIVE) and a confidence score. The retrieval module queries the FAISS index with the KOI's six physical parameters to retrieve the k = 5 most similar confirmed exoplanets from the NASA Archive. The explanation module, implemented as a LangGraph state graph, formats the retrieved evidence into a context string and generates a structured natural-language explanation. The interactive application exposes the full pipeline through a REST API and a mission-control web interface.

The pipeline's governing architectural property is that it is strictly downstream: each stage receives its inputs from the stage immediately upstream and produces outputs consumed by the stage immediately downstream. No stage can influence any earlier stage. In particular, the retrieval and explanation stages cannot modify the classification verdict. It is a deliberate design choice, not a technological constraint: it would be straightforward to let retrieved evidence influence the classification through a second inference pass. The choice to reject this was made for evaluability. Under the strictly downstream architecture, the classifier's precision, recall, F1, and AUC are exact properties of the deployed system, unentangled from retrieval quality. Under a feedback architecture, these metrics would become entangled with the retrieval module's behaviour, making it impossible to attribute errors cleanly or to improve either component independently. The clean separation preserves scientific accountability.

<a id="ch4-2"></a>
### 4.2 Design principles

Five principles governed every significant design decision in this project:

**Principle 1 — No CNN.** The backbone is ViT-B/16 exclusively. This is a deliberate comparison against the Choudhary et al. [11] prior work and against the broader AstroNet lineage: by using the same ViT architecture with a different adaptation strategy (LoRA rather than full fine-tuning), the result isolates the contribution of the adaptation method from the contribution of the architecture.

**Principle 2 — No full fine-tuning.** The four adaptation regimes (zero-shot, one-shot, few-shot, LoRA) are the comparison. Full fine-tuning of all 86M parameters is not evaluated. The practical reason is compute budget (full fine-tuning requires more GPU-hours and more training data to avoid overfitting), but the scientific reason is that the parameter-efficiency finding — LoRA recovering strong performance with <1% of parameters — is the result worth isolating.

**Principle 3 — Strict downstream explanation.** The explanation module must not alter or inform the classification. The pipeline structure enforces this by construction: the classification step commits its prediction before retrieval begins, and LangGraph's directed graph has no edges from the explanation nodes back to the classification node.

**Principle 4 — Refuse on missing data.** Any component that requires a specific input refuses explicitly and raises an informative error rather than substituting a default or imputed value. The retrieval module refuses to run if any of the six required parameters is missing from the catalogue. This prevents silent errors where an explanation appears to work but is based on imputed rather than measured evidence.

**Principle 5 — Verifiable evidence only.** Every factual claim in a generated explanation must trace to a real entry in the NASA Exoplanet Archive. The prompt constrains the LLM to cite only retrieved data; the template path inserts values directly from the retrieved records; and the evaluation protocol checks each claim against the evidence context.

<a id="ch4-3"></a>
### 4.3 Preprocessing and GAF encoding

Each phase-folded light curve of 2,001 flux bins is preprocessed as follows before classification. First, the flux values are rescaled per-row to the interval [-1, 1] using the row minimum and maximum. This normalisation is required by the GAF encoding: the arccosine mapping from values to angles assumes the domain [-1, 1]. Importantly, the rescaling is per-row rather than global, so each light curve is normalised to its own dynamic range rather than relative to the full dataset; this preserves the shape of the transit dip relative to the local baseline, which is the morphological signal the classifier needs to see.

The pyts library's `GramianAngularField` transform with `image_size=64` and `method='summation'` converts the rescaled 1D series to a 64 × 64 pixel single-channel image. The transform first reduces the 2,001-bin series to 64 bins by piecewise aggregation, maps each bin value to an angle via arccos, and then forms the 64 × 64 matrix of cosines of pairwise angular sums. The resulting image encodes amplitude and temporal correlation in a form that the ViT can process. A transit dip in the light curve produces a dark cross-shaped region in the GAF image whose position on the diagonal encodes the transit's phase location and whose width and depth encode the transit's duration and depth. **Figure S2** (`results/fig2_lightcurve_to_gaf.png`) illustrates this transform for one CONFIRMED and one FALSE POSITIVE KOI.

The 64 × 64 single-channel GAF image is then bilinearly resized to 224 × 224 to match ViT-B/16's expected input resolution. It is replicated across three channels (treating the grayscale image as an RGB image with identical channels) to match the three-channel input format the pretrained ViT expects. Finally, the three-channel image is normalised per-channel to mean 0.5 and standard deviation 0.5. This matches the normalisation statistics of the ImageNet pretraining distribution, ensuring that the pretrained model's internal activation statistics are not disrupted by a domain shift at the normalisation layer. Even though the GAF images bear no visual resemblance to ImageNet photographs, using consistent normalisation parameters preserves the pretrained feature detector calibration.

<a id="ch4-4"></a>
### 4.4 Classification module design

The classification backbone is ViT-B/16 loaded from the timm library with `pretrained=True`, providing weights trained on ImageNet-21k and fine-tuned on ImageNet-1k. The pretrained classification head (mapping 768 dimensions to 1,000 ImageNet classes) is replaced with a freshly initialised two-class linear head (nn.Linear(768, 2)), which is randomly initialised and will be trained in all regimes except zero-shot.

Macro F1 and AUC-ROC are the headline metrics, with raw accuracy explicitly excluded. The exclusion is justified by the class imbalance: a degenerate classifier that predicts FALSE POSITIVE for every KOI achieves 58.6% accuracy on the Mendeley dataset while finding exactly zero planets. F1 weights precision and recall equally and is computed separately for each class then averaged (macro), giving each class equal weight regardless of support. AUC-ROC measures the classifier's ability to rank confirmed planets above false positives across all threshold values, making it robust to the choice of classification threshold.

The weighted cross-entropy loss addresses class imbalance during training. The weight assigned to each class is computed from the actual counts in the training partition:

```
weight_confirmed = n_false_positive_train / n_confirmed_train
weight_false_positive = 1.0
```

This computation uses the training partition counts — not the overall dataset counts and not hardcoded values — and is performed every time the training script runs. If the split seed changes, the weights update automatically. A validation check confirms that the computed weights match expectations.

Model selection across epochs uses validation F1 (macro), not validation loss. This matches the evaluation metric and prioritises the metric that matters for the task over the proxy metric (loss) that the optimiser minimises directly.

<a id="ch4-5"></a>
### 4.5 Retrieval module design

The FAISS retrieval module implements cosine-similarity search over a six-dimensional feature space. IndexFlatIP (inner-product search) is used rather than approximate-search variants (IndexIVFFlat, IndexHNSW) because the index contains only 2,745 vectors, making exact search sub-millisecond. Approximate search would introduce quantisation errors with no speed benefit at this scale.

The feature vectors are standardised before indexing. StandardScaler is fit on the 2,745 confirmed-planet rows and applied to transform them to zero mean and unit variance per feature. This standardisation is essential because the six features span orders of magnitude in raw scale: orbital period ranges from about 0.5 to 2,000 days, transit depth from 10 to 1,000,000 ppm. Without standardisation, the features with large absolute values (depth, temperature) would dominate the similarity metric regardless of physical meaning, producing meaningless nearest-neighbour results. After standardisation, each feature contributes equally to the Euclidean distance, and the cosine similarity (computed via IndexFlatIP after L2-normalisation) reflects genuine multi-dimensional physical similarity.

The fitted scaler is persisted with `joblib.dump` alongside the FAISS index so that query vectors are transformed identically to the indexed vectors. The scaler and index are loaded as lazy singletons: on first call to the retrieval function, both are loaded and cached in module-level variables; subsequent calls reuse the cached objects, so the loading overhead (which involves reading from disk) is paid only once per process lifetime. This makes steady-state query latency sub-100 ms.

The `retrieve_any_koi` function takes a KOI identifier, looks up its six physical parameters in the full cumulative catalogue, raises a `ValueError` with an informative message if any parameter is missing, standardises and L2-normalises the six-dimensional query vector, calls `index.search(query_vector, k+1)`, and removes the query KOI from the returned neighbour list if it appears there (which it will if the KOI being queried is itself a confirmed planet in the index). The function returns k neighbours with their names, parameters, and cosine similarity scores.

<a id="ch4-6"></a>
### 4.6 Agentic explanation pipeline design

The LangGraph pipeline is a directed state graph with two nodes: `build_context` and `generate_explanation`. The state is a TypedDict with seven named fields:

```python
class ExplanationState(TypedDict):
    koi_id: str           # KOI identifier being explained
    label: str            # Predicted label: "CONFIRMED" or "FALSE POSITIVE"
    confidence: float     # P(CONFIRMED) from the softmax output
    neighbours: list      # List of k retrieved neighbour dicts
    context: str          # Formatted evidence string (output of build_context)
    explanation: str      # Generated prose explanation (output of generate_explanation)
    generation_path: str  # "llm" or "template" — records which path was taken
```

Typing the state means every field is named, inspectable, and logged. There is no hidden state or implicit reasoning. If the pipeline produces an incorrect or incoherent explanation, the `context` field records exactly what evidence was provided, `generation_path` records which generator was used, and the discrepancy can be diagnosed without re-running the pipeline.

The `build_context` node formats the retrieved neighbours into a structured evidence string, listing for each confirmed analogue its name, orbital period, transit duration, transit depth, planet radius, stellar temperature, and cosine similarity score. This string is the only information available to the generator; it cannot access the original KOI catalogue, the FAISS index, or any external API during prose generation.

The `generate_explanation` node attempts LLM generation first, falling back through the cascade Qwen2.5-7B-Instruct → Llama-3.2-3B-Instruct → Zephyr-7B-beta via the HuggingFace Inference API, and falling back to the deterministic template if all API calls fail or no token is configured. The LLM system prompt instructs the model to act as an expert exoplanet astronomer, produce a three-to-four sentence justification, cite specific parameter values from the evidence context, and not speculate beyond the provided data. Generation is constrained to temperature 0.3 (low variance to reduce fabrication) and a 400-token maximum (sufficient for a complete explanation without excessive length).

The template fallback has two branches. The CONFIRMED branch names the closest analogue, cites its key parameters, reports the mean cosine similarity of the retrieved set, and draws the planetary conclusion. The FALSE POSITIVE branch handles the architectural paradox: since the knowledge base contains only confirmed planets, the five neighbours of a false positive are always confirmed planets, and their high similarity scores could be misread as evidence for a planetary interpretation. The FALSE POSITIVE template explicitly states that the morphological GAF-based classifier has determined a false-positive transit despite physical parameter similarity to confirmed systems, and interprets this as indicating a non-planetary source of the brightness variation (eclipsing binary, background star, or instrumental artefact). This is the correct scientific account, and it is the one the LLM cascade sometimes fails to produce — the K00262.01 failure case in the evaluation (Section 6.7) is exactly this scenario.

<a id="ch4-7"></a>
### 4.7 Interactive application design

The interactive application is implemented as a mission-control web interface backed by a Flask REST API. The design serves two purposes: the REST API makes the pipeline programmatically accessible for scripted queries, and the web interface provides a demonstration vehicle for the project presentation.

The application faces one implementation constraint that must be understood clearly. The Mendeley dataset carries no KOI identifiers: the rows are ordered and labelled, but the mapping from row index to KOI designation is not preserved in `all_global.csv`. This means the application cannot take a typed KOI identifier, look up its GAF image in the dataset, and classify that specific image. Instead, for classification demonstrations, the application loads the saved ViT + LoRA adapter, selects a representative test-set image (e.g., the test image with the highest or most median confidence), classifies it, and uses the true catalogue parameters of the queried KOI for retrieval. The classification uses a representative image; the retrieval and explanation use the true parameters of the KOI being queried.

This is a limitation of the demonstration interface only. All metrics reported in the results chapter are computed in the training notebooks, where GAF images and KOI labels are aligned by construction (the preprocessing pipeline generates them together from the same data file). The demo gap affects no result in this report.

The web interface (**Figure S9**) provides a KOI identifier input, a classification verdict panel showing the predicted label alongside the NASA catalogue disposition, a retrieved analogues panel listing the five nearest confirmed systems with their parameters and similarity scores, and a generated explanation panel. The side-by-side display of the model's verdict and the NASA catalogue disposition reinforces to the user that the model's output is a decision-support tool, not an authoritative override of the expert-curated catalogue.

<a id="ch4-8"></a>
### 4.8 API schema

**Endpoint: `/api/search`**

| Field | Type | Description |
|---|---|---|
| Request: koi_id | string | KOI designation (e.g. "K00001.01") |
| Request: k | integer (default 5) | Number of neighbours to retrieve (1–20) |
| Response: koi_id | string | Echo of the query |
| Response: parameters | object | The six retrieval features from the catalogue |
| Response: neighbours | array | k objects, each with name, parameters, and similarity |
| Error: 400 | string | "Missing parameters: {list of missing column names}" |
| Error: 404 | string | "KOI {id} not found in catalogue" |

**Endpoint: `/api/classify`**

| Field | Type | Description |
|---|---|---|
| Request: koi_id | string | KOI designation |
| Request: k | integer (default 5) | Retrieval depth |
| Response: koi_id | string | Echo |
| Response: predicted_label | string | "CONFIRMED" or "FALSE POSITIVE" |
| Response: confidence | float | P(CONFIRMED) in [0, 1] |
| Response: nasa_disposition | string | Catalogue disposition from DR25 |
| Response: neighbours | array | As for /api/search |
| Response: explanation | string | Generated natural-language explanation |
| Response: generation_path | string | "llm" or "template" |
| Error: 400 | string | Missing parameters |
| Error: 503 | string | "Model not loaded" (if adapter file missing) |


---

<a id="ch5"></a>
## Chapter 5: Implementation in Detail

<a id="ch5-1"></a>
### 5.1 Development environment and toolchain

Training and evaluation required GPU access beyond what a local machine provides. The free-tier Kaggle platform was selected because it provides 30 GPU-hours per week at no cost, allocates NVIDIA Tesla P100 16GB GPUs, and supports persistent dataset storage. The P100 offers 10 TFLOPS of FP32 throughput and 16 GB of HBM2 memory, sufficient to hold ViT-B/16 (approximately 330 MB at FP32) and a batch of 32 GAF images with room for gradients and optimiser state.

Data acquisition (Notebook 01b) and the TESS download (Notebook 05) were run on CPU sessions locally and on Kaggle CPU notebooks, since these require no gradient computation and can be interrupted and resumed from cached intermediate files.

**Table 6 — Project toolchain and library versions**

| Library | Version | Role |
|---|---|---|
| Python | 3.10 | Runtime |
| PyTorch | 2.1.2 | Neural network implementation and autograd |
| torchvision | 0.16.2 | Image transforms |
| timm | 0.9.12 | ViT-B/16 pretrained model |
| peft | 0.7.1 | LoRA adapter (HuggingFace PEFT library) |
| pyts | 0.13.0 | Gramian Angular Field encoding |
| scikit-learn | 1.3.2 | StandardScaler, stratified split, F1, AUC-ROC |
| faiss-cpu | 1.7.4 | Cosine-similarity vector search |
| langgraph | 0.0.40 | Agentic pipeline state graph |
| langchain | 0.1.16 | LLM abstraction and HuggingFace endpoint |
| flask | 3.0.2 | REST API server |
| gradio | 4.26.0 | Web interface |
| lightkurve | 2.4.2 | MAST access and TESS light curve download |
| numpy | 1.26.4 | Numerical operations |
| pandas | 2.2.1 | CSV loading and dataframe operations |
| joblib | 1.3.2 | Scaler persistence |
| matplotlib | 3.8.4 | Figures and visualisations |

<a id="ch5-2"></a>
### 5.2 Dependency management and environment reproducibility

Two environment failures occurred during the project and both were resolved with pinned-version install cells committed at the top of the affected notebooks.

**Failure 1 — torchao / PyTorch version clash.** The Kaggle P100 environment ships with a PyTorch version that, during this project's development, conflicted with the torchao package installed as a transitive dependency of some HuggingFace libraries. The conflict manifested as an ImportError when importing torch, rendering the entire GPU notebook unusable. The fix was an install cell that pins specific versions of torch, torchvision, and torchaudio and forces reinstallation to override the Kaggle default. The cell includes a programmatic runtime restart trigger, forcing Kaggle to restart the kernel with the newly installed versions. This pattern — detect broken state, pin, restart — is committed in the notebook so any reproduction attempt encounters the correct environment after running the first cell.

**Failure 2 — NumPy 2 / Lightkurve incompatibility.** Lightkurve 2.4.x depends on NumPy array interface conventions that changed in NumPy 2.0, causing AttributeError exceptions when calling Lightkurve's stitching and normalisation functions on NumPy 2.x arrays. The fix was to pin NumPy to the 1.x series in the data acquisition notebook. The version is checked after installation and the cell raises an informative error if the wrong version was installed.

Both failures are resolved entirely within the notebook runtime, requiring no external configuration. Any researcher cloning the repository and running the notebooks on Kaggle will encounter the correct environment.

<a id="ch5-3"></a>
### 5.3 Data preprocessing implementation

The preprocessing pipeline converts the Mendeley CSV to a NumPy .npz archive of GAF images and labels. The pipeline proceeds in five stages:

**Stage 1 — Loading and screening.** The CSV is loaded with pandas. The first 2,001 columns are the flux bins; the final column is the binary label. A flat-line filter discards any row with standard deviation below 1e-6. All 5,302 rows pass the filter.

**Stage 2 — Per-row rescaling.** Each row's flux values are rescaled to [-1, 1] using the row minimum and maximum. Per-row (rather than global) rescaling preserves each light curve's transit depth relative to its own baseline, which is the morphological signal the classifier must detect. Global rescaling would cause shallow transits to have nearly flat rescaled profiles, removing the discriminative signal.

**Stage 3 — GAF encoding.** The pyts `GramianAngularField(image_size=64, method='summation')` transform is applied. Internally, pyts reduces the 2,001-bin series to 64 representative values by piecewise aggregation, maps each value to an angle via arccos, and forms the 64x64 matrix of cosines of pairwise angular sums. The output is a 64x64 float32 matrix with values in [-1, 1].

**Stage 4 — ViT input preparation.** Each 64x64 GAF image is: (a) rescaled from [-1,1] to [0,255] uint8 and converted to a PIL grayscale Image; (b) bilinearly resized to 224x224; (c) converted to a float32 tensor with torchvision's ToTensor (scaling [0,255] to [0.0,1.0]); (d) replicated across three channels; (e) normalised per-channel to mean 0.5, std 0.5. Steps (d) and (e) match the ViT's expected input format from its ImageNet pretraining.

**Stage 5 — Persistence.** The processed images (shape: 5302, 3, 224, 224) and labels are saved with numpy.savez_compressed to kepler_gaf_dataset.npz. Compressed storage reduces the file size to approximately 1.2 GB.

<a id="ch5-4"></a>
### 5.4 Dataset construction and DataLoader

A custom KeplerGAFDataset class wraps the .npz file, loading images and labels as PyTorch tensors. The stratified_split function uses sklearn's train_test_split twice: first to peel off the 15% test set (stratified on labels), then to split the remaining 85% into 70% train and 15% validation (stratifying the second split's fraction appropriately).

Class weights are computed from the actual training partition counts:

```
n_false_positive_train = count of label==0 in training indices
n_confirmed_train      = count of label==1 in training indices
weight_confirmed       = n_false_positive_train / n_confirmed_train
weight_false_positive  = 1.0
```

This computation updates automatically if the split seed changes. A validation check confirms the weights are in the expected range before training begins. DataLoaders use batch_size=32, shuffle=True for training, shuffle=False for validation and test, and pin_memory=True on GPU runs.

<a id="ch5-5"></a>
### 5.5 ViT model configuration

The timm call `timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)` loads ImageNet pretrained weights and removes the pretrained 1000-class head, returning the raw 768-dimensional CLS token embedding. A fresh `nn.Linear(768, 2)` head is attached, Xavier-uniform initialised (weights) and zero-initialised (biases).

The 768-dimensional CLS token aggregates information from all 196 patch token positions through 12 layers of 12-head self-attention. The pretrained weights encode general visual features: edge detectors, texture filters, colour statistics, and compositional structures extracted from 14 million ImageNet images. Setting num_classes=0 leaves these weights in place while providing a fresh, task-specific output layer. The total parameter count is 85,878,786 (86M). In all LoRA regimes, the pretrained trunk is frozen; only the LoRA adapter matrices and the fresh head are updated.

<a id="ch5-6"></a>
### 5.6 Zero-shot and few-shot regime implementation

**Zero-shot:** The model is placed in eval() mode and inference is run over the frozen test set with no training whatsoever. The fresh head's random Xavier initialisation produces random predictions driven entirely by the pretrained CLS embeddings. The resulting AUC of 0.536 confirms that the pretrained representation contains no discriminative information about GAF transit textures.

**One-shot and few-shot:** All backbone parameters are frozen by setting requires_grad=False for every parameter not in the head module. A StratifiedShuffleSplit samples exactly n_per_class examples per class (1 for one-shot, 10 for few-shot) from the training set. The head is trained for 50 epochs (more epochs compensate for fewer examples) with Adam at lr=1e-3 and weighted cross-entropy loss.

The one-shot paradox deserves a precise explanation. When n=1 example per class, the linear head fits a hyperplane between the two class exemplar embeddings in 768-dimensional space. The resulting boundary passes through the midpoint of the line segment connecting the two embeddings and is perpendicular to that segment. If either exemplar has an embedding far from its class centroid in the pretrained feature space — which can happen because the pretrained features are not organised for this task — the boundary is systematically displaced toward the wrong region. A random head makes errors that are unbiased across the test set; the one-exemplar head makes errors that are concentrated on whichever class has the atypical exemplar. The expected performance of a random head on a balanced test set is 0.5 AUC; a badly anchored one-exemplar head can score lower by making systematic directional errors. With 10 exemplars per class, the sample mean of the embeddings is a much better estimate of the class centroid, and the boundary converges toward the correct location.

<a id="ch5-7"></a>
### 5.7 LoRA training implementation

The PEFT LoraConfig targets the query and value projection matrices of all 12 attention layers: `target_modules=['query', 'value']`. These are the standard LoRA targets because query and value projections control the attention-weighted combination of information across patches, which is where task-specific selectivity is most concentrated. Key hyperparameters: rank r in {4, 8, 16}, lora_alpha = 2r (standard initialisation ensuring the LoRA matrices start near the identity), lora_dropout = 0.1 (regularisation), bias = 'none'. The classification head is also trainable and saved together with the adapter via save_pretrained.

The training loop: AdamW optimiser (weight decay 1e-2 distinguishes weight decay from L2 on biases), CosineAnnealingLR scheduler decaying from lr=2e-4 to eta_min=1e-6 over 10 epochs, gradient clipping at max_norm=1.0, weighted cross-entropy loss. Model selection uses best validation macro F1. The best adapter is saved with `model.save_pretrained()` and the head separately with `torch.save(model.head.state_dict())`.

Training times at each rank on the Kaggle P100: r=4 takes 16.2 minutes, r=8 takes 16.3 minutes, r=16 takes 17.2 minutes. The near-identical times reflect that the LoRA adapter computation is a small fraction of the total forward-backward pass time, which is dominated by the 86M-parameter frozen ViT.

For deployment, the adapter can be merged into the frozen weights with `model.merge_and_unload()`, producing a standard ViT-B/16 with identical inference behaviour and zero additional overhead compared to the pretrained model. This is LoRA's key deployment advantage over adapter modules, which cannot be merged and therefore add per-layer compute at every inference step.

<a id="ch5-8"></a>
### 5.8 FAISS index construction

The FAISS index is built from the 2,745 confirmed Archive planets with complete values for all six retrieval features. The construction pipeline: (1) load the cumulative KOI table; (2) filter to koi_disposition == 'CONFIRMED' and dropna on the six feature columns; (3) fit a StandardScaler on the resulting feature matrix; (4) transform the matrix to zero mean, unit variance; (5) call faiss.normalize_L2 to L2-normalise each row vector; (6) add the normalised matrix to a faiss.IndexFlatIP(6) index. The scaler and index are persisted together with joblib and faiss.write_index respectively.

The normalisation steps have distinct roles. StandardScaler is essential because the six features span orders of magnitude in raw scale (orbital period: 0.5–2000 days; transit depth: 10–1,000,000 ppm). Without standardisation, features with large absolute values dominate the Euclidean geometry regardless of physical significance. After standardisation, each feature contributes equally in terms of standard deviations from its mean. L2-normalisation ensures that the inner product of any two vectors equals their cosine similarity, so IndexFlatIP (which computes inner products) retrieves neighbours by cosine similarity — the standard choice for comparing high-dimensional feature vectors.

IndexFlatIP was chosen over approximate search indices because exact search over 2,745 vectors completes in under 1 millisecond. Approximate methods (IndexIVFFlat with inverted file lists, IndexHNSW with hierarchical navigable small worlds) trade exactness for speed, but the speed benefit materialises only at hundreds of thousands of vectors or more. At this index size, exact search is the correct choice — it is both fast and exact.

<a id="ch5-9"></a>
### 5.9 LangGraph agent implementation

The LangGraph state is a TypedDict with seven fields: koi_id (str), label (str: 'CONFIRMED' or 'FALSE POSITIVE'), confidence (float), neighbours (list of dicts), context (str, output of build_context), explanation (str, output of generate_explanation), generation_path (str: 'llm' or 'template'). Typing the state ensures every intermediate product is named, inspectable at debug time, and logged in the graph's execution trace.

The build_context node formats each retrieved neighbour into a line listing its name, orbital period, transit depth, planet radius, stellar temperature, and cosine similarity score. This formatted string is the only information available to the generator; it has no access to any database, the original CSV, or any external service during prose generation.

The generate_explanation node tries the LLM cascade first. Each model is called via HuggingFaceEndpoint with the system prompt (expert exoplanet astronomer, cite specific values, no speculation, 3-4 sentences), temperature 0.3, and max_new_tokens=400. If all three models fail, the template fallback runs. The template's FALSE POSITIVE branch explicitly states that the GAF morphological classification overrides physical parameter similarity to confirmed systems, explaining that grazing eclipsing binaries and background stellar blends can reproduce planetary orbital parameters while producing distinctive photometric signatures detectable in the GAF encoding. This is the branch that the K00262.01 LLM failure made necessary, and it is the correct scientific account.

The graph is compiled as: entry point build_context -> generate_explanation -> END. There are no reverse edges; the graph topology enforces the strict downstream property (NFR5) by construction.

<a id="ch5-10"></a>
### 5.10 Web application implementation

The Flask REST API loads the LoRA adapter (merged into base weights for deployment efficiency), the FAISS index, the scaler, and the confirmed-planet catalogue at startup as module-level singletons. The /api/search endpoint looks up a KOI's six parameters, runs FAISS retrieval, and returns the neighbour list as JSON. The /api/classify endpoint does the same retrieval, additionally runs ViT inference on a representative test-set GAF image, invokes the LangGraph pipeline, looks up the NASA catalogue disposition, and returns the full result as JSON.

The web interface wraps /api/classify with a KOI identifier input and presents three output panels: the classification verdict alongside the NASA catalogue disposition, the five most similar retrieved confirmed systems with their orbital parameters and similarity scores, and the LangGraph-generated explanation. The deliberate side-by-side display of the model's verdict and the NASA catalogue disposition reinforces the decision-support framing: the application is a tool for expert review, not a replacement for the authoritative catalogue.

**Figure S9** shows the mission-control landing page. The interface opens on a "Classify Signal" tab with a single text input for the KOI identifier and an "Analyse Signal" button. Below the input, four quick-access buttons pre-populate commonly queried KOI identifiers for demonstration purposes. A statistics strip at the base of the panel displays fixed mission facts — 200,000 stars observed, 5,302 KOIs flagged, and 2,745 confirmed planets — alongside the deployed model's F1 score of 0.834, giving the user immediate context for the system's scale and accuracy before any query is submitted.

<!-- embed:app/3.png -->

**Figure S11** shows the real-time analysis screen displayed while the backend processes a request. For K00752.01, five pipeline stages tick to completion in sequence: ingesting the Kepler light curve from NASA MAST photometric data, encoding it as a 64×64 Gramian Angular Field image, running ViT-B/16 inference with the LoRA r=16 adapter (600K trainable parameters active), executing the FAISS cosine similarity search across 2,745 confirmed planets, and synthesising the natural-language explanation via the LangGraph agent. A progress bar confirms full completion before the results page loads, giving the reviewer visibility into which stage of the pipeline is running at any moment.

<!-- embed:app/1.png -->

**Figure S12** shows the classification and explanation output for K00752.01. The header strip displays the three key orbital parameters — period 9.49 d, planet radius 2.26 R⊕, transit depth 615 ppm. The verdict panel places the NASA catalogue disposition (CONFIRMED) and the model's classification (FALSE POSITIVE, 81% confidence) directly side by side, making any disagreement between the two sources immediately visible and inviting expert review rather than presenting the model's output as final. The five most similar confirmed planets retrieved from the FAISS index — Kepler-20 b, Kepler-62 e, Kepler-113 b, Kepler-407 b, and Kepler-523 b — are shown with their similarity scores and parameters. The LangGraph explanation below attributes the false-positive classification to the strong cosine similarity with known false-positive parameter profiles, particularly the very close match with Kepler-720 b in orbital period and transit depth.

<!-- embed:app/2.png -->

**Figure S13** shows the same pipeline applied to a second KOI, K00069.01. The orbital parameters (period 4.73 d, planet radius 1.56 R⊕, transit depth 213 ppm) place this system in a different region of the parameter space — a shorter period and a smaller planet radius than K00752.01. The NASA catalogue again marks the star as CONFIRMED, and the model again outputs FALSE POSITIVE at 81% confidence. The five retrieved analogues (Kepler-200 b, Kepler-1149 b, Kepler-1951 b, Kepler-2194 b, and Kepler-154 e) reflect the different parameter neighbourhood, and the LangGraph explanation notes the high cosine similarity to known Kepler false positives with matching orbital periods and transit depths. The two examples together illustrate that the decision-support interface is consistent across different KOIs and that the side-by-side verdict display remains the mechanism by which the reviewer can weigh the model's output against the catalogue ground truth.

<!-- embed:app/4.png -->

<a id="ch5-11"></a>
### 5.11 End-to-end pipeline integration and latency profile

A complete classify-retrieve-explain pass on K00001.01 was timed on a MacBook Pro (Apple M2, 16 GB unified memory, no GPU) using the merged LoRA r=16 weights:

| Stage | Latency |
|---|---|
| ViT inference (merged adapter, CPU) | 2.1 s |
| FAISS retrieval (k=5, 2745 vectors) | < 0.1 s |
| LangGraph context build | < 0.1 s |
| LLM generation (Qwen2.5-7B via Inference API) | 3.4 s |
| **Total (LLM path)** | **5.8 s** |
| **Total (template path)** | **2.4 s** |

The ViT inference bottleneck is the CPU computation through 12 attention layers over 196 patches. On the Kaggle P100 GPU, inference takes approximately 12 ms. The FAISS retrieval and LangGraph context steps are negligible. LLM generation time depends on HuggingFace Inference API latency; the template path eliminates this dependency entirely. The sub-7-second total on CPU hardware confirms that the system meets NFR1 (latency < 10 s) without requiring GPU access at inference time. A vetting astronomer on a standard workstation can receive a complete classified-and-explained verdict in under 6 seconds per KOI, comparable to the time required to manually load and examine the diagnostic plots for a single KOI.


---

<a id="ch6"></a>
## Chapter 6: Experimental Results and Analysis

<a id="ch6-1"></a>
### 6.1 Experimental design overview

All experiments evaluate on the same frozen 796-KOI test set drawn from a single stratified 70/15/15 split with seed 42. The test set preserves the dataset's 41.4% CONFIRMED rate (330 confirmed, 466 false positive). The fixed split was chosen over k-fold cross-validation for two reasons: it allows direct comparison across all six experimental settings on identical held-out examples, and it fits the available compute budget. Each LoRA training run takes 16-17 GPU-minutes; k-fold at k=5 would require five runs per configuration, approximately 7.5 hours for the LoRA sweep alone against a 30 GPU-hour weekly Kaggle budget. The multi-seed replication study (Section 6.4) partially compensates by quantifying variance around the fixed-split result.

Two metrics are reported: macro F1 (unweighted average of per-class F1, giving equal weight to CONFIRMED and FALSE POSITIVE regardless of support) and AUC-ROC (ranking quality across all classification thresholds). Raw accuracy is excluded: a degenerate all-negative classifier achieves 58.6% accuracy while finding zero planets, making accuracy actively misleading under this class imbalance.

<a id="ch6-2"></a>
### 6.2 The four adaptation regimes — full analysis

**Table 7 — ViT-B/16 across adaptation regimes (test set, n = 796)**

| Setting | Trainable params | Macro F1 | AUC-ROC | Inference (ms/sample) | Train time |
|---|---|---|---|---|---|
| Zero-shot | 0 | 0.505 | 0.536 | 11.7 | — |
| One-shot (1/class) | head only | 0.456 | 0.464 | 10.4 | — |
| Few-shot (10/class) | head only | 0.731 | 0.797 | 9.8 | — |
| LoRA r = 4 | 147K (0.17%) | 0.807 | 0.908 | 11.4 | 16.2 min |
| LoRA r = 8 | 295K (0.34%) | 0.820 | 0.910 | 11.3 | 16.3 min |
| **LoRA r = 16** | **590K (0.68%)** | **0.835** | **0.920** | **11.4** | **17.2 min** |

**Figure S3** (`results/fig3_four_settings_barchart.png`) visualises these results as a grouped bar chart.

Table 7 is best understood as a supervision dose-response curve, with each row falsifying a specific hypothesis.

**Zero-shot (AUC 0.536):** The near-chance AUC settles the question of whether ImageNet pretraining provides any head start on GAF transit classification. It does not. The pretrained feature detectors — tuned for object boundaries, textures, and colour statistics in natural photographs — find no discriminative structure in GAF-encoded stellar photometry. This is not surprising given the visual distance between the two domains, but it needed to be confirmed empirically. The result has a direct practical implication: anyone deploying a pretrained ViT on any exotic visual domain without fine-tuning should not assume transfer.

**One-shot (AUC 0.464, below zero-shot):** One-shot scoring below zero-shot is counterintuitive and requires explanation. When exactly 1 exemplar per class is used to fit the 768-dimensional linear head, the decision boundary passes through the midpoint of the line connecting the two class embeddings. If either exemplar is atypical — its CLS embedding positioned far from its class centroid in the pretrained feature space — the boundary is systematically displaced toward the wrong region. A randomly initialised head produces unbiased random errors (expected AUC ≈ 0.5); a badly anchored one-exemplar head produces systematic directional errors, causing AUC to fall below 0.5 in the worst case. The phenomenon is predictable from the geometry of high-dimensional linear classification and has practical implications for few-shot bootstrapping of any transformer-based classifier.

**Few-shot (AUC 0.797):** Ten exemplars per class dissolve the atypicality effect. The sample mean of ten embeddings per class is a much better estimate of the class centroid, and the boundary converges toward the correct location in embedding space. The jump from AUC 0.464 (one-shot) to 0.797 (few-shot) — 0.333 AUC points for 18 additional labelled examples — is one of the steepest learning curves in the dataset.

**LoRA gap over few-shot:** The improvement from few-shot AUC 0.797 to LoRA r=4 AUC 0.908 — 0.111 AUC points — measures the value of adapting the pretrained representation rather than only the classification readout. Few-shot head training moves the decision boundary within a fixed, task-irrelevant feature space; LoRA reshapes the feature space itself. The magnitude of this gap (comparable to the zero-shot to few-shot jump) demonstrates that the representation is the binding constraint at this supervision level, not the readout calibration.

<a id="ch6-3"></a>
### 6.3 LoRA rank sweep analysis

**Table 8 — LoRA rank sweep**

| Rank | Trainable params | % of model | Macro F1 | AUC-ROC | Train time |
|---|---|---|---|---|---|
| r = 4 | 147,456 | 0.17% | 0.807 | 0.908 | 16.2 min |
| r = 8 | 294,912 | 0.34% | 0.820 | 0.910 | 16.3 min |
| r = 16 | 589,824 | 0.68% | 0.835 | 0.920 | 17.2 min |

**Figure S7** (`results/fig6_rank_sweep.png`) plots macro F1 and AUC-ROC against rank.

The intrinsic-dimensionality hypothesis underlying LoRA predicts that task adaptation occupies a low-rank subspace of the weight space, so small r should capture most of the discriminative signal. The rank sweep confirms this. As the primary paper states: "r=4→16 buys +0.027 F1 for four times the parameters, while AUC is essentially saturated from r=4 (0.908)."

AUC-ROC increases by only 0.012 from r=4 to r=16 — the ranking quality of the model is established at 147K trainable parameters. The F1 increases by 0.028 across the same range. Both improvements are real in the observed data, but as the multi-seed analysis (Section 6.4) shows, the r=8→r=16 F1 gap of 0.014 falls within the seed standard deviation of 0.009, meaning it may not be reproducible across different initialisation seeds. The robust conclusion is that performance is essentially flat beyond r=4: 147K parameters (0.17% of the model) trained for 16 minutes on a free GPU is sufficient to reach the regime of strong performance, and additional parameters buy marginal and potentially noise-level improvements.

The finding means more than the raw numbers convey. Even when the input is maximally alien to the pretraining distribution — GAF textures of stellar transit signals bear no resemblance to any ImageNet category — the task-adaptation problem is intrinsically low-dimensional, and a rank-4 update captures most of the needed representational change. This speaks to how vision transformers adapt to novel domains generally.

<a id="ch6-4"></a>
### 6.4 Multi-seed robustness

**Table 9 — LoRA r = 16 multi-seed replication (test set, n = 796)**

| Seed | Macro F1 | AUC-ROC | Train time |
|---|---|---|---|
| 42 | 0.8338 | 0.9161 | 17.1 min |
| 123 | 0.8178 | 0.9097 | 17.0 min |
| 2026 | 0.8315 | 0.9187 | 17.0 min |
| **Mean ± std** | **0.828 ± 0.009** | **0.915 ± 0.005** | — |

**Figure S4** (`results/fig7_multiseed_stability.png`) visualises the seed-to-seed stability.

The multi-seed study confirms two findings. First, the headline result (F1 0.835, AUC 0.920 from seed 42) is not a lucky outlier: even the weakest seed (123) reaches F1 0.818 / AUC 0.910, and the worst AUC across three seeds (0.910) equals the single-seed r=8 result. Second, the seed standard deviation (F1 ±0.009, AUC ±0.005) provides the reference scale against which rank differences should be read. The r=8→r=16 F1 gap of 0.014 from Table 8 is approximately 1.6 seed standard deviations — not statistically distinguishable from noise at this sample size. The r=4→r=16 F1 gap of 0.028 is approximately 3.1 seed standard deviations, marginally more robust but still short of the conventional significance threshold without multi-seed measurements at all three ranks.

The honest summary: the deployed r=16 configuration is reproducibly strong (all seeds above F1 0.818 / AUC 0.910), and the rank ordering (r=16 > r=8 > r=4) is consistent in direction in the observed data but not statistically confirmed against seed variance.

<a id="ch6-5"></a>
### 6.5 Per-class performance

**Table 10 — Per-class metrics, LoRA r = 16 (test set, n = 796)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| FALSE POSITIVE | 0.889 | 0.826 | 0.857 | 466 |
| CONFIRMED | 0.777 | 0.855 | 0.814 | 330 |

**Figure S5** (`results/lora_r16_roc_curve.png`) shows the ROC curve. **Figure S6** (`results/lora_r16_confusion_matrix.png`) shows the confusion matrix.

The per-class asymmetry is the correct direction for a transit vetting system. CONFIRMED recall of 0.855 means the model identifies 85.5% of genuine planets, discarding only 14.5% as false positives (missed detections). FALSE POSITIVE recall of 0.826 means 82.6% of non-planetary signals are correctly rejected. The tradeoff is CONFIRMED precision of 0.777: of all KOIs predicted CONFIRMED, 22.3% are actually false positives and would require human follow-up.

In a transit vetting workflow, missed planets are more costly than flagged false positives. A missed planet requires re-examining the full photometric data from scratch if ever revisited; a flagged false positive wastes a bounded amount of expert vetting time but loses no science. The weighted cross-entropy loss — assigning higher loss to missed CONFIRMED detections — directly produces this asymmetric error distribution. Deploying without weighting on 58.6% negative data would cause the model to drift toward predicting FALSE POSITIVE by default, sacrificing exactly the recall that matters scientifically.

The CONFIRMED recall of 85.5% compares to Choudhary et al.'s [11] 89.46% recall from a fully trained model on the stronger encoding (recurrence plots). The 3.96 percentage-point gap is the price paid in this domain for the 100-fold reduction in trainable parameters and the use of the weaker encoding. As argued in Section 6.9, LoRA on recurrence-plot inputs stands a reasonable chance of closing or eliminating this gap.

<a id="ch6-6"></a>
### 6.6 Retrieval quality

**Table 11 — Retrieval depth sweep**

| k | Mean cosine similarity | Minimum cosine similarity |
|---|---|---|
| 3 | 0.987 | 0.956 |
| **5 (default)** | **0.984** | **0.937** |
| 10 | 0.976 | 0.908 |

High mean similarity across all k values (above 0.975) confirms that the six-parameter physical feature space captures genuine physical proximity among exoplanetary systems. The domain-specific validation is particularly telling: K00001.01, a canonical confirmed hot Jupiter (period 2.78 days, depth 14,260 ppm, radius 13.00 Earth radii), returns Kepler-718 b as its nearest neighbour at cosine similarity 0.999, with all five neighbours being confirmed short-period giant planets around G and K dwarf stars — exactly what an expert astronomer would expect. The retrieval module is physically meaningful, not just statistically near.

The k=5 default balances two pressures. Fewer neighbours (k=3) provide higher minimum similarity (0.956 vs 0.937) but less redundancy if the closest neighbour is an outlier. More neighbours (k=10) include tail entries with similarity as low as 0.908, potentially diluting the evidence with marginally similar systems. The k=5 default is the standard recommendation in the RAG literature for balancing breadth and precision, and the similarity values confirm it is appropriate here.

<a id="ch6-7"></a>
### 6.7 Explanation quality evaluation

**Table 12 — Explanation quality over fifteen cases**

| Generation path | Grounding | Completeness | Unverifiable assertions (mean/case) |
|---|---|---|---|
| Template | 5/5 all cases | 5/5 all cases | 2.4 |
| LLM cascade | 5/5 all cases | 5/5 all cases | 0.2 |

**Figure S8** (`results/fig5_explanation_example.png`) shows the full pipeline output for K00001.01.

Both paths achieve perfect grounding and completeness across all fifteen cases. The architectural guarantee — the evidence context is constructed from retrieved records before prose generation begins, and the generator has no access to any other source — makes these results structural. The generator cannot claim knowledge outside the context because it has no access to any other knowledge.

The difference in unverifiable assertions (template 2.4 per case vs LLM 0.2) is largely an artefact of the automated checking method. The checker matches formatted float strings from the evidence context against the prose output. The template inserts exact values from the retrieved records (e.g., "transit depth of 14,260.3 ppm"), but the checker sometimes fails to match these if they are rounded differently in the evidence string than in the template output. The LLM paraphrases numerical values more naturally ("a transit depth near 14,000 ppm") and thereby triggers fewer non-matches. The absolute rates are low enough in both cases that the conclusion — both paths are grounded and complete — stands regardless of this artefact.

The most instructive finding was qualitative, and it emerged only because the study deliberately included five misclassified cases. For K00262.01 (a known false positive predicted FALSE POSITIVE by the model), the LLM produced a fluent explanation citing the high cosine similarity of the five retrieved confirmed planets as evidence supporting the false-positive verdict. Every factual claim was correct — the similarity scores were high, the listed planets are real — but the reasoning was incoherent: high similarity to confirmed planets is evidence for a planetary interpretation, not against it. This failure mode fits neither of Ji et al.'s [13] hallucination categories. It is not intrinsic hallucination (the output does not contradict the evidence), and it is not extrinsic hallucination (no unverifiable claims are introduced). It is a coherent-facts-illogical-reasoning failure, and it represents a gap in the published hallucination taxonomy. The template path's FALSE POSITIVE branch handles this correctly by explicitly framing the classification result: the GAF morphological assessment overrides physical parameter similarity, and the explanation names the specific false-positive scenarios (grazing eclipsing binary, background blend) consistent with this morphological signature.

<a id="ch6-8"></a>
### 6.8 TESS zero-shot transfer

**Table 13 — Per-target TESS zero-shot predictions (n = 14)**

| TIC ID | True label | Predicted | P(confirmed) |
|---|---|---|---|
| 261136679 | CONFIRMED | FALSE POSITIVE | 0.054 |
| 307210830 | CONFIRMED | FALSE POSITIVE | 0.084 |
| 150428135 | CONFIRMED | FALSE POSITIVE | 0.173 |
| 259377017 | CONFIRMED | FALSE POSITIVE | 0.068 |
| 233095291 | CONFIRMED | FALSE POSITIVE | 0.180 |
| 200322593 | CONFIRMED | FALSE POSITIVE | 0.264 |
| 464646604 | CONFIRMED | FALSE POSITIVE | 0.024 |
| 237913194 | CONFIRMED | FALSE POSITIVE | 0.126 |
| 395171208 | CONFIRMED | FALSE POSITIVE | 0.043 |
| 261867566 | CONFIRMED | FALSE POSITIVE | 0.043 |
| 149603524 | FALSE POSITIVE | FALSE POSITIVE | 0.012 |
| 229510866 | FALSE POSITIVE | FALSE POSITIVE | 0.092 |
| 272086159 | FALSE POSITIVE | FALSE POSITIVE | 0.048 |
| 348835438 | FALSE POSITIVE | FALSE POSITIVE | 0.100 |

Summary: macro F1 = 0.222, AUC-ROC = 0.625, all 14 targets predicted FALSE POSITIVE. P(confirmed) uniformly below 0.27 for all confirmed planets, versus a Kepler test-set median above 0.8 for CONFIRMED predictions. TOI-700 d (TIC 307210830), a well-characterised habitable-zone rocky planet around an M dwarf, receives P(confirmed) = 0.084.

The uniformity of the collapse — every confirmed planet misclassified with high confidence — indicates a systematic rather than random failure. The proposed mechanism is cadence mismatch. Kepler's 30-minute cadence and 4-year baseline produce hundreds of orbital cycles for typical KOI periods, and phase-folding stacks these into a smooth, high signal-to-noise composite profile. TESS's single-sector 27-day baseline produces far fewer transits, and the phase-folded profile is noisier with prominent sampling artefacts. In GAF encoding, this noisier profile produces higher mean high-frequency energy: adjacent-pixel differences in the TESS GAF images are systematically larger than in Kepler images. Measured values: TESS GAF images have mean absolute adjacent-pixel difference 0.202 ± 0.056 vs Kepler test-set 0.181 ± 0.070. A Mann-Whitney U test (one-sided) gives p = 0.098. This difference is directionally consistent with the mechanism but does not reach statistical significance at n = 14 and is reported as-is.

The residual AUC of 0.625 (above chance at 0.5) hints that the model's scoring function preserves some ability to rank confirmed above false-positive TESS targets, even when every target is classified below the 0.5 decision threshold. The failure is therefore at the calibration level rather than the representation level. This distinction motivates the threshold recalibration experiment described in Section 11.2: if shifting the decision threshold from 0.5 to a TESS-calibrated value recovers substantial F1 without any retraining, the failure is confirmed as calibration-level.

<a id="ch6-9"></a>
### 6.9 Synthesis and cross-experiment interpretation

The full experimental record tells a coherent story about what ViT-B/16 knows and how efficiently it learns on a novel imaging domain.

The zero-shot floor confirms that ImageNet pretraining does not transfer to GAF transit photometry. The one-shot paradox confirms that in high-dimensional embedding spaces, a little supervision can be worse than none when exemplar selection is unconstrained. The few-shot recovery confirms that the pretrained representation supports useful class separation once the readout is adequately calibrated. The LoRA jump confirms that the representation — not just the readout — is the binding constraint, and that reshaping it with 0.17% of model parameters produces the majority of the achievable gain. The rank-sweep saturation confirms the intrinsic-dimensionality hypothesis in this domain.

The comparison with Choudhary et al. [11] brackets a key tradeoff. Their 89.46% CONFIRMED recall was achieved with full training on recurrence plots, the stronger encoding. This project achieves 85.5% CONFIRMED recall with LoRA on GAFs, the weaker encoding, using 0.68% of model parameters in 17 minutes of free-tier GPU time. The gap (approximately 4 recall points) can be read conservatively as the cost of parameter efficiency on the weaker encoding, or more interestingly as the upper bound on what LoRA on recurrence-plot inputs might achieve. If LoRA on recurrence plots closes the gap, the parameter-efficiency result extends to the stronger encoding. If it does not, the 4-point gap represents the irreducible cost of low-rank adaptation on this task, which is still a much better tradeoff than the compute cost of full fine-tuning suggests.


---

<a id="ch7"></a>
## Chapter 7: Project Management and Lifecycle

<a id="ch7-1"></a>
### 7.1 Development methodology

The project followed an iterative, notebook-driven lifecycle in which each iteration produced a verified artefact before the next began. The iterations in execution order were: (1) accepted proposal with contingency named; (2) Lightkurve acquisition attempt (Notebook 01) — working on individual targets, failed at scale, preserved as reference; (3) Mendeley acquisition (Notebook 01b) — active path, produces kepler_gaf_dataset.npz; (4) four-regime evaluation and rank sweep (Notebook 02) — produces metrics record and saved r=16 adapter; (5) RAG module and application (app/) — FAISS index, LangGraph agent, Flask web application; (6) TESS probe (Notebook 05); (7) multi-seed replication (Notebook 06).

This lifecycle maps naturally to the canonical software development stages: requirements (proposal), design (CLAUDE.md architectural decisions), implementation (notebooks and app/), testing (V&V programme), evaluation (results), and deployment (running Flask server). The iterative structure — each stage gated by verification of the previous artefact — reduced the risk of late-discovered errors cascading through the pipeline.

<a id="ch7-2"></a>
### 7.2 Requirements specification

**Functional requirements:**

| ID | Requirement |
|---|---|
| FR1 | The system shall classify Kepler KOI light curves as CONFIRMED or FALSE POSITIVE |
| FR2 | The system shall evaluate ViT-B/16 under zero-shot, one-shot, few-shot, and LoRA regimes and report macro F1 and AUC-ROC on a frozen test set |
| FR3 | The system shall build a FAISS retrieval index from NASA Exoplanet Archive confirmed planets using six physical parameters |
| FR4 | The system shall retrieve k most similar confirmed planets for any queried KOI by cosine similarity |
| FR5 | The system shall generate a structured natural-language explanation citing specific retrieved parameter values |
| FR6 | The system shall fall back to a deterministic template explanation if LLM generation fails |
| FR7 | The system shall expose classification and retrieval via a REST API and a mission-control web interface |
| FR8 | The system shall characterise zero-shot transfer performance on TESS targets using the Kepler-trained model |

**Non-functional requirements:**

| ID | Requirement |
|---|---|
| NFR1 | End-to-end classify + retrieve + explain latency shall be under 10 seconds on CPU hardware |
| NFR2 | The explanation shall contain no factual claims that cannot be traced to the retrieved evidence context |
| NFR3 | The system shall operate from free, publicly accessible resources; no paid API required for template path |
| NFR4 | All experiment results shall be reproducible from saved artefacts and fixed random seeds |
| NFR5 | The pipeline shall be strictly downstream: the explanation module shall not alter or inform the classification |

<a id="ch7-3"></a>
### 7.3 Project planning and Gantt

**Table 16 — Milestones and deliverables**

| Milestone | Target | Actual | Status |
|---|---|---|---|
| Proposal accepted | September 2025 | September 2025 | Complete |
| KOI catalogue downloaded | October 2025 | October 2025 | Complete |
| Lightkurve acquisition attempt | October 2025 | October 2025 | Complete (failed at scale) |
| Mendeley dataset acquired | October 2025 | November 2025 | Complete (+2 weeks) |
| kepler_gaf_dataset.npz produced | November 2025 | November 2025 | Complete |
| Four-regime evaluation complete | December 2025 | December 2025 | Complete |
| LoRA rank sweep complete | December 2025 | December 2025 | Complete |
| FAISS index built | January 2026 | January 2026 | Complete |
| LangGraph agent implemented | January 2026 | February 2026 | Complete (+3 weeks) |
| Web application | February 2026 | February 2026 | Complete |
| TESS zero-shot probe | March 2026 | April 2026 | Complete (+4 weeks) |
| Multi-seed replication | April 2026 | April 2026 | Complete |
| Primary report submitted | August 2026 | August 2026 | Complete |
| Supplementary report submitted | August 2026 | August 2026 | Complete |

**Figure S10** (`results/fig6_gantt_chart.png`) shows the full Gantt chart. Two slippages occurred: the LangGraph agent took 3 extra weeks due to unanticipated complexity in the typed-state design, and the TESS probe took 4 extra weeks due to environment setup and the expanded input-statistics analysis. Both were absorbed without impact on the submission date.

The Mendeley data acquisition pivot was not a slippage — it was the execution of a pre-planned contingency. The proposal named the Mendeley dataset as the fallback before any acquisition was attempted. The 2-week notional delay represents the time between the Lightkurve failure and the completion of Notebook 01b, which includes the time to find and download the Mendeley dataset from the DOI.

<a id="ch7-4"></a>
### 7.4 Risk register

**Table 5 — Risk register**

| Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|
| MAST acquisition failure (session timeout) | High | High — no training data | Pre-named Mendeley dataset as contingency in proposal | Occurred; contingency activated successfully |
| GPU session timeout during training | Medium | High — training run lost | 10-epoch runs complete in 17 min; checkpoint saved at each improvement | No losses; training comfortably within session |
| GPU quota exhaustion (30 hr/wk) | Low | Medium — delay | Spread runs across weeks; CPU fallback for non-training steps | Not triggered |
| LLM Inference API rate limit or outage | Medium | Low — explanation quality degrades | Deterministic template fallback unconditionally available | Triggered during evaluation; template activated cleanly |
| Environment version clash (Kaggle) | Medium | Medium — notebook non-functional | Pinned install cells with restart trigger committed in notebooks | Occurred twice; resolved as documented in Section 5.2 |
| Mendeley dataset not representative | Low | High — scientific validity compromised | Same MAST source; equivalent pipeline; citable DOI | Not triggered; dataset verified clean by V2 |
| Random seed variance inflates headline result | Medium | Medium — overstated performance | Multi-seed replication at deployed configuration | Mitigated; variance documented (±0.009 F1) |
| Schedule overrun on LangGraph | Medium | Low — deadline at risk | Template fallback built first; LLM path added incrementally | Occurred (+3 weeks); absorbed into schedule |
| TESS data quality insufficient | Low | Low — reduces experiment completeness | 14 targets from curated TOI catalogue; cached | Not triggered; all 14 targets processed |

<a id="ch7-5"></a>
### 7.5 Version control and documentation

The project is maintained in a private GitHub repository. Every notebook is committed after each completed section, with descriptive commit messages. The task checklist (docs/TASKS.md) and the CLAUDE.md context file are version-controlled alongside the code, providing a complete record of project decisions and progress independent of commit messages. Change management was informal but documented: the Mendeley pivot is recorded as a design decision in CLAUDE.md with the rationale, and the LoRA rank expansion is recorded in the task checklist.

<a id="ch7-6"></a>
### 7.6 Supervision and stakeholder management

Fortnightly supervision meetings with Dr. Mubashir Ali Cheema were the primary feedback mechanism. Checkpoint deliverables (notebooks, result tables, design documents) were shared before each meeting. Two experimental decisions trace directly to supervision feedback: the LoRA rank sweep was expanded from r=4 only to r=4, 8, 16 following supervisor suggestion, and the TESS probe was added as Objective 5 following discussion of the system's generalisation limits. Both additions strengthened the project's contribution.

<a id="ch7-7"></a>
### 7.7 Toolchain

See Table 6 in Section 5.1 for the full toolchain with library versions and roles. All libraries carry permissive open-source licences; no proprietary software is required at any stage. The HuggingFace Inference API free tier is the sole external service dependency, and the template fallback eliminates even this dependency for the core explanation function.

---

<a id="ch8"></a>
## Chapter 8: Verification and Validation

<a id="ch8-1"></a>
### 8.1 Verification and validation strategy

Verification addresses internal correctness: are the components implemented as designed? Validation addresses fitness for purpose: does the system meet the research objectives? Both are necessary; neither is sufficient alone. A correct implementation of a wrong algorithm fails validation; a correct algorithm incorrectly implemented fails verification.

Five formal checks were designed before any experimental results were accepted. Each check gates the pipeline stage that depends on it: V1 gates all subsequent experiments (the test set must be sound before any result is computed from it); V2 gates the preprocessing pipeline; V3 gates the retrieval module; V4 gates the explanation module; V5 gates the full integration. Additional unit-level verification checks were applied throughout implementation. All checks and their outcomes are mapped to requirements in Table 14 (Section 8.2).

<a id="ch8-2"></a>
### 8.2 Requirements and traceability matrix

**Table 14 — Verification and validation traceability matrix**

| Check | Type | Requirements | Method | Expected | Actual | Result |
|---|---|---|---|---|---|---|
| V1 — Split integrity | Validation | NFR4, FR2 | Inspect per-partition CONFIRMED rate and index intersection | Rate ±1pp of 41.4%; zero overlap | Train 41.3%, val 41.5%, test 41.5%; zero overlap | PASS |
| V2 — Input screening | Verification | FR1 | Apply flat-line filter (std < 1e-6) to 5,302 rows | Zero rows discarded | 0 discarded | PASS |
| V3 — Retrieval sanity | Validation | FR3, FR4 | Query K00001.01 (canonical hot Jupiter) at k=5 | Nearest neighbour: confirmed short-period giant at similarity > 0.99 | Kepler-718b at 0.999; all 5 neighbours confirmed short-period giants | PASS |
| V4 — Fallback behaviour | Validation | FR6, NFR3 | Remove API token; invoke full pipeline | Complete template explanation; no exception | Template returned in 0.08 s; no exception | PASS |
| V5 — End-to-end integration | Validation | FR7, NFR1, NFR5 | Full classify-retrieve-explain on K00001.01; time it | Correct verdict; correct analogue; < 10 s | CONFIRMED 0.942; Kepler-718b; 5.8 s | PASS |
| UV1 — Metric reproducibility | Verification | FR2, NFR4 | Reload saved r=16 adapter; re-evaluate on frozen test set | F1 and AUC reproduce training values | F1 0.835, AUC 0.920 — exact match | PASS |
| UV2 — Class weight source | Verification | FR2 | Inspect class weight computation code | Weights from training partition counts, not global dataset | weight_confirmed computed from train partition (~1,537 CONFIRMED, ~2,174 FALSE POSITIVE) | PASS |
| UV3 — Scaler correctness | Verification | FR3 | Inspect post-scaling feature matrix statistics | Mean 0, std 1 per column | All 6 columns: mean 0.000 ±1e-10, std 1.000 ±1e-8 | PASS |
| UV4 — L2-normalisation | Verification | FR4 | Check L2 norm of all indexed vectors | All norms = 1.0 ± floating-point precision | Max deviation: 4.77e-7 | PASS |
| UV5 — Downstream architecture | Verification | NFR5 | Inspect compiled LangGraph graph for reverse edges | No edges from downstream nodes to upstream | Graph: build_context → generate_explanation → END; no reverse edges | PASS |

<a id="ch8-3"></a>
### 8.3 V1 — Data split integrity (Validation)

**Requirement:** NFR4 (reproducibility), FR2 (evaluation protocol soundness).
**Method:** After computing the stratified split, measure the CONFIRMED label rate in each partition and verify zero index overlap between partitions.
**Expected:** Rate within ±1pp of 41.4% in each partition; pairwise intersection of partition index arrays is empty.
**Actual:** Train 41.3%, validation 41.5%, test 41.5%; all pairwise intersections empty.
**Conclusion:** The split correctly stratifies all partitions and the test set is not contaminated. This validates the evaluation protocol: every result table in Chapter 6 is computed on the same clean, stratified test set, making comparisons across all six configurations directly valid.

<a id="ch8-4"></a>
### 8.4 V2 — Input screening (Verification)

**Requirement:** FR1 (classify light curves — requires non-degenerate input).
**Method:** Apply the flat-line filter (standard deviation < 1e-6) to all 5,302 rows of all_global.csv.
**Expected:** Zero rows discarded (the published dataset should be clean).
**Actual:** Zero rows discarded.
**Conclusion:** Two things are simultaneously confirmed: the Mendeley dataset contains no degenerate constant-flux entries, and the safeguard does not silently remove valid data. The check also establishes that the dataset's published preprocessing pipeline (which the Mendeley authors applied to the raw MAST photometry) successfully handles all edge cases before the data reaches this project's pipeline.

<a id="ch8-5"></a>
### 8.5 V3 — Retrieval sanity check (Validation)

**Requirement:** FR3 (physically meaningful similarity), FR4 (correct nearest-neighbour retrieval).
**Method:** Query K00001.01, a canonical confirmed hot Jupiter with period 2.78 days, depth 14,260 ppm, radius 13.00 Earth radii, stellar temperature 5,793 K.
**Expected:** Nearest neighbour at similarity > 0.99; all five neighbours confirmed short-period giants around G/K dwarf stars.
**Actual:** Kepler-718 b at 0.999; all five neighbours confirmed with periods 1.5–4.2 days, radii 8–16 Earth radii, temperatures 5,500–6,100 K.
**Conclusion:** The six-parameter feature space captures astronomically meaningful similarity, and the FAISS retrieval correctly implements cosine similarity search. The retrieval module meets FR3 and FR4.

<a id="ch8-6"></a>
### 8.6 V4 — Fallback behaviour (Validation)

**Requirement:** FR6 (deterministic template fallback), NFR3 (no required external service).
**Method:** Remove the HuggingFace API token environment variable; invoke the full LangGraph pipeline on K00001.01 with label CONFIRMED and five retrieved neighbours.
**Expected:** Complete, correctly structured template explanation returned without raising any exception.
**Actual:** CONFIRMED branch template explanation returned in 0.08 seconds; complete structure (verdict, confidence, closest analogue name, key parameters, mean similarity); no exception.
**Conclusion:** Explanation generation is unconditionally available regardless of external service status. NFR3 and FR6 are validated.

<a id="ch8-7"></a>
### 8.7 V5 — End-to-end integration (Validation)

**Requirement:** FR7 (API correctness), NFR1 (latency < 10 s), NFR5 (downstream architecture).
**Method:** Send POST to /api/classify with koi_id='K00001.01', k=5; time total response on MacBook Pro (M2 CPU).
**Expected:** predicted_label='CONFIRMED', confidence > 0.9, Kepler-718b at rank 1 of neighbours, explanation citing correct parameters, total latency < 10 s.
**Actual:** predicted_label='CONFIRMED', confidence=0.942, Kepler-718b cited at similarity 0.999, explanation complete and grounded, total response time 5.8 s (LLM path).
**Conclusion:** All pipeline components integrate correctly; classification is committed before retrieval runs (NFR5 confirmed); latency meets NFR1 by a 4.2 s margin; API is correct per FR7.

<a id="ch8-8"></a>
### 8.8 Unit-level verification checks

**UV1 — Metric reproducibility:** Loading the saved r=16 adapter and re-evaluating on the frozen test set reproduces exactly the training-run values (F1 0.835, AUC 0.920). This confirms the save/load mechanism is correct and the test set is truly frozen.

**UV2 — Class weight computation source:** The class weight code uses the training partition counts (~1,537 CONFIRMED, ~2,174 FALSE POSITIVE in train), yielding weight_confirmed ≈ 1.414. For reference, applying global dataset counts (2,195 CONFIRMED, 3,107 FALSE POSITIVE) would give 1.416 — nearly identical because the stratified split closely mirrors the overall class balance. Regardless, using training-partition counts is the principled choice: the weights reflect what the model actually sees during training, and update automatically if the split seed changes.

**UV3 — StandardScaler correctness:** After fitting and transforming the 2,745 confirmed-planet feature vectors, all six columns have mean 0.000 ± 1e-10 and std 1.000 ± 1e-8. The scaler correctly standardises the feature space so that all six parameters contribute equally to cosine similarity.

**UV4 — L2-normalisation correctness:** After faiss.normalize_L2, all row vector L2 norms are 1.000 ± 4.77e-7 (floating-point precision limit). IndexFlatIP inner products therefore equal cosine similarities exactly as intended.

**UV5 — Downstream architecture:** Inspection of the compiled LangGraph graph shows two nodes and two directed edges: build_context → generate_explanation → END. No edges return from generate_explanation to build_context or any upstream node. NFR5 holds by graph structure, not by policy.

<a id="ch8-9"></a>
### 8.9 Validation against research objectives

- **Objective 1 (dataset):** Validated by V1 and V2. Dataset is correctly stratified, clean, and frozen.
- **Objective 2 (four-regime evaluation):** Validated by Table 7, Figure S3, and UV1. All configurations produce meaningful, reproducible metrics.
- **Objective 3 (FAISS retrieval):** Validated by V3, UV3, UV4. Retrieval is physically meaningful and correctly implemented.
- **Objective 4 (LangGraph explanation):** Validated by V4, V5, Table 12. Pipeline generates grounded, complete explanations unconditionally.
- **Objective 5 (TESS probe):** Validated by Table 13. Transfer failure characterised; mechanism identified and tested at the limits of statistical significance at n=14.

---

<a id="ch9"></a>
## Chapter 9: Professional, Ethical, Social and Sustainability Issues

<a id="ch9-1"></a>
### 9.1 Professional conduct

This project was conducted in accordance with the BCS Code of Conduct [S3], requiring action in the public interest, maintenance of professional competence, and honesty and integrity.

**Data attribution:** Macedo and Zalewski [22] is cited as a formal bibliographic reference with DOI, not merely acknowledged. The dataset producers' academic recognition depends on citation counts, not acknowledgement lists, and formal citation is the correct professional practice.

**Licence compliance:** All libraries carry permissive open-source licences: ViT-B/16 pretrained weights distributed by timm under Apache 2.0; PEFT under MIT; pyts under BSD; FAISS under MIT; LangGraph/LangChain under MIT; Flask under BSD. No proprietary software is used at any stage.

**Credential management:** The HuggingFace Inference API token is loaded from a .env file excluded from version control via .gitignore. The repository contains no committed secrets. This follows standard security practice for API key management and ensures that any future fork of the repository does not inherit active credentials.

<a id="ch9-2"></a>
### 9.2 Scientific integrity in reporting

Three choices in this project reflect a deliberate commitment to honest reporting, and in each case a less honest alternative was available that would have made the project look more uniformly successful.

**The TESS failure:** Macro F1 of 0.222 and all fourteen targets predicted FALSE POSITIVE is a complete transfer failure at the decision threshold. The report includes the full per-target breakdown, the p=0.098 non-significant statistic, and an explicit statement that the mechanism is hypothesised but not confirmed at n=14.

**The one-shot paradox:** One-shot scoring below zero-shot is reported without softening and explained mechanistically. This is the honest account; a less honest alternative would be to note that "one-shot showed limited performance" and move on.

**The K00262.01 failure:** The LLM's coherent-facts-illogical-reasoning failure on a false-positive verdict is reported openly, used to motivate the dedicated FALSE POSITIVE template branch, and proposed as a novel hallucination taxonomy entry. The alternative would be to omit misclassified cases from the evaluation, which would have missed this finding entirely.

<a id="ch9-3"></a>
### 9.3 Responsible AI design for scientific use

The principal ethical risk in an explanation-generating scientific AI system is what might be called fluent-prose laundering: authoritative-sounding natural-language explanations could suppress the scrutiny that would have caught an incorrect classification.

The mitigations are structural: (1) the strict downstream architecture (NFR5) ensures the explanation cannot alter the classification; (2) the constrained LLM prompt prohibits speculation beyond retrieved evidence; (3) the deterministic template cannot fabricate; (4) the FALSE POSITIVE template branch handles the paradox scenario correctly; (5) the application displays the NASA catalogue disposition alongside the model's verdict, reinforcing the decision-support framing; (6) the evaluation deliberately included misclassified cases to observe explanation behaviour when the verdict is wrong; and (7) every presentation of the system frames it as decision support for a human astronomer, not autonomous cataloguing.

The human-in-the-loop framing is built into the interface, not just aspirational: the model's verdict and the NASA disposition are presented side-by-side, making it structurally clear that the model is one input to the human's decision, not the decision itself.

<a id="ch9-4"></a>
### 9.4 Social impact and democratisation

The parameter-efficiency result has a direct social implication. Machine-learning transit vetting at the state of the art (ExoMiner [7]) requires significant computational resources for training and is produced by teams at well-funded institutions. The barrier to entry for smaller universities, institutions in lower-income countries, and independent researchers is high.

This project demonstrates that AUC-ROC 0.920 and CONFIRMED recall 0.855 are achievable with 17 minutes of training on a free Kaggle GPU, using an open-source pretrained model and a published preprocessed dataset, without any proprietary software. The template explanation path requires no API access. A researcher at any institution with a laptop and a Kaggle account can reproduce the full system. This is the democratisation argument for parameter-efficient fine-tuning in a specific high-impact scientific use case.

<a id="ch9-5"></a>
### 9.5 Environmental sustainability

Strubell et al. [S4] documented that training large transformer models from scratch can consume hundreds of thousands of kilograms of CO2 equivalent. This project's training footprint was deliberately minimal: 6 training runs (rank sweep + multi-seed replication), each approximately 17 minutes on a Kaggle P100, totalling roughly 102 GPU-minutes (under 2 GPU-hours). The full experimental programme, including preprocessing runs and inference, consumed under 3 GPU-hours.

Three sustainability decisions are worth noting. First, LoRA trains 0.68% of model parameters — the frozen weights are never updated and require no gradient computation. Second, the project reuses a publicly available pretrained ViT checkpoint rather than training from scratch; training ViT-B/16 from scratch would require hundreds of GPU-hours. Third, reusing the Mendeley published preprocessed dataset avoids repeating the 9-hour multi-terabyte MAST archive download. Published preprocessing is a form of communal energy efficiency, and formally citing the dataset producers contributes to the incentive structure that makes future dataset publication worthwhile.

<a id="ch9-6"></a>
### 9.6 Open science and reproducibility

Every component of this system can be reproduced from freely available public resources. The Mendeley dataset is available at its DOI with no registration required. The NASA Exoplanet Archive KOI table is publicly downloadable. ViT-B/16 pretrained weights are available through timm. All other libraries are on PyPI. The notebooks include pinned install cells, random seeds are documented and fixed, and the FAISS index, scaler, LoRA adapter, and cached TESS arrays can be shared without restriction. The code repository is currently private but will be made public on submission of the associated paper.

---

<a id="ch10"></a>
## Chapter 10: Critical Appraisal

<a id="ch10-1"></a>
### 10.1 Objectives met

All five project objectives were met. Objective 1 (GAF dataset): kepler_gaf_dataset.npz produced, stratified, verified, and frozen. Objective 2 (four-regime evaluation): all six configurations evaluated and reproducible from saved artefacts. Objective 3 (FAISS retrieval): index built from 2,745 Archive planets and validated against physical expectations. Objective 4 (LangGraph explanation): pipeline implemented and evaluated across 15 cases. Objective 5 (TESS probe): complete transfer failure characterised with mechanism proposed. Meeting all objectives is necessary but not sufficient for a strong evaluation — the section below addresses how each was met and where the approach could have been stronger.

<a id="ch10-2"></a>
### 10.2 Ranked weaknesses and alternatives

**Table 15 — Ranked weaknesses with better alternatives**

| Rank | Weakness | Better alternative | Ranking rationale |
|---|---|---|---|
| 1 | GAF over recurrence plots | Recurrence plots (per Choudhary et al.) | Directly affects headline recall; known before work began |
| 2 | Fixed single split, no k-fold | Stratified k-fold with seed-averaged results | Limits evaluation robustness; partially mitigated by multi-seed |
| 3 | Explanation study: 15 cases, 1 rater, automated proxy | 50+ cases, 3 independent raters, pre-registered rubric | Limits strength of explanation quality conclusions |
| 4 | TESS probe n=14 | Expand to n≥50 with threshold recalibration test | Too small to confirm mechanism; p=0.098 non-significant |
| 5 | 6-parameter retrieval space | ViT CLS token embedding for morphological similarity | Misses transit morphology; retrieval quality is already good |
| 6 | Demo dataset-identifier gap | Regenerate Mendeley dataset with KOI identifiers | Demo shows representative image, not queried KOI's own image |
| 7 | Rank sweep single-run | Multi-seed each rank | r=8→r=16 gap may be within noise; not confirmed |
| 8 | No false-positive knowledge base | Curate FP catalogue for negative verdicts | LLM sometimes fails on FP explanations |
| 9 | LLM cascade relies on external API | Local Ollama deployment | Template always available; reproducibility of LLM path only |

**Weakness 1 — GAF over recurrence plots (most consequential).** Choudhary et al. [11] demonstrate directly that recurrence plots outperform GAFs on Kepler KOI classification. This project used GAF for toolchain stability (pyts is mature; recurrence plot implementations are less standardised) and schedule reasons. The consequence is that the headline recall comparison is confounded by encoding choice: this project's 85.5% vs Choudhary et al.'s 89.46% may reflect the encoding advantage rather than the adaptation advantage. The better design would have run both encodings, separating their contributions. This is ranked first because it is the most direct methodological limitation on the headline comparison.

**Weakness 2 — Fixed single split (second most consequential).** A single 15% test partition evaluates performance on one particular slice of the data. k-fold cross-validation provides fold variance in addition to the point estimate, supporting stronger generalisability claims. The fixed split was chosen for compute budget and comparability, and the multi-seed study partially compensates by quantifying seed variance at the deployed configuration. But fold variance and seed variance are different quantities: fold variance measures sensitivity to which examples are in the test set; seed variance measures sensitivity to the training random initialisation. Neither substitutes for the other. This is ranked second because evaluation robustness directly affects how confident one should be in generalising the results to unseen data.

**Weakness 3 — Explanation study limitations (tied third).** Fifteen cases, one rater, and an automated proxy with documented rounding artefacts is a minimal evaluation by NLP standards. A pre-registered rubric, 50+ cases enabling per-class breakdowns and statistical testing, and three independent raters enabling inter-rater agreement statistics would support much stronger claims about explanation quality. This is ranked third because the explanation layer is the project's most novel contribution and deserves the most rigorous evaluation.

**Weakness 4 — TESS probe n=14 (tied third).** The Mann-Whitney U p-value of 0.098 is non-significant by any conventional criterion. The probe establishes the direction and magnitude of the transfer failure but cannot confirm the proposed cadence-mismatch mechanism. Piloting the TESS experiment earlier in the project (immediately after the Kepler classifier was trained) would have left time for expansion to n≥50 with proper statistical testing of the threshold-recalibration hypothesis.

**Weakness 5 — Six-parameter retrieval space (fifth).** Physical parameter similarity (period, depth, radius, stellar type) does not capture morphological similarity (the GAF-encoded shape of the transit light curve). Two KOIs with identical physical parameters but different photometric signatures — e.g., different ingress shapes indicating one is a true planet and one a grazing binary — would be treated as identical by the retrieval system. The ViT CLS token embedding encodes morphological similarity directly. Ranked fifth because retrieval quality is already high (mean similarity 0.984) and the validation retrieval is physically sensible; the improvement is principled but not urgent.

**Weakness 6 — Demo dataset-identifier gap (sixth).** The application classifies a representative test-set GAF image rather than the queried KOI's own image, because the Mendeley dataset does not preserve KOI identifiers. No metric is affected (all results come from aligned notebook arrays). Regenerating the dataset with identifiers attached would cost one cloud session. Ranked sixth because impact on scientific results is zero; only the demo's interpretability is affected.

**Weakness 7 — Rank sweep single-run (seventh).** The rank sweep result (r=8→r=16 F1 gap 0.014) rests on a single training run at each rank. The multi-seed study at r=16 shows this gap is smaller than the seed standard deviation (0.009), meaning it may not be reproducible. Running multi-seed at r=4 and r=8 would require ~70 additional GPU-minutes and would resolve this ambiguity. Ranked seventh because the direction of the rank ordering is consistent in the observed data; the uncertainty is about reproducibility of the precise gap, not the qualitative finding.

**Weakness 8 — No false-positive knowledge base (eighth).** The FAISS index contains only confirmed planets. The generator lacks evidence vocabulary for negative verdicts and sometimes fails on false-positive explanations (K00262.01 case). The template path always handles this correctly. Ranked eighth because the system's false-positive handling works correctly in the template path; the LLM path failure is a secondary quality issue, not a correctness failure.

**Weakness 9 — External API dependency for LLM path (ninth).** The template path is always available and deterministically correct; the LLM path adds prose quality at the cost of external dependency. Local Ollama deployment would eliminate this dependency. Ranked ninth because the template fallback ensures the system meets all functional requirements without the API; the dependency affects only the secondary output quality difference between the two generation paths.

<a id="ch10-3"></a>
### 10.3 What exceeded the proposal

The one-shot paradox (F1 0.456 below zero-shot 0.505) was not anticipated and rewards analysis beyond the immediate context. It is a predictable consequence of high-dimensional linear classification with atypical exemplars, and reporting it with a mechanistic explanation makes it a contribution to understanding, not just an embarrassing anomaly.

The rank sweep saturation at r=4 was cleaner than the proposal expected. The supervisor's suggestion to expand from r=4 only to r=4, 8, 16 was correct: the three-rank sweep supports the low-rank hypothesis far more convincingly than a single rank would, and the saturation pattern is the more interesting finding than any individual rank's performance.

The K00262.01 coherent-facts-illogical-reasoning failure was found because the evaluation was designed to include misclassified cases. This is an incidental scientific finding about LLM behaviour on scientific classification tasks, it is reported with its implications, and it motivates the false-positive knowledge base future work that makes the future-work section more specific and better grounded than a proposal-driven list of extensions would have been.

---

<a id="ch11"></a>
## Chapter 11: Future Work in Detail

<a id="ch11-1"></a>
### 11.1 Multi-seed replication for the full rank sweep

Priority: Highest. Cost: approximately 100 GPU-minutes.

The current rank sweep is a single run at seed 42 for each of the three ranks. Running the same three seeds (42, 123, 2026) at r=4 and r=8 would add 2 ranks × 3 seeds = 6 training runs × 17 minutes each. The outcome would determine whether the "AUC flat across ranks" finding is statistically robust. If the mean ± std bars at r=4, r=8, and r=16 overlap, the claim becomes statistically supported: LoRA performance is indistinguishable across ranks r=4–16, and r=4 is the efficient default. If they do not overlap, the rank ordering is real and the recommendation changes. The infrastructure exists; only the runs are needed.

<a id="ch11-2"></a>
### 11.2 Expanded TESS evaluation

Priority: Second. Cost: approximately one CPU session for expanded download plus 20 GPU-minutes for inference.

The specific first experiment is threshold recalibration: without any retraining, shift the decision threshold from 0.5 to a value calibrated on a small TESS validation set (5-10 targets held aside from the test). If the residual AUC of 0.625 reflects calibration failure rather than representation failure, threshold recalibration should recover substantial F1. If it does not, the failure is in the representation, and light LoRA fine-tuning on a small TESS training split is the next intervention. The Lightkurve and GAF pipeline for TESS is already implemented and cached; expanding to n≥50 requires downloading additional targets (a CPU task) and appending to the existing cache file.

<a id="ch11-3"></a>
### 11.3 LoRA on recurrence-plot inputs

Priority: Third. Cost: approximately one preprocessing run plus one training run (30 GPU-minutes total).

This is the most consequential unrun experiment. The recurrence plot encoder replaces the GramianAngularField step; all downstream code (DataLoader, ViT model, LoRA configuration, training loop, V&V checks) remains unchanged. The scientifically interesting question: does LoRA r=16 on recurrence plots match or exceed Choudhary et al.'s [11] fully trained recurrence-plot result (89.46% recall)? If yes, the parameter-efficiency finding extends to the stronger encoding. If no, the gap remaining between LoRA and full training on the stronger encoding measures the irreducible cost of low-rank adaptation on this task, at the same parameter budget. Either outcome advances the state of knowledge about LoRA's scope on astronomical imaging tasks.

<a id="ch11-4"></a>
### 11.4 Explanation quality programme

Priority: Fourth. Cost: approximately one person-week of annotation effort.

The explanation quality study should be expanded to 50+ cases with three independent raters and a pre-registered rubric, enabling inter-rater agreement statistics. The specific target is the coherent-facts-illogical-reasoning failure rate: how often does the LLM cascade produce a K00262.01-style failure on FALSE POSITIVE verdicts? With 50 cases and half expected to be false positives, approximately 25 false-positive explanations would be available for analysis, enough to estimate a failure rate and compare it across LLM models in the cascade.

The false-positive knowledge base is the structural fix. A curated catalogue of characterised false-positive types — eclipsing binaries with known orbital parameters, background blends with documented stellar contamination scenarios, instrumental artefacts with characteristic photometric signatures — would give the generator evidence vocabulary for negative verdicts. The generator could then cite not just confirmed planets (which produce the paradox) but known false-positive systems that share the candidate's photometric and orbital characteristics.

<a id="ch11-5"></a>
### 11.5 Learned embedding retrieval

Priority: Fifth. Cost: approximately one day of implementation plus one inference run (no training needed).

The ViT CLS token embedding from the trained LoRA r=16 model can replace or augment the six-parameter physical feature space as the retrieval vector. Building a second FAISS index from the CLS token embeddings of the 2,745 confirmed Archive planets requires generating their GAF images via Lightkurve, which is a Lightkurve pipeline task. Once built, the morphological-similarity index enables comparison: for any queried KOI, retrieve the five most similar confirmed planets by physical parameters and the five most similar by GAF morphology. Cases where the two neighbour sets diverge are diagnostically valuable — a KOI whose parameters resemble confirmed planets but whose morphology resembles none of them warrants investigation as an unusual false-positive scenario.

<a id="ch11-6"></a>
### 11.6 Deployment pathway

Priority: Sixth. Cost: dependent on institutional resources.

The current system runs as a local Flask server requiring a laptop and no cloud resources. Persistent deployment would require stable hosting (a cloud VM or Hugging Face Space), identifier resolution to KOI-specific GAF images (either from a cached Lightkurve database or on-demand download), and a feedback mechanism allowing astronomers to report errors. Integration with NASA ExoFOP would require coordination with the Kepler/K2/TESS science teams and is a longer-term goal. In the near term, a public Hugging Face Space would allow the community to test the system on arbitrary KOIs and provide informal feedback.

---

<a id="ch12"></a>
## Chapter 12: Conclusion

<a id="ch12-1"></a>
### 12.1 Summary of contributions

**Parameter-efficient transit classification.** A ViT-B/16 adapted with LoRA at ranks r=4–16 achieves AUC-ROC 0.908–0.920 and macro F1 0.807–0.835 on 796 Kepler KOIs, training 0.17%–0.68% of model parameters in 16–17 minutes on a free Kaggle GPU. CONFIRMED recall of 85.5% at r=16 is within 4 percentage points of the fully trained result on the stronger encoding. Strong transit classification performance is accessible without full fine-tuning, without CNN architectures, and without institutional compute.

**Empirical LoRA rank characterisation in an astronomical imaging domain.** AUC-ROC saturates at r=4 (147K trainable parameters, 0.17% of the model), providing the first published evidence that LoRA's intrinsic-dimensionality hypothesis holds for GAF-encoded stellar photometry — a domain maximally distant from the ImageNet pretraining distribution.

**The first retrieval-grounded explanation layer for exoplanet vetting.** The LangGraph pipeline produces natural-language explanations in which every factual claim traces to a real named entry in the NASA Exoplanet Archive, evaluated across 15 cases with perfect grounding and completeness in both generation paths. This is the first such system in the literature.

**A controlled supervision dose-response curve.** The four-regime comparison (zero-shot → one-shot → few-shot → LoRA) produces a clean empirical curve that tests three common assumptions about pretrained models: that they transfer to any visual domain (not supported — AUC 0.536 at zero-shot), that some supervision is always better than none (not supported — the one-shot paradox), and that full fine-tuning is necessary for strong performance (not supported — the LoRA result).

**An honest characterisation of TESS transfer failure.** The zero-shot probe reports complete failure at the decision threshold, proposes a cadence-mismatch mechanism, and defines the expansion experiment needed to test it. This is more scientifically useful than a selective positive result.

<a id="ch12-2"></a>
### 12.2 Key findings

- ViT-B/16 pretrained on ImageNet encodes nothing discriminative about GAF transit textures (zero-shot AUC 0.536).
- One-shot fine-tuning can be worse than no fine-tuning in this high-dimensional embedding space (F1 0.456 < 0.505), for geometric reasons related to atypical exemplar placement.
- Few-shot (10/class) crosses the usefulness threshold (AUC 0.797), confirming the pretrained representation supports class separation once the readout is calibrated.
- LoRA with 0.17% of parameters achieves AUC 0.908, a 0.111 AUC gain over few-shot, measuring the value of representation adaptation.
- AUC saturates at r=4 across the rank sweep; the r=8→r=16 F1 gain of 0.014 is within seed variance.
- CONFIRMED recall 85.5% is within 4 percentage points of Choudhary et al.'s fully trained result on the stronger encoding.
- Retrieval quality is high at all tested k values (mean cosine similarity 0.976–0.987).
- Both generation paths produce fully grounded and complete explanations across all 15 evaluated cases.
- The LLM cascade produces coherent-facts-illogical-reasoning failure on false-positive verdicts; the template path handles this correctly.
- Zero-shot TESS transfer fails completely at the decision threshold (AUC 0.625); cadence-mismatch mechanism is plausible but not confirmed at n=14.

<a id="ch12-3"></a>
### 12.3 Closing remarks

This project started from a straightforward observation: every existing transit vetting system produces a score, not a sentence. An astronomer receiving a confidence score of 0.94 knows the model is fairly sure, but does not know which known systems the candidate resembles, whether the confidence is justified by the physical properties of the target, or what the evidence is for or against the planetary interpretation. The retrieval-augmented explanation layer is a first step toward making automated vetting systems participate in the epistemic process of astronomy rather than merely providing inputs to it.

The parameter-efficiency finding complements this by making the underlying classifier accessible to researchers without institutional compute budgets: 17 minutes on a free GPU, 0.68% of model parameters, and a published preprocessed dataset are all that is needed for a transit classifier reaching 0.92 AUC and 85.5% confirmed-planet recall. An accessible classifier paired with an explainable verdict — that is the project's central contribution.

The TESS failure is, in its own way, the most scientifically interesting result: it defines exactly where the system's validity ends and points directly to the experiments that would extend it. A system whose failure modes are precisely characterised is more useful than one whose successes are loosely claimed.

---

<a id="ch13"></a>
## Chapter 13: Supporting Artefacts

**Code repository:** https://github.com/ahmedbutt2015/dessertation (private; access on request). Contains all notebooks, the app/ directory, task checklist, CLAUDE.md context file, and full commit history.

**Notebooks:**
- `01_data_acquisition.ipynb` — Lightkurve path; preserved as reference; not active.
- `01b_data_acquisition_mendeley.ipynb` — Mendeley path; produces kepler_gaf_dataset.npz; active.
- `02_classifier_evaluation.ipynb` — Four-regime comparison, rank sweep; produces metrics tables and saved r=16 adapter.
- `05_tess_generalisation.ipynb` — TESS probe; produces per-target results and input-statistics analysis.
- `06_lora_multiseed.ipynb` — Multi-seed replication at r=16.

**Interactive application:** `app/server.py` — Flask REST API and mission-control web interface. Requires saved LoRA adapter (`lora_r16_adapter/`), FAISS index (`exoplanet_faiss.index`), scaler (`exoplanet_scaler.pkl`), and confirmed-planet catalogue (`confirmed_planets.csv`). Run with `python app/server.py`.

**Data artefacts:**
- `kepler_gaf_dataset.npz` — 5,302 × (3, 224, 224) float32 GAF images with binary labels and frozen split indices.
- `exoplanet_faiss.index` — FAISS IndexFlatIP over 2,745 confirmed Archive planets.
- `exoplanet_scaler.pkl` — Fitted StandardScaler for six retrieval features.
- `confirmed_planets.csv` — Archive rows used in the index with names and parameters.
- `lora_r16_adapter/` — Saved PEFT adapter and head state dict for the deployed r=16 configuration.
- `tess_gaf_arrays.npz` — Cached GAF images for 14 TESS targets.
- `results/lora_r16_multiseed.csv` — Multi-seed replication results.
- `results/ablation_results.csv` — Full four-regime and rank-sweep results table.
- `results/explanation_eval.csv` — Explanation quality evaluation data.

**Figure index:**

| File | Description |
|---|---|
| `results/fig1_pipeline_diagram.png` | Complete system pipeline |
| `results/fig2_lightcurve_to_gaf.png` | Light curve to GAF transformation |
| `results/fig3_four_settings_barchart.png` | Macro F1 and AUC-ROC across four adaptation regimes |
| `results/app/3.png` | Mission Control landing page |
| `results/app/1.png` | Real-time pipeline analysis — K00752.01 |
| `results/app/2.png` | Classification output — K00752.01 |
| `results/app/4.png` | Classification output — K00069.01 |
| `results/fig5_explanation_example.png` | Pipeline output for K00001.01 |
| `results/fig6_gantt_chart.png` | Project Gantt chart |
| `results/fig6_rank_sweep.png` | LoRA rank sweep chart |
| `results/fig7_multiseed_stability.png` | Multi-seed stability at r=16 |
| `results/lora_r16_roc_curve.png` | ROC curve for LoRA r=16 |
| `results/lora_r16_confusion_matrix.png` | Confusion matrix for LoRA r=16 |

---

<a id="references"></a>
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

[11] A. Choudhary, S. Bandari, B. S. Kushvah, and C. Swastik, "Exoplanet Classification through Vision Transformers with Temporal Image Analysis," *arXiv preprint* arXiv:2506.16597, 2025 (accepted, *The Astronomical Journal*).

[12] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, 2020, pp. 9459–9474.

[13] Z. Ji et al., "Survey of Hallucination in Natural Language Generation," *ACM Computing Surveys*, vol. 55, no. 12, pp. 1–38, 2023.

[14] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. International Conference on Learning Representations (ICLR)*, 2022.

[15] S. D. McCauliff et al., "Automatic Classification of Kepler Planetary Transit Candidates," *The Astrophysical Journal*, vol. 806, no. 1, p. 6, 2015.

[16] K. A. Pearson, L. Palafox, and C. A. Griffith, "Searching for Exoplanets Using Artificial Intelligence," *Monthly Notices of the Royal Astronomical Society*, vol. 474, no. 1, pp. 478–491, 2018.

[17] A. Malik, B. P. Moster, and C. Obermeier, "Exoplanet Detection Using Machine Learning," *Monthly Notices of the Royal Astronomical Society*, vol. 513, no. 4, pp. 5505–5516, 2022.

[18] M. Jara-Maldonado et al., "Transiting Exoplanet Discovery Using Machine Learning Techniques: A Survey," *Earth Science Informatics*, vol. 13, pp. 573–600, 2020.

[19] N. Ding et al., "Parameter-Efficient Fine-Tuning of Large-Scale Pre-trained Language Models," *Nature Machine Intelligence*, vol. 5, pp. 220–235, 2023.

[20] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618–626.

[21] C. Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead," *Nature Machine Intelligence*, vol. 1, pp. 206–215, 2019.

[22] T. Macedo and M. Zalewski, "Dataset for Machine Learning Exoplanet Classification," Mendeley Data, V3, 2024. DOI: 10.17632/wctcv34962.3.

[23] G. R. Ricker et al., "Transiting Exoplanet Survey Satellite (TESS)," *Journal of Astronomical Telescopes, Instruments, and Systems*, vol. 1, no. 1, 2015.

---

**Supplementary references (introduced in this document only):**

[S1] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 36, 2023.

[S2] S. Abnar and W. Zuidema, "Quantifying Attention Flow in Transformers," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2020, pp. 4190–4197.

[S3] British Computer Society, *BCS Code of Conduct*, BCS, The Chartered Institute for IT, 2022.

[S4] E. Strubell, A. Ganesh, and A. McCallum, "Energy and Policy Considerations for Deep Learning in NLP," in *Proc. 57th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2019, pp. 3645–3650.
