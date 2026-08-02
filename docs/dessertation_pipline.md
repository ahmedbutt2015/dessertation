# Dissertation Pipeline: Fine-Tuning vs RAG vs Agentic Systems

## Objective
To evaluate and compare different AI paradigms:
- Base models
- Fine-tuning (Full vs LoRA)
- Retrieval-Augmented Generation (RAG)
- Advanced reasoning systems (Agentic RAG)

---

## Phase 1: Baseline Model Evaluation

### Goal
Establish baseline performance of pretrained models.

### Steps
- Select 2–3 pretrained models
- Run on dataset without modification
- Evaluate performance

### Metrics
- Accuracy / F1 Score (classification)
- IoU / Dice Score (segmentation)
- Latency / inference time

---

## Phase 2: Fine-Tuning vs LoRA

### Goal
Compare parameter-efficient vs full fine-tuning

### Steps
- Fine-tune models on dataset
- Apply LoRA (Low-Rank Adaptation)
- Train using same dataset

### Compare
- Performance improvement
- Training cost
- Memory usage
- Inference speed

---

## Phase 3: RAG vs Fine-Tuning

### Goal
Compare retrieval-based vs parametric learning

### Steps
- Build RAG pipeline:
  - Chunk data
  - Create embeddings
  - Store in vector DB
- Query system using retrieval

### Compare
- Fine-tuned model vs RAG system
- Accuracy vs flexibility
- Hallucination reduction

---

## Phase 4: Advanced System (Choose ONE)

### Option A: Agentic RAG (Recommended)
- Multi-step reasoning
- Tool usage
- Feedback loops

### Option B: Graph RAG
- Knowledge graph construction
- Structured retrieval

---

## Phase 5: Ablation Study

### Goal
Understand impact of each component

### Experiments
- Without fine-tuning
- Without RAG
- Different LoRA ranks
- Different chunk sizes

---

## Final Deliverables
- Model comparison results
- Efficiency vs performance trade-offs
- System architecture
- Real-world applicability analysis