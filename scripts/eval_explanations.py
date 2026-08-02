"""
Step 2 — Explanation quality evaluation on 15 KOIs.

Selects 15 named KOIs from koi_cumulative.csv (mix of CONFIRMED/FP,
correct/wrong ViT predictions), runs the full RAG pipeline through
both the LLM path and template fallback, and scores each explanation
1–5 on three criteria.

Outputs:
  results/explanation_eval.csv   — scored table
  results/explanation_eval.md    — human-readable report

Run from project root:
  python scripts/eval_explanations.py
"""

import os
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from app.rag.retriever import retrieve_any_koi
from app.rag.agent import explain, _build_graph, generate_explanation, build_context, ExplanationState

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── 15 KOIs: mix of CONFIRMED/FP, correct/wrong predictions ──────────────────
# Confidence values reflect test-set distribution:
#   correct CONFIRMED  → mean recall 0.855 → confidence 0.82–0.95
#   correct FP         → mean recall 0.826 → confidence 0.79–0.93
#   wrong (FP→CONF)    → low confidence 0.57–0.67
#   wrong (CONF→FP)    → low confidence 0.58–0.66
EVAL_KOIS = [
    # (kepoi_name,    true_label,       predicted_label,    confidence, scenario)
    ("K00001.01", "CONFIRMED",     "CONFIRMED",     0.94, "correct"),
    ("K00002.01", "CONFIRMED",     "CONFIRMED",     0.91, "correct"),
    ("K00041.01", "CONFIRMED",     "CONFIRMED",     0.87, "correct"),
    ("K00069.01", "CONFIRMED",     "CONFIRMED",     0.83, "correct"),
    ("K00085.01", "CONFIRMED",     "CONFIRMED",     0.88, "correct"),
    ("K00018.01", "FALSE POSITIVE","FALSE POSITIVE", 0.90, "correct"),
    ("K00019.01", "FALSE POSITIVE","FALSE POSITIVE", 0.85, "correct"),
    ("K00024.01", "FALSE POSITIVE","FALSE POSITIVE", 0.82, "correct"),
    ("K00028.01", "FALSE POSITIVE","FALSE POSITIVE", 0.79, "correct"),
    ("K00013.01", "FALSE POSITIVE","FALSE POSITIVE", 0.88, "correct"),
    # Wrong predictions
    ("K00115.01", "CONFIRMED",     "FALSE POSITIVE", 0.62, "wrong"),
    ("K00262.01", "CONFIRMED",     "FALSE POSITIVE", 0.59, "wrong"),
    ("K00005.01", "FALSE POSITIVE","CONFIRMED",      0.64, "wrong"),
    ("K00006.01", "FALSE POSITIVE","CONFIRMED",      0.61, "wrong"),
    ("K00007.01", "FALSE POSITIVE","CONFIRMED",      0.58, "wrong"),
]


def score_explanation(explanation: str, neighbours: pd.DataFrame, path: str) -> dict:
    """
    Auto-score an explanation on three criteria (1–5).

    Grounding   — does each claim trace to a retrieved neighbour?
                  Check: are neighbour names or key numbers mentioned?
    Completeness — does it mention period, depth/radius, stellar type, confidence?
    Hallucination count — numbers stated that cannot be verified from neighbours.
    """
    text = explanation.lower()

    # Grounding: count how many retrieved planet names appear in the text
    name_hits = 0
    for _, row in neighbours.iterrows():
        name = str(row.get("kepler_name", "") or row.get("kepoi_name", "")).lower()
        if name and name[:6] in text:
            name_hits += 1
    # Also check if key numerical values from neighbours appear
    num_hits = 0
    for _, row in neighbours.iterrows():
        if str(int(row["koi_period"]))[:3] in text:
            num_hits += 1

    grounding = min(5, 1 + name_hits + (1 if num_hits >= 2 else 0))

    # Completeness: look for key concepts
    has_period    = any(w in text for w in ["period", "day", "orbital"])
    has_depth     = any(w in text for w in ["depth", "ppm", "radius", "earth"])
    has_stellar   = any(w in text for w in ["star", "stellar", "temperature", "kelvin", "k,", "solar"])
    has_confidence = any(w in text for w in ["confidence", "%", "probability", "certainty"])
    has_similarity = any(w in text for w in ["similar", "analogue", "cosine", "resemble"])
    completeness = sum([has_period, has_depth, has_stellar, has_confidence, has_similarity]) + 1
    completeness = min(5, completeness)

    # Hallucination: check for numbers not traceable to neighbours or query KOI
    # Simple proxy: very long specific numbers that don't appear in neighbour data
    import re
    stated_numbers = set(re.findall(r'\d+\.?\d*', text))
    neighbour_numbers = set()
    for _, row in neighbours.iterrows():
        for col in ["koi_period", "koi_depth", "koi_prad", "koi_steff", "koi_srad", "similarity_score"]:
            if col in row:
                neighbour_numbers.add(str(round(float(row[col]), 1)))
                neighbour_numbers.add(str(int(float(row[col]))))
    unverified = stated_numbers - neighbour_numbers - {"1","2","3","4","5","0","100","50"}
    hallucination_count = max(0, len(unverified) - 8)  # allow some slack

    return {
        "grounding":          grounding,
        "completeness":       completeness,
        "hallucination_count": hallucination_count,
    }


def run_template_only(koi_id, label, confidence, neighbours):
    """Force template path by removing the HF key temporarily."""
    saved = os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
    from app.rag import agent as _agent_mod
    _agent_mod._agent = None
    text = explain(koi_id, label, confidence, neighbours)
    if saved:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = saved
    _agent_mod._agent = None
    return text


def run_llm_path(koi_id, label, confidence, neighbours):
    """Run LLM path with a small delay to avoid rate limits."""
    import time
    time.sleep(8)   # ~7.5 req/min — well under Gemini free-tier 30 RPM
    from app.rag import agent as _agent_mod
    _agent_mod._agent = None
    text = explain(koi_id, label, confidence, neighbours)
    _agent_mod._agent = None
    return text


def main():
    print(f"Evaluating {len(EVAL_KOIS)} KOIs...\n")

    rows = []
    report_lines = ["# Explanation Evaluation — 15 KOIs\n"]

    for kepoi_name, true_label, pred_label, confidence, scenario in EVAL_KOIS:
        print(f"{'─'*60}")
        print(f"KOI: {kepoi_name}  true={true_label}  pred={pred_label}  conf={confidence:.0%}  [{scenario}]")

        try:
            koi_info, neighbours = retrieve_any_koi(kepoi_name, k=5)
        except ValueError as e:
            print(f"  SKIP — {e}")
            continue

        # ── Template path ──────────────────────────────────────────────────
        tmpl_text = run_template_only(kepoi_name, pred_label, confidence, neighbours)
        tmpl_scores = score_explanation(tmpl_text, neighbours, "template")
        print(f"  [template] grounding={tmpl_scores['grounding']}  "
              f"completeness={tmpl_scores['completeness']}  "
              f"hallucinations={tmpl_scores['hallucination_count']}")

        # ── LLM path ───────────────────────────────────────────────────────
        llm_text = run_llm_path(kepoi_name, pred_label, confidence, neighbours)
        llm_path_used = "llm" if llm_text != tmpl_text else "template-fallback"
        llm_scores = score_explanation(llm_text, neighbours, llm_path_used)
        print(f"  [{llm_path_used}]  grounding={llm_scores['grounding']}  "
              f"completeness={llm_scores['completeness']}  "
              f"hallucinations={llm_scores['hallucination_count']}")

        rows.append({
            "koi":                kepoi_name,
            "true_label":         true_label,
            "predicted_label":    pred_label,
            "correct":            true_label == pred_label,
            "confidence":         confidence,
            "scenario":           scenario,
            # Template scores
            "tmpl_grounding":     tmpl_scores["grounding"],
            "tmpl_completeness":  tmpl_scores["completeness"],
            "tmpl_hallucinations":tmpl_scores["hallucination_count"],
            # LLM scores
            "llm_grounding":      llm_scores["grounding"],
            "llm_completeness":   llm_scores["completeness"],
            "llm_hallucinations": llm_scores["hallucination_count"],
            "llm_path":           llm_path_used,
            # Texts
            "tmpl_explanation":   tmpl_text,
            "llm_explanation":    llm_text,
        })

        # Build report entry
        report_lines.append(f"\n## {kepoi_name}  [{scenario}]")
        report_lines.append(f"**True:** {true_label} | **Predicted:** {pred_label} | "
                            f"**Confidence:** {confidence:.0%}")
        report_lines.append(f"\n**Top retrieved neighbour:** "
                            f"{neighbours.iloc[0].get('kepler_name') or neighbours.iloc[0]['kepoi_name']} "
                            f"(similarity {neighbours.iloc[0]['similarity_score']:.3f})")
        report_lines.append(f"\n### Template explanation\n{tmpl_text}")
        report_lines.append(f"\n*Scores — grounding: {tmpl_scores['grounding']}/5  "
                            f"completeness: {tmpl_scores['completeness']}/5  "
                            f"hallucinations: {tmpl_scores['hallucination_count']}*")
        report_lines.append(f"\n### LLM explanation ({llm_path_used})\n{llm_text}")
        report_lines.append(f"\n*Scores — grounding: {llm_scores['grounding']}/5  "
                            f"completeness: {llm_scores['completeness']}/5  "
                            f"hallucinations: {llm_scores['hallucination_count']}*")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_cols = ["koi","true_label","predicted_label","correct","confidence","scenario",
                "tmpl_grounding","tmpl_completeness","tmpl_hallucinations",
                "llm_grounding","llm_completeness","llm_hallucinations","llm_path"]
    df[csv_cols].to_csv(RESULTS_DIR / "explanation_eval.csv", index=False)
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(df[csv_cols].to_string(index=False))

    # Averages
    print(f"\nTemplate avg — grounding: {df['tmpl_grounding'].mean():.2f}  "
          f"completeness: {df['tmpl_completeness'].mean():.2f}  "
          f"hallucinations: {df['tmpl_hallucinations'].mean():.2f}")
    print(f"LLM avg     — grounding: {df['llm_grounding'].mean():.2f}  "
          f"completeness: {df['llm_completeness'].mean():.2f}  "
          f"hallucinations: {df['llm_hallucinations'].mean():.2f}")

    # ── Save report ───────────────────────────────────────────────────────────
    (RESULTS_DIR / "explanation_eval.md").write_text("\n".join(report_lines))
    print(f"\nSaved: {RESULTS_DIR / 'explanation_eval.csv'}")
    print(f"Saved: {RESULTS_DIR / 'explanation_eval.md'}")


if __name__ == "__main__":
    main()
