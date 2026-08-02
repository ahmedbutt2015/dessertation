"""
LangGraph agentic explanation pipeline.

Architecture (3-node graph):
  build_context  →  generate_explanation  →  END

The graph takes a classification result + retrieved similar planets
and produces a natural language scientific explanation.

LLM priority:
  1. HuggingFace Inference API (HUGGINGFACEHUB_API_TOKEN) — tries Qwen2.5-7B → Llama-3.2-3B → Zephyr-7B
  2. Template-based fallback (no API required)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

# Load .env from project root if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env", override=False)
except ImportError:
    pass

import pandas as pd
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ExplanationState(TypedDict):
    koi_id: str
    label: str
    confidence: float
    similar_planets: pd.DataFrame
    context: str
    explanation: str


# ---------------------------------------------------------------------------
# Node 1 — build_context
# Formats the retrieved neighbours into a structured evidence string.
# ---------------------------------------------------------------------------

def build_context(state: ExplanationState) -> ExplanationState:
    planets = state["similar_planets"]
    lines = []
    for i, (_, row) in enumerate(planets.iterrows(), 1):
        name = (
            str(row["kepler_name"])
            if "kepler_name" in row and pd.notna(row["kepler_name"])
            else row["kepoi_name"]
        )
        sim = row.get("similarity_score", 0.0)
        lines.append(
            f"  {i}. {name}: period={row['koi_period']:.2f}d, "
            f"depth={row['koi_depth']:.0f}ppm, "
            f"planet_radius={row['koi_prad']:.2f}Re, "
            f"star_temp={row['koi_steff']:.0f}K, "
            f"cosine_similarity={sim:.3f}"
        )
    state["context"] = "\n".join(lines)
    return state


# ---------------------------------------------------------------------------
# Node 2 — generate_explanation
# Calls Claude API if key present, else returns rich template.
# ---------------------------------------------------------------------------

def generate_explanation(state: ExplanationState) -> ExplanationState:
    system_prompt = (
        "You are an expert exoplanet astronomer reviewing machine learning "
        "classification results from NASA Kepler light curves. "
        "Write a concise 3–4 sentence scientific justification for the "
        "classification, citing the physical similarities to known confirmed "
        "systems. Be precise — mention specific orbital periods, transit depths, "
        "or planetary radii. Do not speculate beyond what the data shows."
    )
    human_prompt = (
        f"KOI: {state['koi_id']}\n"
        f"Classification: {state['label']} "
        f"(model confidence: {state['confidence']:.0%})\n\n"
        f"Top 5 most similar confirmed Kepler planets (by cosine similarity "
        f"of orbital and stellar parameters):\n"
        f"{state['context']}\n\n"
        "Generate a scientific explanation for this classification."
    )

    # ── 1. HuggingFace Inference API ─────────────────────────────────────────
    hf_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip()
    if hf_key:
        from huggingface_hub import InferenceClient
        _candidates = [
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "HuggingFaceH4/zephyr-7b-beta",
        ]
        for _model in _candidates:
            try:
                client = InferenceClient(model=_model, token=hf_key)
                response = client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": human_prompt},
                    ],
                    max_tokens=400,
                    temperature=0.3,
                )
                state["explanation"] = response.choices[0].message.content
                return state
            except Exception as _e:
                print(f"    [HF {_model.split('/')[1]}] {type(_e).__name__}: {str(_e)[:80]}")

    # ── 2. Template fallback (no API required) ────────────────────────────────
    state["explanation"] = _template_explanation(
        state["koi_id"],
        state["label"],
        state["confidence"],
        state["similar_planets"],
    )
    return state


# ---------------------------------------------------------------------------
# Template fallback — no LLM required
# ---------------------------------------------------------------------------

def _template_explanation(
    koi_id: str,
    label: str,
    confidence: float,
    similar_planets: pd.DataFrame,
) -> str:
    top = similar_planets.iloc[0]
    top_name = (
        str(top["kepler_name"])
        if "kepler_name" in top and pd.notna(top["kepler_name"])
        else top["kepoi_name"]
    )
    names = ", ".join(
        str(r["kepler_name"])
        if "kepler_name" in r and pd.notna(r["kepler_name"])
        else r["kepoi_name"]
        for _, r in similar_planets.iterrows()
    )
    avg_sim = similar_planets["similarity_score"].mean() if "similarity_score" in similar_planets.columns else 0.0

    if label == "CONFIRMED":
        return (
            f"{koi_id} is classified as a CONFIRMED exoplanet candidate "
            f"(ViT-B/16 + LoRA r=16 confidence: {confidence:.0%}). "
            f"Its transit signal most closely resembles {top_name}: "
            f"period {top['koi_period']:.2f} days, transit depth {top['koi_depth']:.0f} ppm, "
            f"planet radius {top['koi_prad']:.2f} Earth radii "
            f"(cosine similarity {top.get('similarity_score', 0.0):.3f}). "
            f"The mean cosine similarity across the top-5 confirmed analogues is {avg_sim:.3f}, "
            f"strongly supporting a genuine planetary transit interpretation. "
            f"Retrieved confirmed systems: {names}."
        )
    else:
        return (
            f"{koi_id} is classified as a FALSE POSITIVE "
            f"(ViT-B/16 + LoRA r=16 confidence: {confidence:.0%}). "
            f"The transit morphology is inconsistent with a genuine planetary signal — "
            f"likely caused by an eclipsing binary, background star contamination, "
            f"or a systematic instrumental artefact. "
            f"Although the closest confirmed systems by orbital parameters are "
            f"{names} (mean similarity {avg_sim:.3f}), "
            f"the GAF image structure does not match confirmed planet transit patterns."
        )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> object:
    graph = StateGraph(ExplanationState)
    graph.add_node("build_context", build_context)
    graph.add_node("generate_explanation", generate_explanation)
    graph.set_entry_point("build_context")
    graph.add_edge("build_context", "generate_explanation")
    graph.add_edge("generate_explanation", END)
    return graph.compile()


_agent = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain(
    koi_id: str,
    label: str,
    confidence: float,
    similar_planets: pd.DataFrame,
) -> str:
    """
    Run the LangGraph agent and return a natural language explanation.

    Parameters
    ----------
    koi_id         : e.g. 'K00001.01'
    label          : 'CONFIRMED' or 'FALSE POSITIVE'
    confidence     : model confidence 0–1
    similar_planets: DataFrame from retriever.retrieve()

    Returns
    -------
    explanation : multi-sentence string
    """
    global _agent
    if _agent is None:
        _agent = _build_graph()

    result = _agent.invoke({
        "koi_id": koi_id,
        "label": label,
        "confidence": confidence,
        "similar_planets": similar_planets,
        "context": "",
        "explanation": "",
    })
    return result["explanation"]
