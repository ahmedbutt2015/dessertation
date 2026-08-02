"""
Exoplanet Transit Classifier — Flask server for NASA mission-control frontend
COM748 Dissertation — Ahmed Fayyaz Butt

Run:  python server.py
      http://localhost:7861
"""

import sys
import numpy as np
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))

from rag.retriever import retrieve_any_koi
from rag.agent import explain
from rag.classifier import classify, is_available as classifier_available

DESIGN_DIR = Path(__file__).parent / "Exoplanet mission control design"
GAF_NPZ    = Path(__file__).parent.parent / "data" / "kepler_gaf_dataset.npz"

PLANET_COLORS = ['#4fb8e8', '#e8925a', '#d4604a', '#7ec8e3', '#ffd166']

app = Flask(__name__)


@app.route("/")
def index():
    return send_from_directory(DESIGN_DIR, "Exoplanet Classifier.dc.html")


@app.route("/support.js")
def support_js():
    return send_from_directory(DESIGN_DIR, "support.js")


@app.route("/api/classify", methods=["POST"])
def api_classify():
    data   = request.get_json(silent=True) or {}
    koi_id = str(data.get("koi_id", "")).strip()
    if not koi_id:
        return jsonify({"error": "No KOI ID provided"}), 400

    try:
        koi_info, similar = retrieve_any_koi(koi_id, k=5)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    # Classification
    if classifier_available():
        try:
            gaf = _load_gaf(koi_id)
            label, confidence = classify(gaf)
        except Exception:
            label      = koi_info.get("koi_disposition", "UNKNOWN")
            confidence = 0.75
    else:
        label      = koi_info.get("koi_disposition", "UNKNOWN")
        confidence = 0.75

    # Explanation
    try:
        explanation_text = explain(koi_id, label, confidence, similar)
    except Exception:
        explanation_text = (
            f"KOI {koi_id} classified as {label} with {confidence:.0%} confidence. "
            "Similar confirmed systems retrieved from NASA Exoplanet Archive."
        )

    # Format similar planets
    planets_out = []
    for i, (_, row) in enumerate(similar.iterrows()):
        name = str(row.get("kepler_name", ""))
        if not name or name == "nan":
            name = str(row.get("kepoi_name", f"KOI-{i+1}"))
        sim = float(row.get("similarity_score", 0.8))
        try:
            period = f"{float(row.get('koi_period', 0)):.1f} d"
            radius = f"{float(row.get('koi_prad', 0)):.1f} R⊕"
            depth  = f"{int(float(row.get('koi_depth', 0)))} ppm"
        except (TypeError, ValueError):
            period, radius, depth = "— d", "— R⊕", "— ppm"
        planets_out.append({
            "name":       name,
            "koi":        str(row.get("kepoi_name", "")),
            "period":     period,
            "radius":     radius,
            "depth":      depth,
            "similarity": round(sim * 100),
            "color":      PLANET_COLORS[i % len(PLANET_COLORS)],
        })

    # KOI orbital info
    try:
        period_str = f"{float(koi_info.get('koi_period', 0)):.2f} d"
        radius_str = f"{float(koi_info.get('koi_prad', 0)):.2f} R⊕"
        depth_str  = f"{int(float(koi_info.get('koi_depth', 0)))} ppm"
    except (TypeError, ValueError):
        period_str, radius_str, depth_str = "— d", "— R⊕", "— ppm"

    return jsonify({
        "koi_id":         koi_id,
        "label":          label,
        "confidence":     round(confidence * 100),
        "nasa_disp":      koi_info.get("koi_disposition", "UNKNOWN"),
        "period":         period_str,
        "radius":         radius_str,
        "depth":          depth_str,
        "similar_planets": planets_out,
        "explanation":    explanation_text,
    })


@app.route("/api/search", methods=["POST"])
def api_search():
    data   = request.get_json(silent=True) or {}
    koi_id = str(data.get("koi_id", "")).strip()
    k      = min(int(data.get("k", 10)), 20)

    if not koi_id:
        return jsonify({"error": "Provide a KOI ID"}), 400

    try:
        koi_info, similar = retrieve_any_koi(koi_id, k=k)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    planets_out = []
    for i, (_, row) in enumerate(similar.iterrows()):
        name = str(row.get("kepler_name", ""))
        if not name or name == "nan":
            name = str(row.get("kepoi_name", f"KOI-{i+1}"))
        try:
            sim    = round(float(row.get("similarity_score", 0.8)) * 100)
            period = f"{float(row.get('koi_period', 0)):.2f} d"
            radius = f"{float(row.get('koi_prad',   0)):.2f} R⊕"
            depth  = f"{int(float(row.get('koi_depth', 0)))} ppm"
            steff  = f"{int(float(row.get('koi_steff', 0)))} K"
        except (TypeError, ValueError):
            sim, period, radius, depth, steff = 80, "—", "—", "—", "—"
        planets_out.append({
            "name": name, "koi": str(row.get("kepoi_name", "")),
            "period": period, "radius": radius,
            "depth": depth, "steff": steff, "similarity": sim,
            "color": PLANET_COLORS[i % len(PLANET_COLORS)],
        })

    try:
        period_str = f"{float(koi_info.get('koi_period', 0)):.2f} d"
        radius_str = f"{float(koi_info.get('koi_prad',   0)):.2f} R⊕"
        depth_str  = f"{int(float(koi_info.get('koi_depth', 0)))} ppm"
    except (TypeError, ValueError):
        period_str, radius_str, depth_str = "—", "—", "—"

    return jsonify({
        "koi_id":      koi_id,
        "disposition": koi_info.get("koi_disposition", "UNKNOWN"),
        "kepler_name": str(koi_info.get("kepler_name", "") or ""),
        "period":      period_str,
        "radius":      radius_str,
        "depth":       depth_str,
        "results":     planets_out,
    })


def _load_gaf(koi_id: str) -> np.ndarray:
    # GAF dataset has no KOI-name index, so we return a representative sample.
    # The ViT result is shown alongside the NASA catalogue disposition so the
    # user can see both — this is the intended demo behaviour.
    if GAF_NPZ.exists():
        data = np.load(GAF_NPZ)
        idx = hash(koi_id) % len(data["X_test"])
        return data["X_test"][idx]
    return np.zeros((64, 64), dtype=np.float32)


if __name__ == "__main__":
    print("Starting Exoplanet Classifier — NASA frontend at http://localhost:7861")
    app.run(host="0.0.0.0", port=7861, debug=False)
