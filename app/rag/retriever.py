"""
FAISS retrieval module.
Loads the pre-built index and returns k most similar confirmed planets.
"""

import faiss
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent.parent / 'models' / 'rag'
DATA_DIR   = Path(__file__).parent.parent.parent / 'data' / 'Dataset_Machine_Learning_Exoplanets_2024'

FEATURES = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_steff', 'koi_srad']

_index    = None
_metadata = None   # confirmed planets only (FAISS index rows)
_scaler   = None
_all_kois = None   # full cumulative KOI table — all dispositions


def _load():
    global _index, _metadata, _scaler, _all_kois
    if _index is not None:
        return
    _index    = faiss.read_index(str(MODELS_DIR / 'faiss_index.bin'))
    _metadata = pd.read_csv(MODELS_DIR / 'confirmed_planets.csv', index_col=0)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        _scaler = pickle.load(f)
    # Full cumulative KOI table — used to look up ANY KOI's disposition + params
    cumulative = DATA_DIR / 'koi_cumulative.csv'
    if cumulative.exists():
        _all_kois = pd.read_csv(cumulative, comment='#')


def retrieve(query: dict, k: int = 5) -> pd.DataFrame:
    """
    Find the k most similar confirmed planets to a query KOI.

    Parameters
    ----------
    query : dict with keys matching FEATURES
        e.g. {'koi_period': 11.2, 'koi_duration': 2.5, 'koi_depth': 840,
               'koi_prad': 1.6, 'koi_steff': 5200, 'koi_srad': 0.95}
    k : number of results to return (default 5)

    Returns
    -------
    DataFrame with columns: kepoi_name, kepler_name, similarity_score,
                             koi_period, koi_duration, koi_depth, koi_prad,
                             koi_steff, koi_srad, koi_teq
    """
    _load()

    vec = np.array([[query[f] for f in FEATURES]], dtype=np.float32)
    vec_scaled = _scaler.transform(vec).astype(np.float32)

    norm = np.linalg.norm(vec_scaled, axis=1, keepdims=True)
    vec_norm = (vec_scaled / (norm + 1e-8)).astype(np.float32)

    D, I = _index.search(vec_norm, k=k)

    results = _metadata.iloc[I[0]].copy()
    results.insert(0, 'similarity_score', D[0].round(4))
    results = results.reset_index(drop=True)

    return results


def retrieve_by_koi(kepoi_name: str, k: int = 5) -> pd.DataFrame:
    """
    Look up a KOI by name and return its k most similar confirmed neighbours.
    The query KOI itself is excluded from results.
    Works for CONFIRMED KOIs only (they are in the FAISS index).
    For any KOI regardless of disposition, use retrieve_any_koi().
    """
    _load()

    mask = _metadata['kepoi_name'] == kepoi_name
    if not mask.any():
        raise ValueError(f'KOI {kepoi_name!r} not found in confirmed planets index.')

    idx = _metadata[mask].index[0]
    query = {f: _metadata.loc[idx, f] for f in FEATURES}

    results = retrieve(query, k=k + 1)
    results = results[results['kepoi_name'] != kepoi_name].head(k)
    return results.reset_index(drop=True)


def lookup_koi(kepoi_name: str) -> dict | None:
    """
    Return basic info for any KOI from the cumulative table.
    Returns dict with keys: kepoi_name, kepler_name, koi_disposition,
    koi_period, koi_duration, koi_depth, koi_prad, koi_steff, koi_srad.
    Returns None if KOI not found.
    """
    _load()
    if _all_kois is None:
        return None
    mask = _all_kois['kepoi_name'] == kepoi_name
    if not mask.any():
        return None
    row = _all_kois[mask].iloc[0]
    return row.to_dict()


def retrieve_any_koi(kepoi_name: str, k: int = 5) -> tuple[dict | None, pd.DataFrame]:
    """
    Works for ANY KOI — confirmed, false positive, or candidate.
    Looks up the KOI's physical parameters from the cumulative table,
    then retrieves k most similar confirmed planets from FAISS.

    Returns
    -------
    (koi_info, results_df)
    koi_info : dict with disposition + parameters, or None if not found
    results_df : DataFrame of k most similar confirmed planets
    """
    _load()

    koi_info = lookup_koi(kepoi_name)

    if koi_info is None:
        raise ValueError(f"KOI '{kepoi_name}' not found in the NASA cumulative catalogue.")

    # Check we have the 6 features needed
    missing = [f for f in FEATURES if pd.isna(koi_info.get(f))]
    if missing:
        raise ValueError(
            f"KOI '{kepoi_name}' is missing parameters: {missing}. "
            f"Cannot compute similarity without them."
        )

    query = {f: float(koi_info[f]) for f in FEATURES}
    results = retrieve(query, k=k + 1)
    # Exclude itself if it's in the confirmed index
    results = results[results['kepoi_name'] != kepoi_name].head(k)
    return koi_info, results.reset_index(drop=True)
