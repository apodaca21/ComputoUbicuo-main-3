#!/usr/bin/env python3
"""
Genera dataset etiquetado para KNN (US/13).

Fuentes:
  1. CSV con columna `label` (derecho | inclinado_jorobado)
  2. Heurística de postura sobre distancias (como PostureAnalyzer en iOS)
  3. Muestras sintéticas si hay pocas filas reales
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import DISTANCE_FEATURE_NAMES
from features import compute_upper_body_distances
from loader import load_pose_csv
from minmax_scaling import fit_minmax, transform_minmax
from spatial_filter import apply_spatial_subset

LABEL_GOOD = "derecho"
LABEL_BAD = "inclinado_jorobado"


def heuristic_label(row: pd.Series) -> str:
    """Reglas simples sobre distancias normalizadas (aprox. PostureAnalyzer)."""
    neck_l = row.get("dist_neck__left_shoulder", np.nan)
    neck_r = row.get("dist_neck__right_shoulder", np.nan)
    shoulders = row.get("dist_left_shoulder__right_shoulder", np.nan)
    neck_root = row.get("dist_neck__root", np.nan)

    if any(np.isnan(v) for v in [neck_l, neck_r, shoulders]):
        return LABEL_GOOD

    asym = abs(neck_l - neck_r) / max(shoulders, 1e-6)
    if asym > 0.35 or neck_root > 0.42:
        return LABEL_BAD
    return LABEL_GOOD


def synthesize_samples(n_per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    """Muestras sintéticas Min-Max para entrenar KNN con volumen suficiente."""
    rows = []
    for label, base in [
        (LABEL_GOOD, {"spread": 0.08, "torso": 0.28}),
        (LABEL_BAD, {"spread": 0.22, "torso": 0.48}),
    ]:
        for i in range(n_per_class):
            noise = rng.normal(0, 0.04, size=len(DISTANCE_FEATURE_NAMES))
            record = {name: np.nan for name in DISTANCE_FEATURE_NAMES}
            record["dist_neck__left_shoulder"] = np.clip(
                base["torso"] + noise[0], 0, 1
            )
            record["dist_neck__right_shoulder"] = np.clip(
                base["torso"] + base["spread"] + noise[1], 0, 1
            )
            record["dist_left_shoulder__right_shoulder"] = np.clip(
                0.32 + base["spread"] + noise[2], 0, 1
            )
            record["dist_left_shoulder__left_elbow"] = np.clip(0.22 + noise[3], 0, 1)
            record["dist_left_elbow__left_wrist"] = np.clip(0.18 + noise[4], 0, 1)
            record["dist_right_shoulder__right_elbow"] = np.clip(0.22 + noise[5], 0, 1)
            record["dist_right_elbow__right_wrist"] = np.clip(0.18 + noise[6], 0, 1)
            record["dist_left_shoulder__left_hip"] = np.clip(0.35 + noise[7], 0, 1)
            record["dist_right_shoulder__right_hip"] = np.clip(0.35 + noise[8], 0, 1)
            record["dist_left_hip__right_hip"] = np.clip(0.20 + noise[9], 0, 1)
            record["dist_neck__root"] = np.clip(base["torso"] + noise[10], 0, 1)
            record["label"] = label
            record["source"] = "synthetic"
            record["sample_id"] = f"{label}_{i}"
            rows.append(record)
    return pd.DataFrame(rows)


def build_from_pose_csv(csv_path: Path, min_confidence: float = 0.2) -> pd.DataFrame:
    raw = load_pose_csv(csv_path)
    filtered = apply_spatial_subset(raw, min_confidence=min_confidence)
    distances = compute_upper_body_distances(filtered)
    scaler = fit_minmax(distances)
    scaled = transform_minmax(distances, scaler)

    if "label" in raw.columns:
        labels = (
            raw.groupby(["frame_id", "track_id"])["label"]
            .first()
            .reset_index()
        )
        scaled = scaled.merge(labels, on=["frame_id", "track_id"], how="left")
        scaled["label"] = scaled["label"].fillna("").apply(
            lambda x: LABEL_BAD if "inclin" in str(x).lower() or "bad" in str(x).lower() else LABEL_GOOD
        )
    else:
        scaled["label"] = scaled.apply(heuristic_label, axis=1)

    scaled["source"] = csv_path.name
    scaled["sample_id"] = scaled.apply(
        lambda r: f"{r['frame_id']}_{r['track_id']}", axis=1
    )
    return scaled


def main() -> int:
    parser = argparse.ArgumentParser(description="Generar dataset etiquetado US/13")
    parser.add_argument(
        "--pose-csv",
        type=Path,
        nargs="*",
        default=[Path(__file__).parent / "data" / "training_pose_recording.csv"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "labeled_postures.csv",
    )
    parser.add_argument("--synthetic-per-class", type=int, default=80)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for p in args.pose_csv:
        if p.exists():
            frames.append(build_from_pose_csv(p))

    if frames:
        real = pd.concat(frames, ignore_index=True)
    else:
        real = pd.DataFrame()

    rng = np.random.default_rng(42)
    synthetic = synthesize_samples(args.synthetic_per_class, rng)

    feature_cols = list(DISTANCE_FEATURE_NAMES)
    if not real.empty:
        real = real[feature_cols + ["label", "source", "sample_id"]]
    combined = pd.concat([real, synthetic], ignore_index=True)
    combined = combined.dropna(subset=feature_cols)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    counts = combined["label"].value_counts().to_dict()
    print(f"Dataset etiquetado: {args.output}")
    print(f"  Total muestras: {len(combined)}")
    print(f"  Distribución: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
