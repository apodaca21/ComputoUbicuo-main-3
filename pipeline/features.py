"""Extracción de distancias entre articulaciones del tren superior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DISTANCE_FEATURE_NAMES, UPPER_BODY_EDGES


def _euclidean(ax: float, ay: float, bx: float, by: float) -> float:
    return float(np.hypot(ax - bx, ay - by))


def compute_upper_body_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa por (frame_id, track_id) y calcula distancias euclidianas
    entre pares de articulaciones del tren superior.

    Si falta alguna articulación de un par, la distancia queda como NaN.
    """
    group_cols = ["timestamp", "frame_id", "track_id"]
    rows: list[dict] = []

    for keys, group in df.groupby(group_cols, sort=True):
        ts, frame_id, track_id = keys
        joint_map = {
            row.joint: (row.x, row.y)
            for row in group.itertuples(index=False)
        }

        record: dict = {
            "timestamp": ts,
            "frame_id": frame_id,
            "track_id": track_id,
        }
        for (ja, jb), name in zip(UPPER_BODY_EDGES, DISTANCE_FEATURE_NAMES):
            if ja in joint_map and jb in joint_map:
                ax, ay = joint_map[ja]
                bx, by = joint_map[jb]
                record[name] = _euclidean(ax, ay, bx, by)
            else:
                record[name] = np.nan
        rows.append(record)

    return pd.DataFrame(rows)


def distances_to_tensor(distances_df: pd.DataFrame) -> np.ndarray:
    """
    Convierte el DataFrame de distancias a tensor NumPy.

    Forma: (n_sequences, n_frames, n_features)
    Una secuencia = una pista (track_id) ordenada por frame_id.
    """
    feature_cols = list(DISTANCE_FEATURE_NAMES)
    sequences: list[np.ndarray] = []

    for track_id, track_df in distances_df.groupby("track_id", sort=True):
        ordered = track_df.sort_values("frame_id")
        matrix = ordered[feature_cols].to_numpy(dtype=np.float64)
        sequences.append(matrix)

    if not sequences:
        return np.empty((0, 0, len(feature_cols)), dtype=np.float64)

    max_len = max(s.shape[0] for s in sequences)
    n_feat = sequences[0].shape[1]
    n_seq = len(sequences)
    tensor = np.full((n_seq, max_len, n_feat), np.nan, dtype=np.float64)

    for i, seq in enumerate(sequences):
        tensor[i, : seq.shape[0], :] = seq

    return tensor
