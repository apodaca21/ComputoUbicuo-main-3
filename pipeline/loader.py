"""Carga del archivo CSV con datos de pose grabados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import REQUIRED_CSV_COLUMNS


def load_pose_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Carga el CSV de grabación y valida el esquema.

    Formato esperado (una fila por articulación por frame/pista):
        timestamp, frame_id, track_id, joint, x, y, confidence

    Coordenadas x, y en espacio normalizado [0, 1] (Vision / captura iOS).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas faltantes en {path.name}: {missing}. "
            f"Se requieren: {list(REQUIRED_CSV_COLUMNS)}"
        )

    df = df.copy()
    df["joint"] = df["joint"].astype(str).str.strip().str.lower()
    df["frame_id"] = df["frame_id"].astype(int)
    df["track_id"] = df["track_id"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["confidence"] = df["confidence"].astype(float)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    invalid = df[df[["x", "y", "confidence"]].isna().any(axis=1)]
    if not invalid.empty:
        raise ValueError(
            f"Hay {len(invalid)} filas con x, y o confidence inválidos en {path.name}"
        )

    return df.sort_values(["frame_id", "track_id", "joint"]).reset_index(drop=True)
