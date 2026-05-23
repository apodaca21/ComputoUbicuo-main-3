"""Constantes del pipeline US/11 — articulaciones Vision / MediaPipe compatibles."""

from __future__ import annotations

# Articulaciones del tren superior (se conservan tras filtrado espacial).
UPPER_BODY_JOINTS: tuple[str, ...] = (
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "root",
)

# Subset Selection: coordenadas irrelevantes (rostro, rodillas, tobillos).
EXCLUDED_JOINTS: frozenset[str] = frozenset(
    {
        # Rostro
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        # Piernas (fuera del tren superior)
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
)

# Aristas del tren superior para calcular distancias euclidianas (features).
UPPER_BODY_EDGES: tuple[tuple[str, str], ...] = (
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("neck", "root"),
)

# Columnas obligatorias del CSV grabado.
REQUIRED_CSV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "frame_id",
    "track_id",
    "joint",
    "x",
    "y",
    "confidence",
)

DISTANCE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"dist_{a}__{b}" for a, b in UPPER_BODY_EDGES
)
