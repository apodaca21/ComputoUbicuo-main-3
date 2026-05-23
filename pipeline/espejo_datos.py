#!/usr/bin/env python3
"""
Data augmentation por reflexión horizontal (espejo) para corregir sesgo direccional.

Lee dataset_completo.csv, duplica cada muestra reflejando X (1 - x), intercambia
articulaciones izquierda/derecha y concatena con el original en dataset_simetrico.csv.

Uso:
    python espejo_datos.py
    python espejo_datos.py --input data/mi_dataset.csv --output data/salida.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Pares anatómicos izquierda ↔ derecha (Vision / MediaPipe).
LEFT_RIGHT_JOINT_PAIRS: tuple[tuple[str, str], ...] = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

LEFT_TO_RIGHT: dict[str, str] = {left: right for left, right in LEFT_RIGHT_JOINT_PAIRS}
RIGHT_TO_LEFT: dict[str, str] = {right: left for left, right in LEFT_RIGHT_JOINT_PAIRS}


def mirror_joint_name(joint: str) -> str:
    """Nombre de articulación tras espejo horizontal (left ↔ right)."""
    j = str(joint).strip().lower()
    if j in LEFT_TO_RIGHT:
        return LEFT_TO_RIGHT[j]
    if j in RIGHT_TO_LEFT:
        return RIGHT_TO_LEFT[j]
    return j


def counterpart_joint(joint: str) -> str:
    """Articulación del lado opuesto; centrales (nose, neck, root) devuelven el mismo nombre."""
    return mirror_joint_name(joint)


def detect_format(df: pd.DataFrame) -> str:
    """Detecta formato: long (joint,x,y), wide (left_*_x) o distances (dist_*)."""
    cols = set(df.columns)
    if {"joint", "x", "y"}.issubset(cols):
        return "long"
    if any(re.match(r"^left_.+_x$", c) for c in cols):
        return "wide"
    if any(c.startswith("dist_") for c in cols):
        return "distances"
    raise ValueError(
        "Formato no reconocido. Se espera:\n"
        "  - long: columnas joint, x, y\n"
        "  - wide: columnas left_shoulder_x, right_shoulder_x, ...\n"
        "  - distances: columnas dist_neck__left_shoulder, ..."
    )


def mirror_long_pose(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formato largo: una fila por articulación.
    x' = 1 - x del lado opuesto; y' = y del lado opuesto; centrales: x' = 1 - x.
    """
    group_cols = [c for c in ("timestamp", "frame_id", "track_id", "sample_id") if c in df.columns]
    if not group_cols:
        group_cols = ["frame_id"] if "frame_id" in df.columns else []

    meta_cols = [c for c in df.columns if c not in ("joint", "x", "y", "confidence")]
    value_cols = [c for c in ("x", "y", "confidence") if c in df.columns]

    mirrored_rows: list[dict] = []

    if group_cols:
        groups = df.groupby(group_cols, sort=False)
    else:
        groups = [(None, df)]

    for _, group in groups:
        joints: dict[str, pd.Series] = {}
        for _, row in group.iterrows():
            joints[str(row["joint"]).strip().lower()] = row

        for joint, row in joints.items():
            cp = counterpart_joint(joint)
            if cp in joints and cp != joint:
                src = joints[cp]
                new_x = 1.0 - float(src["x"])
                new_y = float(src["y"])
            else:
                new_x = 1.0 - float(row["x"])
                new_y = float(row["y"])

            out: dict = {}
            for c in meta_cols:
                out[c] = row[c]
            out["joint"] = joint
            out["x"] = new_x
            out["y"] = new_y
            if "confidence" in value_cols:
                out["confidence"] = (
                    float(joints[cp]["confidence"])
                    if cp in joints and cp != joint
                    else float(row["confidence"])
                )
            mirrored_rows.append(out)

    return pd.DataFrame(mirrored_rows, columns=df.columns)


def _wide_joint_pairs(columns: list[str]) -> list[tuple[str, str, str]]:
    """
    Detecta pares (left_prefix, right_prefix, suffix) p.ej.
    ('left_shoulder', 'right_shoulder', '_x').
    """
    pattern = re.compile(r"^(left|right)_(.+)_([xy])$")
    buckets: dict[tuple[str, str], dict[str, str]] = {}

    for col in columns:
        m = pattern.match(col)
        if not m:
            continue
        side, body, axis = m.group(1), m.group(2), m.group(3)
        key = (body, axis)
        buckets.setdefault(key, {})[side] = col

    pairs: list[tuple[str, str, str]] = []
    for (body, axis), sides in buckets.items():
        if "left" in sides and "right" in sides:
            pairs.append((sides["left"], sides["right"], axis))

    return pairs


def mirror_wide_pose(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formato ancho: left_shoulder_x, left_shoulder_y, ...
    left_*_x = 1 - right_*_x original; left_*_y = right_*_y original (y sin invertir).
    """
    clone = df.copy()
    pairs = _wide_joint_pairs(list(df.columns))

    if not pairs:
        raise ValueError(
            "No se encontraron columnas left_*_x / right_*_x en el dataset ancho."
        )

    for left_col, right_col, axis in pairs:
        if axis == "x":
            clone[left_col] = 1.0 - df[right_col].astype(float)
            clone[right_col] = 1.0 - df[left_col].astype(float)
        else:
            clone[left_col] = df[right_col]
            clone[right_col] = df[left_col]

    # Columnas centrales con solo *_x / *_y (nose_x, neck_x, root_x, ...)
    center_pattern = re.compile(r"^([a-z_]+)_([xy])$")
    for col in df.columns:
        m = center_pattern.match(col)
        if not m:
            continue
        name, axis = m.group(1), m.group(2)
        if name.startswith("left_") or name.startswith("right_"):
            continue
        if axis == "x":
            clone[col] = 1.0 - df[col].astype(float)

    return clone


def _parse_distance_column(col: str) -> tuple[str, str] | None:
    if not col.startswith("dist_"):
        return None
    body = col[5:]
    if "__" not in body:
        return None
    a, b = body.split("__", 1)
    return a, b


def mirror_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de distancias: cada columna recibe el valor de su homóloga espejada.
    p.ej. dist_neck__left_shoulder ← dist_neck__right_shoulder (filas originales).
    """
    dist_cols = [c for c in df.columns if c.startswith("dist_")]
    if not dist_cols:
        raise ValueError("No hay columnas dist_* en el dataset.")

    clone = df.copy()
    for col in dist_cols:
        parsed = _parse_distance_column(col)
        if not parsed:
            continue
        a, b = parsed
        source_col = f"dist_{mirror_joint_name(a)}__{mirror_joint_name(b)}"
        if source_col in df.columns:
            clone[col] = df[source_col]
        else:
            clone[col] = df[col]

    return clone


def mirror_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica espejo según el formato detectado."""
    fmt = detect_format(df)
    if fmt == "long":
        return mirror_long_pose(df)
    if fmt == "wide":
        return mirror_wide_pose(df)
    return mirror_distance_features(df)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Coloca dataset_completo.csv o usa --input."
        )
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Augmentation por espejo horizontal (simetría izquierda/derecha)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "data" / "dataset_completo.csv",
        help="CSV de entrada (default: pipeline/data/dataset_completo.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "dataset_simetrico.csv",
        help="CSV de salida concatenado",
    )
    args = parser.parse_args()

    df = load_dataset(args.input)
    if df.empty:
        print("El dataset de entrada está vacío.", file=sys.stderr)
        return 1

    label_col = "label" if "label" in df.columns else None
    if label_col:
        df[label_col] = df[label_col].astype(str)

    fmt = detect_format(df)
    print(f"Formato detectado: {fmt}")
    print(f"Filas originales: {len(df)}")

    mirrored = mirror_dataframe(df)

    # label intacto; marcar origen del clon si hay columnas auxiliares
    if label_col and label_col in mirrored.columns:
        mirrored[label_col] = df[label_col].values if len(mirrored) == len(df) else mirrored[label_col]

    if "source" in mirrored.columns:
        mirrored["source"] = mirrored["source"].astype(str) + "_espejo"
    if "sample_id" in mirrored.columns and len(mirrored) == len(df):
        mirrored["sample_id"] = df["sample_id"].astype(str) + "_mirror"

    combined = pd.concat([df, mirrored], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    print(f"Filas espejadas: {len(mirrored)}")
    print(f"Total dataset_simetrico: {len(combined)}")
    print(f"Guardado en: {args.output.resolve()}")

    if label_col:
        print("Distribución de labels (combinado):")
        print(combined[label_col].value_counts().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
