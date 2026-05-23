#!/usr/bin/env python3
"""
Convierte grabaciones pose (formato long: joint,x,y) a CSV ancho para Create ML / Core ML.
Opcionalmente fusiona etiquetas desde dataset_completo.csv por sample_id.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Mismo orden y nombres que PostureCoreMLClassifier.swift / TSLPostureModel.mlmodel
JOINT_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("nose", "nose_x", "nose_y"),
    ("left_eye", "leftEye_x", "leftEye_y"),
    ("right_eye", "rightEye_x", "rightEye_y"),
    ("left_ear", "leftEar_x", "leftEar_y"),
    ("right_ear", "rightEar_x", "rightEar_y"),
    ("neck", "neck_x", "neck_y"),
    ("left_shoulder", "leftShoulder_x", "leftShoulder_y"),
    ("right_shoulder", "rightShoulder_x", "rightShoulder_y"),
    ("left_elbow", "leftElbow_x", "leftElbow_y"),
    ("right_elbow", "rightElbow_x", "rightElbow_y"),
    ("left_wrist", "leftWrist_x", "leftWrist_y"),
    ("right_wrist", "rightWrist_x", "rightWrist_y"),
    ("left_hip", "leftHip_x", "leftHip_y"),
    ("right_hip", "rightHip_x", "rightHip_y"),
    ("root", "root_x", "root_y"),
    ("left_knee", "leftKnee_x", "leftKnee_y"),
    ("right_knee", "rightKnee_x", "rightKnee_y"),
    ("left_ankle", "leftAnkle_x", "leftAnkle_y"),
    ("right_ankle", "rightAnkle_x", "rightAnkle_y"),
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(
    col for _, x_col, y_col in JOINT_COLUMNS for col in (x_col, y_col)
)


def snake_to_camel_joint(joint: str) -> str:
    j = str(joint).strip().lower()
    for snake, x_col, _ in JOINT_COLUMNS:
        if j == snake.replace("_", "") or j == snake:
            return snake
    parts = j.split("_")
    if len(parts) == 1:
        return parts[0]
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def normalize_label(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in ("safe", "derecho", "seguro", "good", "0"):
        return "derecho"
    if s in ("unsafe", "inseguro", "inclinado_jorobado", "bad", "1"):
        return "inclinado_jorobado"
    if "inclin" in s or "jorob" in s or "insegur" in s:
        return "inclinado_jorobado"
    return "derecho"


def long_pose_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    if not {"joint", "x", "y"}.issubset(df_long.columns):
        raise ValueError("Se requiere formato long con columnas joint, x, y")

    group_cols = [c for c in ("timestamp", "frame_id", "track_id") if c in df_long.columns]
    if not group_cols:
        raise ValueError("Faltan columnas de agrupación (frame_id, track_id)")

    rows: list[dict] = []
    for key, group in df_long.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        record: dict = dict(zip(group_cols, key))
        joints = {
            str(r["joint"]).strip().lower(): r for _, r in group.iterrows()
        }
        for snake, x_col, y_col in JOINT_COLUMNS:
            row = joints.get(snake)
            if row is None:
                alt = snake.replace("_", "")
                row = next(
                    (joints[k] for k in joints if k.replace("_", "") == alt),
                    None,
                )
            if row is not None:
                record[x_col] = float(row["x"])
                record[y_col] = float(row["y"])
            else:
                record[x_col] = 0.5
                record[y_col] = 0.5
        record["sample_id"] = f"{record.get('frame_id', 0)}_{record.get('track_id', 1)}"
        rows.append(record)

    return pd.DataFrame(rows)


def attach_labels(wide: pd.DataFrame, labels_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    if "sample_id" not in labels.columns or "label" not in labels.columns:
        raise ValueError(f"{labels_path} debe tener sample_id y label")
    lab = labels[["sample_id", "label"]].drop_duplicates(subset=["sample_id"])
    lab["label"] = lab["label"].map(normalize_label)
    out = wide.merge(lab, on="sample_id", how="left")
    out["label"] = out["label"].fillna("derecho")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="CSV ancho para entrenamiento Core ML")
    parser.add_argument(
        "--pose-csv",
        type=Path,
        nargs="+",
        default=[Path(__file__).parent / "data" / "training_pose_recording.csv"],
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path(__file__).parent / "data" / "dataset_completo.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "data" / "dataset_coreml_wide.csv",
    )
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for path in args.pose_csv:
        if not path.exists():
            print(f"Omitido (no existe): {path}")
            continue
        df = pd.read_csv(path)
        wide = long_pose_to_wide(df)
        wide["source"] = path.name
        frames.append(wide)

    if not frames:
        print("ERROR: ningún CSV de pose válido.", file=__import__("sys").stderr)
        return 1

    combined = pd.concat(frames, ignore_index=True)
    if args.labels_csv.exists():
        combined = attach_labels(combined, args.labels_csv)

    combined = combined[list(FEATURE_COLUMNS) + ["label", "sample_id", "source"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wide CSV: {args.output} ({len(combined)} filas)")
    print(combined["label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
