#!/usr/bin/env python3
"""
Generación de dataset simétrico por reflexión horizontal (data augmentation).

Toma dataset_completo.csv y produce dataset_simetrico_final.csv duplicando cada
muestra con espejo anatómico correcto (corrige sesgo direccional del entrenamiento).

Soporta tres formatos de entrada:
  - wide:   columnas left_shoulder_x, nose_x, … (Core ML / Vision export)
  - long:   columnas joint, x, y, confidence (grabaciones crudas)
  - distances: columnas dist_neck__left_shoulder, … (KNN / pipeline actual)

Uso:
    python generar_dataset_simetrico.py
    python generar_dataset_simetrico.py --input data/dataset_completo.csv --output data/dataset_simetrico_final.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Pares anatómicos izquierda ↔ derecha (snake_case y camelCase Vision)
# ---------------------------------------------------------------------------

LEFT_RIGHT_BODY_PARTS: tuple[str, ...] = (
    "eye",
    "ear",
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "knee",
    "ankle",
)

LEFT_RIGHT_JOINT_PAIRS_SNAKE: tuple[tuple[str, str], ...] = tuple(
    (f"left_{part}", f"right_{part}") for part in LEFT_RIGHT_BODY_PARTS
)

LEFT_RIGHT_JOINT_PAIRS_CAMEL: tuple[tuple[str, str], ...] = tuple(
    (f"left{part.capitalize()}", f"right{part.capitalize()}") for part in LEFT_RIGHT_BODY_PARTS
)

LEFT_TO_RIGHT_SNAKE: dict[str, str] = dict(LEFT_RIGHT_JOINT_PAIRS_SNAKE)
RIGHT_TO_LEFT_SNAKE: dict[str, str] = {r: l for l, r in LEFT_RIGHT_JOINT_PAIRS_SNAKE}

LEFT_TO_RIGHT_CAMEL: dict[str, str] = dict(LEFT_RIGHT_JOINT_PAIRS_CAMEL)
RIGHT_TO_LEFT_CAMEL: dict[str, str] = {r: l for l, r in LEFT_RIGHT_JOINT_PAIRS_CAMEL}


def mirror_joint_name(joint: str) -> str:
    """Intercambia left ↔ right en el nombre de articulación."""
    j = str(joint).strip()
    lower = j.lower()

    if lower in LEFT_TO_RIGHT_SNAKE:
        return LEFT_TO_RIGHT_SNAKE[lower]
    if lower in RIGHT_TO_LEFT_SNAKE:
        return RIGHT_TO_LEFT_SNAKE[lower]

    for left, right in LEFT_RIGHT_JOINT_PAIRS_CAMEL:
        if j.startswith(left):
            return right + j[len(left) :]
        if j.startswith(right):
            return left + j[len(right) :]

    return j


# ---------------------------------------------------------------------------
# Detección de formato
# ---------------------------------------------------------------------------


def detect_format(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"joint", "x", "y"}.issubset(cols):
        return "long"
    if any(re.match(r"^(left_|left[A-Z]).+_x$", c) for c in cols):
        return "wide"
    if any(str(c).startswith("dist_") for c in cols):
        return "distances"
    raise ValueError(
        "Formato no reconocido. Se espera:\n"
        "  - wide: left_shoulder_x, nose_x, …\n"
        "  - long: joint, x, y\n"
        "  - distances: dist_neck__left_shoulder, …"
    )


# ---------------------------------------------------------------------------
# Formato WIDE (requisitos estrictos del usuario)
# ---------------------------------------------------------------------------


def _wide_lr_partner_column(col: str) -> str | None:
    """
    Devuelve la columna pareja left↔right con el mismo sufijo (_x, _y, _confidence).
    Soporta left_shoulder_x y leftShoulder_x.
    """
    for left, right in LEFT_RIGHT_JOINT_PAIRS_SNAKE:
        for suffix in ("_x", "_y", "_confidence"):
            if col == f"{left}{suffix}":
                return f"{right}{suffix}"
            if col == f"{right}{suffix}":
                return f"{left}{suffix}"

    for left, right in LEFT_RIGHT_JOINT_PAIRS_CAMEL:
        for suffix in ("_x", "_y", "_confidence"):
            if col == f"{left}{suffix}":
                return f"{right}{suffix}"
            if col == f"{right}{suffix}":
                return f"{left}{suffix}"

    return None


def _swap_left_right_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Intercambia nombres de columnas left_* ↔ right_* (valores viajan con el nombre)."""
    rename: dict[str, str] = {}
    used: set[str] = set()

    for col in df.columns:
        if col in used:
            continue
        partner = _wide_lr_partner_column(col)
        if partner is None or partner not in df.columns:
            continue
        rename[col] = f"__SWAP__{partner}"
        rename[partner] = col
        used.add(col)
        used.add(partner)

    if not rename:
        return df.copy()

    out = df.rename(columns=rename)
    out.columns = [
        c[len("__SWAP__") :] if c.startswith("__SWAP__") else c for c in out.columns
    ]
    return out


def mirror_wide_pose(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Clonar el DataFrame.
    2. Invertir TODAS las columnas *_x: x' = 1.0 - x.
    3. Intercambiar nombres left_* ↔ right_* (cruce anatómico).
    Columnas *_y y *_confidence no se transforman numéricamente; solo participan en el swap.
    """
    clone = df.copy()

    x_cols = [c for c in clone.columns if str(c).endswith("_x")]
    for col in x_cols:
        clone[col] = 1.0 - pd.to_numeric(df[col], errors="coerce")

    clone = _swap_left_right_column_names(clone)
    return clone


# ---------------------------------------------------------------------------
# Formato LONG (joint, x, y, confidence)
# ---------------------------------------------------------------------------


def mirror_long_pose(df: pd.DataFrame) -> pd.DataFrame:
    """Espejo por muestra: x' desde lado opuesto; y' sin invertir."""
    group_cols = [c for c in ("timestamp", "frame_id", "track_id", "sample_id") if c in df.columns]
    meta_cols = [c for c in df.columns if c not in ("joint", "x", "y", "confidence")]

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
            cp = mirror_joint_name(joint).lower()
            if cp in joints and cp != joint.lower():
                src = joints[cp]
                new_x = 1.0 - float(src["x"])
                new_y = float(src["y"])
                conf_src = float(src["confidence"]) if "confidence" in src.index else None
            else:
                new_x = 1.0 - float(row["x"])
                new_y = float(row["y"])
                conf_src = float(row["confidence"]) if "confidence" in row.index else None

            out: dict = {c: row[c] for c in meta_cols}
            out["joint"] = joint
            out["x"] = new_x
            out["y"] = new_y
            if conf_src is not None:
                out["confidence"] = conf_src
            mirrored_rows.append(out)

    return pd.DataFrame(mirrored_rows, columns=df.columns)


# ---------------------------------------------------------------------------
# Formato DISTANCES (dataset_completo.csv actual)
# ---------------------------------------------------------------------------


def _parse_distance_column(col: str) -> tuple[str, str] | None:
    if not str(col).startswith("dist_"):
        return None
    body = str(col)[5:]
    if "__" not in body:
        return None
    a, b = body.split("__", 1)
    return a.strip().lower(), b.strip().lower()


def mirror_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cada dist_A__B del clon recibe el valor de dist_mirror(A)__mirror(B) del original.
    Las distancias son invariantes al espejo; solo se permutan por simetría L/R.
    """
    clone = df.copy()
    dist_cols = [c for c in df.columns if str(c).startswith("dist_")]

    if not dist_cols:
        raise ValueError("No hay columnas dist_* en el dataset.")

    for col in dist_cols:
        parsed = _parse_distance_column(col)
        if not parsed:
            clone[col] = df[col]
            continue
        a, b = parsed
        source_col = f"dist_{mirror_joint_name(a)}__{mirror_joint_name(b)}"
        if source_col in df.columns:
            clone[col] = df[source_col].values
        else:
            clone[col] = df[col].values

    return clone


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------


def mirror_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    fmt = detect_format(df)
    if fmt == "wide":
        return mirror_wide_pose(df)
    if fmt == "long":
        return mirror_long_pose(df)
    return mirror_distance_features(df)


def annotate_mirror_metadata(mirrored: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Marca filas espejadas sin alterar label."""
    out = mirrored.copy()
    if "source" in out.columns and len(out) == len(original):
        out["source"] = original["source"].astype(str) + "_espejo"
    if "sample_id" in out.columns and len(out) == len(original):
        out["sample_id"] = original["sample_id"].astype(str) + "_mirror"
    return out


def print_balance_report(
    original: pd.DataFrame,
    combined: pd.DataFrame,
    label_col: str | None,
) -> None:
    n_orig = len(original)
    n_total = len(combined)
    print("\n=== Verificación de balanceo ===")
    print(f"Filas originales:     {n_orig}")
    print(f"Filas espejadas:      {n_orig}")
    print(f"Filas totales:        {n_total}")
    print(f"Ratio total/original: {n_total / n_orig if n_orig else 0:.2f} (esperado: 2.00)")

    if n_total != 2 * n_orig:
        print("ADVERTENCIA: el total no es el doble del original.", file=sys.stderr)

    if label_col and label_col in combined.columns:
        print(f"\nDistribución de '{label_col}' (dataset combinado):")
        counts = combined[label_col].astype(str).value_counts().sort_index()
        for label, count in counts.items():
            pct = 100.0 * count / n_total if n_total else 0
            print(f"  {label}: {count} ({pct:.1f}%)")
        print(f"\nDistribución de '{label_col}' (solo original):")
        orig_counts = original[label_col].astype(str).value_counts().sort_index()
        for label, count in orig_counts.items():
            pct = 100.0 * count / n_orig if n_orig else 0
            print(f"  {label}: {count} ({pct:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera dataset simétrico por reflexión horizontal (augmentation L/R)."
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
        default=Path(__file__).parent / "data" / "dataset_simetrico_final.csv",
        help="CSV de salida concatenado",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: no se encontró {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input)
    if df.empty:
        print("ERROR: el dataset de entrada está vacío.", file=sys.stderr)
        return 1

    label_col = "label" if "label" in df.columns else None
    fmt = detect_format(df)

    print(f"Entrada:  {args.input.resolve()}")
    print(f"Formato:  {fmt}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Filas:    {len(df)}")

    mirrored = mirror_dataframe(df)
    mirrored = annotate_mirror_metadata(mirrored, df)

    combined = pd.concat([df, mirrored], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    print(f"\nSalida:   {args.output.resolve()}")
    print_balance_report(df, combined, label_col)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
