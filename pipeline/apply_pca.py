#!/usr/bin/env python3
"""Aplica el modelo PCA guardado a nuevos datos normalizados."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from pca_trainer import load_normalized_matrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Aplicar PCA en producción (Python)")
    parser.add_argument("scaled_csv", type=Path, help="03_distances_minmax.csv")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "models" / "pca_model.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "output" / "pca_transformed.csv",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Error: entrena primero con train_pca.py ({args.model})", file=sys.stderr)
        return 1

    model = joblib.load(args.model)
    X = load_normalized_matrix(args.scaled_csv)
    Z = model.transform(X)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, Z, delimiter=",", header=",".join(
        [f"PC{i+1}" for i in range(Z.shape[1])]
    ), comments="")

    meta = {
        "input_shape": list(X.shape),
        "output_shape": list(Z.shape),
        "n_components": int(model.n_components_),
    }
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"PCA aplicado: {X.shape} → {Z.shape}")
    print(f"  Guardado: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
