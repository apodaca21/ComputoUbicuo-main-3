#!/usr/bin/env python3
"""
US/12 — Entrenar PCA sobre datos normalizados (US/11).

Uso:
    python train_pca.py output/03_distances_minmax.csv
    python train_pca.py data/sample_pose_recording.csv --run-pipeline-first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pca_trainer import (
    export_reduced_tensor,
    load_normalized_matrix,
    save_pca_artifacts,
    train_pca,
)
from process_pose_csv import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="US/12: entrenar PCA y guardar modelo")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path(__file__).parent / "output" / "03_distances_minmax.csv",
        help="CSV Min-Max (US/11) o CSV crudo si --run-pipeline-first",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "models",
        help="Directorio del modelo PCA (default: pipeline/models)",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="Varianza acumulada mínima a conservar (default: 0.95)",
    )
    parser.add_argument(
        "--min-components",
        type=int,
        default=2,
        help="Mínimo de componentes principales",
    )
    parser.add_argument(
        "--run-pipeline-first",
        action="store_true",
        help="Ejecuta US/11 sobre el CSV antes de entrenar PCA",
    )
    args = parser.parse_args()

    pipeline_dir = Path(__file__).parent / "output"
    scaled_csv = args.input

    if args.run_pipeline_first:
        if not args.input.exists():
            print(f"Error: no existe {args.input}", file=sys.stderr)
            return 1
        run_pipeline(args.input, pipeline_dir)
        scaled_csv = pipeline_dir / "03_distances_minmax.csv"

    if not scaled_csv.exists():
        print(f"Error: no existe {scaled_csv}. Ejecuta US/11 primero.", file=sys.stderr)
        return 1

    try:
        X = load_normalized_matrix(scaled_csv)
        artifacts = train_pca(
            X,
            variance_threshold=args.variance_threshold,
            min_components=args.min_components,
        )
        paths = save_pca_artifacts(artifacts, args.output_dir)
        export_reduced_tensor(
            artifacts.sklearn_model,
            X,
            paths["reduced_npy"],
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    r = artifacts.report
    print("PCA US/12 entrenado.")
    print(f"  Muestras: {r.n_samples}")
    print(f"  Features entrada: {r.n_features_in} → componentes: {r.n_components}")
    print(f"  Varianza explicada total: {r.total_explained_variance:.2%}")
    print("  Por componente:")
    for i, (ratio, cum) in enumerate(
        zip(r.explained_variance_ratio, r.cumulative_explained_variance), start=1
    ):
        print(f"    PC{i}: {ratio:.2%} (acumulada {cum:.2%})")
    print(f"  Modelo joblib: {paths['joblib']}")
    print(f"  Producción JSON: {paths['production_json']}")
    print(f"  Tensor reducido: {paths['reduced_npy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
