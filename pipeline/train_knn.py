#!/usr/bin/env python3
"""
US/13 — Entrenar y validar KNN (80/10/10).

Uso:
    python generate_labeled_dataset.py
    python train_knn.py
    python train_knn.py --use-pca
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from generate_labeled_dataset import main as generate_dataset_main
from knn_trainer import (
    load_labeled_dataset,
    print_classification_summary,
    save_knn_artifacts,
    train_knn_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="US/13: KNN entrenamiento y validación")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "data" / "labeled_postures.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "models",
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Usar features reducidas por PCA (US/12)",
    )
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Regenera labeled_postures.csv antes de entrenar",
    )
    parser.add_argument(
        "--k-candidates",
        type=int,
        nargs="*",
        default=[1, 3, 5, 7, 9, 11, 15],
    )
    args = parser.parse_args()

    if args.generate_dataset or not args.dataset.exists():
        print("Generando dataset etiquetado…")
        generate_dataset_main()

    if not args.dataset.exists():
        print(f"Error: no existe {args.dataset}", file=sys.stderr)
        return 1

    try:
        X, y, class_labels = load_labeled_dataset(args.dataset)
        model, report, splits = train_knn_pipeline(
            X,
            y,
            class_labels,
            use_pca=args.use_pca,
            models_dir=args.output_dir,
            k_candidates=args.k_candidates,
        )
        paths = save_knn_artifacts(
            model,
            report,
            splits.X_train,
            splits.y_train,
            args.output_dir,
        )

        minmax_src = Path(__file__).parent / "output" / "minmax_scaler.json"
        if minmax_src.exists():
            import shutil
            shutil.copy(minmax_src, args.output_dir / "minmax_scaler.json")
            print(f"  MinMax copiado a: {args.output_dir / 'minmax_scaler.json'}")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("KNN US/13 entrenado.")
    print(f"  Split: train={report.n_train} (80%) val={report.n_val} (10%) test={report.n_test} (10%)")
    print(f"  Mejor K (validación 10%): {report.best_k}")
    print(f"  Features: {report.n_features} (PCA={report.use_pca})")
    print(f"  Test Accuracy: {report.test_metrics.accuracy:.2%}")
    print(f"  Test F1-Score: {report.test_metrics.f1_score:.4f}")
    print("  Scores validación por K:")
    for k, score in sorted(report.validation_scores.items()):
        mark = " ← elegido" if k == report.best_k else ""
        print(f"    K={k}: F1={score:.4f}{mark}")

    print_classification_summary(model, splits.X_test, splits.y_test, class_labels, "Test (10%)")
    print(f"\n  Modelo joblib: {paths['joblib']}")
    print(f"  Modelo pkl (migración): {paths['pkl']}")
    print(f"  Producción iOS: {paths['production']}")
    print(f"  Reporte: {paths['report']}")
    print("  Salida discreta: 0=Seguro, 1=Inseguro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
