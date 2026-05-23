#!/usr/bin/env python3
"""
Entrena TSLPostureModel.mlmodel (clasificador tabular 38 features x,y)
compatible con PostureCoreMLClassifier.swift.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from preparar_coreml_wide import FEATURE_COLUMNS, normalize_label

LABEL_DERECHO = "derecho"
LABEL_INSEGURO = "inclinado_jorobado"


def load_xy_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en {path}: {missing[:5]}…")
    if "label" not in df.columns:
        raise ValueError("Falta columna 'label'")

    df = df.dropna(subset=list(FEATURE_COLUMNS))
    df["label"] = df["label"].map(normalize_label)
    X = df[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()
    return df, X, y


def export_coreml(
    model: RandomForestClassifier,
    output_path: Path,
    class_labels: list[str],
) -> None:
    import coremltools as ct

    feature_cols = list(FEATURE_COLUMNS)
    input_features = [(name, ct.models.datatypes.Double()) for name in feature_cols]

    mlmodel = ct.converters.sklearn.convert(
        model,
        input_features=input_features,
        output_feature_names="label",
    )
    mlmodel.author = "The Silent Coach — pipeline"
    mlmodel.short_description = "Postura: derecho vs inclinado_jorobado (dataset simétrico)"
    mlmodel.save(str(output_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrenar TSLPostureModel.mlmodel")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "data" / "dataset_coreml_wide_simetrico.csv",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path(__file__).parent / "models" / "TSLPostureModel.mlmodel",
    )
    parser.add_argument(
        "--ios-dir",
        type=Path,
        default=Path(__file__).parent.parent / "TSL" / "TSL",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: no existe {args.dataset}", file=sys.stderr)
        print("Ejecuta antes: preparar_coreml_wide.py && generar_dataset_simetrico.py", file=sys.stderr)
        return 1

    df, X, y = load_xy_dataset(args.dataset)
    if len(df) < 20:
        print(f"ERROR: pocas muestras ({len(df)}). Necesitas más datos etiquetados.", file=sys.stderr)
        return 1

    classes = sorted(set(y))
    print(f"Muestras: {len(df)} | clases: {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if len(set(y)) > 1 else None,
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Informe test ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    export_coreml(clf, args.output_model, class_labels=classes)
    print(f"\nModelo guardado: {args.output_model.resolve()}")

    report = {
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": classes,
        "feature_count": len(FEATURE_COLUMNS),
    }
    report_path = args.output_model.parent / "coreml_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ios_model = args.ios_dir / "TSLPostureModel.mlmodel"
    shutil.copy2(args.output_model, ios_model)
    print(f"Copiado a iOS: {ios_model.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
