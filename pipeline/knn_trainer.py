"""
US/13 — Entrenamiento y validación del clasificador KNN.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from config import DISTANCE_FEATURE_NAMES

# US-14 — salida discreta binaria
LABEL_SEGURO = "derecho"
LABEL_INSEGURO = "inclinado_jorobado"
DISCRETE_SEGURO = 0
DISCRETE_INSEGURO = 1
LABEL_TO_DISCRETE: dict[str, int] = {
    LABEL_SEGURO: DISCRETE_SEGURO,
    LABEL_INSEGURO: DISCRETE_INSEGURO,
}


@dataclass
class KNNMetrics:
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    support: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class KNNTrainingReport:
    best_k: int
    k_candidates: list[int]
    validation_scores: dict[int, float]
    train_metrics: KNNMetrics
    validation_metrics: KNNMetrics
    test_metrics: KNNMetrics
    class_labels: list[str]
    n_train: int
    n_val: int
    n_test: int
    use_pca: bool
    n_features: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["train_metrics"] = self.train_metrics.to_dict()
        d["validation_metrics"] = self.validation_metrics.to_dict()
        d["test_metrics"] = self.test_metrics.to_dict()
        return d


def load_labeled_dataset(
    csv_path: Path,
    feature_cols: tuple[str, ...] = DISTANCE_FEATURE_NAMES,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"El dataset debe incluir columna 'label': {csv_path}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Features faltantes: {missing}")

    df = df.dropna(subset=list(feature_cols))
    X = df[list(feature_cols)].to_numpy(dtype=np.float64)
    y_raw = df["label"].astype(str).to_numpy()
    classes = [LABEL_SEGURO, LABEL_INSEGURO]
    y = np.array(
        [LABEL_TO_DISCRETE.get(v, DISCRETE_SEGURO) for v in y_raw],
        dtype=np.int64,
    )
    return X, y, classes


def apply_pca_if_available(X: np.ndarray, models_dir: Path) -> tuple[np.ndarray, bool]:
    pca_path = models_dir / "pca_model.joblib"
    if not pca_path.exists():
        return X, False
    pca = joblib.load(pca_path)
    return pca.transform(X), True


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, ...]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Las proporciones deben sumar 1.0")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=y
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(
    model: KNeighborsClassifier,
    X: np.ndarray,
    y: np.ndarray,
    average: str = "binary",
) -> KNNMetrics:
    pred = model.predict(X)
    # F1 binario si hay 2 clases; si no, weighted
    avg = average if len(set(y)) <= 2 else "weighted"
    return KNNMetrics(
        accuracy=float(accuracy_score(y, pred)),
        f1_score=float(f1_score(y, pred, average=avg, zero_division=0)),
        precision=float(precision_score(y, pred, average=avg, zero_division=0)),
        recall=float(recall_score(y, pred, average=avg, zero_division=0)),
        support=int(len(y)),
    )


def tune_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k_candidates: list[int] | None = None,
) -> tuple[int, dict[int, float]]:
    if k_candidates is None:
        k_candidates = [1, 3, 5, 7, 9, 11, 15]

    max_k = len(X_train)
    k_candidates = [k for k in k_candidates if k <= max_k]
    if not k_candidates:
        k_candidates = [1]

    scores: dict[int, float] = {}
    best_k = k_candidates[0]
    best_f1 = -1.0

    for k in k_candidates:
        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        avg = "binary" if len(set(y_val)) <= 2 else "weighted"
        f1 = float(f1_score(y_val, pred, average=avg, zero_division=0))
        scores[k] = f1
        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    return best_k, scores


@dataclass
class KNNSplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def train_knn_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    class_labels: list[str],
    use_pca: bool = False,
    models_dir: Path | None = None,
    k_candidates: list[int] | None = None,
) -> tuple[KNeighborsClassifier, KNNTrainingReport, KNNSplitData]:
    if use_pca and models_dir:
        X, applied = apply_pca_if_available(X, models_dir)
        use_pca = applied

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    best_k, val_scores = tune_k(X_train, y_train, X_val, y_val, k_candidates)

    model = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    model.fit(X_train, y_train)

    train_m = evaluate_model(model, X_train, y_train)
    val_m = evaluate_model(model, X_val, y_val)
    test_m = evaluate_model(model, X_test, y_test)

    report = KNNTrainingReport(
        best_k=best_k,
        k_candidates=list(val_scores.keys()),
        validation_scores={int(k): float(v) for k, v in val_scores.items()},
        train_metrics=train_m,
        validation_metrics=val_m,
        test_metrics=test_m,
        class_labels=class_labels,
        n_train=len(y_train),
        n_val=len(y_val),
        n_test=len(y_test),
        use_pca=use_pca,
        n_features=int(X.shape[1]),
    )

    splits = KNNSplitData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    return model, report, splits


def save_knn_artifacts(
    model: KNeighborsClassifier,
    report: KNNTrainingReport,
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = output_dir / "knn_model.joblib"
    pkl_path = output_dir / "knn_model.pkl"
    report_path = output_dir / "knn_report.json"
    prod_path = output_dir / "knn_production.json"

    joblib.dump(model, joblib_path)
    joblib.dump(model, pkl_path)

    production = {
        "version": 1,
        "k": report.best_k,
        "metric": "euclidean",
        "class_labels": report.class_labels,
        "discrete_output": {
            "0": "seguro",
            "1": "inseguro",
        },
        "discrete_mapping": {
            LABEL_SEGURO: DISCRETE_SEGURO,
            LABEL_INSEGURO: DISCRETE_INSEGURO,
        },
        "use_pca": report.use_pca,
        "n_features": report.n_features,
        "training_vectors": X_train.tolist(),
        "training_labels": [int(i) for i in y_train.tolist()],
        "test_accuracy": report.test_metrics.accuracy,
        "test_f1_score": report.test_metrics.f1_score,
    }
    prod_path.write_text(json.dumps(production, indent=2), encoding="utf-8")

    full_report = report.to_dict()
    y_test_pred_note = "Ver knn_report.json para métricas train/val/test"
    full_report["note"] = y_test_pred_note
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")

    # Informe detallado sklearn en test (re-entrenar split para predicciones)
    return {
        "joblib": joblib_path,
        "pkl": pkl_path,
        "report": report_path,
        "production": prod_path,
    }


def print_classification_summary(
    model: KNeighborsClassifier,
    X: np.ndarray,
    y: np.ndarray,
    class_labels: list[str],
    split_name: str,
) -> None:
    pred = model.predict(X)
    print(f"\n--- {split_name} ---")
    print(
        classification_report(
            y,
            pred,
            target_names=class_labels,
            zero_division=0,
        )
    )
