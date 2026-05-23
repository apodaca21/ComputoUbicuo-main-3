"""
US/12 — Reducción de dimensionalidad (PCA) sobre datos normalizados (US/11).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import DISTANCE_FEATURE_NAMES


@dataclass
class PCAModelReport:
    n_samples: int
    n_features_in: int
    n_components: int
    explained_variance_ratio: list[float]
    cumulative_explained_variance: list[float]
    total_explained_variance: float
    variance_threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PCAModelArtifacts:
    sklearn_model: PCA
    report: PCAModelReport
    feature_names: tuple[str, ...]


def load_normalized_matrix(
    scaled_csv: Path,
    feature_cols: tuple[str, ...] = DISTANCE_FEATURE_NAMES,
) -> np.ndarray:
    """Carga la matriz (n_samples, n_features) desde distancias Min-Max."""
    df = pd.read_csv(scaled_csv)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en {scaled_csv}: {missing}")

    matrix = df[list(feature_cols)].to_numpy(dtype=np.float64)
    # Descarta filas con NaN (articulación faltante en ese frame).
    valid_mask = ~np.isnan(matrix).any(axis=1)
    matrix = matrix[valid_mask]
    if matrix.shape[0] < 2:
        raise ValueError(
            f"Se necesitan al menos 2 muestras válidas para PCA; hay {matrix.shape[0]}."
        )
    return matrix


def select_n_components(
    pca_full: PCA,
    variance_threshold: float = 0.95,
    min_components: int = 2,
) -> int:
    """Elige cuántos componentes conservar según varianza explicada acumulada."""
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n = max(min_components, min(n, pca_full.n_components_))
    if cumulative[n - 1] < variance_threshold and n < pca_full.n_components_:
        n = min(n + 1, pca_full.n_components_)
    return n


def train_pca(
    X: np.ndarray,
    variance_threshold: float = 0.95,
    min_components: int = 2,
) -> PCAModelArtifacts:
    """
    Entrena PCA con scikit-learn sobre datos ya normalizados (US/11).
    """
    n_samples, n_features = X.shape
    n_full = min(n_samples, n_features)
    pca_full = PCA(n_components=n_full, random_state=42)
    pca_full.fit(X)

    n_keep = select_n_components(
        pca_full,
        variance_threshold=variance_threshold,
        min_components=min(min_components, n_full),
    )
    n_keep = min(n_keep, n_full)

    pca = PCA(n_components=n_keep, random_state=42)
    pca.fit(X)

    cumulative = np.cumsum(pca.explained_variance_ratio_).tolist()
    report = PCAModelReport(
        n_samples=int(X.shape[0]),
        n_features_in=n_features,
        n_components=n_keep,
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        cumulative_explained_variance=cumulative,
        total_explained_variance=float(cumulative[-1]) if cumulative else 0.0,
        variance_threshold=variance_threshold,
    )

    return PCAModelArtifacts(
        sklearn_model=pca,
        report=report,
        feature_names=DISTANCE_FEATURE_NAMES,
    )


def transform_pca(model: PCA, X: np.ndarray) -> np.ndarray:
    return model.transform(X)


def save_pca_artifacts(
    artifacts: PCAModelArtifacts,
    output_dir: Path,
) -> dict[str, Path]:
    """Guarda modelo para Python (joblib) y producción (JSON para CoreML/Swift)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = output_dir / "pca_model.joblib"
    json_prod_path = output_dir / "pca_production.json"
    report_path = output_dir / "pca_report.json"
    reduced_npy_path = output_dir / "tensor_pca_reduced.npy"

    joblib.dump(artifacts.sklearn_model, joblib_path)

    pca = artifacts.sklearn_model
    production = {
        "version": 1,
        "feature_names": list(artifacts.feature_names),
        "n_features_in": int(pca.n_features_in_),
        "n_components": int(pca.n_components_),
        "mean": pca.mean_.tolist(),
        "components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_explained_variance": artifacts.report.total_explained_variance,
    }
    json_prod_path.write_text(json.dumps(production, indent=2), encoding="utf-8")
    report_path.write_text(
        json.dumps(artifacts.report.to_dict(), indent=2),
        encoding="utf-8",
    )

    return {
        "joblib": joblib_path,
        "production_json": json_prod_path,
        "report": report_path,
        "reduced_npy": reduced_npy_path,
    }


def export_reduced_tensor(
    model: PCA,
    X: np.ndarray,
    output_path: Path,
) -> np.ndarray:
    """Aplica PCA y guarda matriz reducida (n_samples, n_components)."""
    Z = transform_pca(model, X)
    np.save(output_path, Z)
    return Z
