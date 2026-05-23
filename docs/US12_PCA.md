# US-12 — Reducción de dimensionalidad (PCA)

## Subtareas

| # | Tarea | Implementación |
|---|--------|----------------|
| 1 | Integrar librería ML | **scikit-learn** (`pipeline/requirements.txt`) |
| 2 | Entrenar PCA con datos normalizados | `train_pca.py` + `pca_trainer.py` (entrada: `03_distances_minmax.csv` de US/11) |
| 3 | Varianza explicada | `pca_report.json` — ratios por componente y acumulada |
| 4 | Guardar modelo producción | `models/pca_model.joblib` (Python) + `models/pca_production.json` (iOS/Swift) |

## Flujo completo

```bash
cd pipeline
pip install -r requirements.txt

# US/11 + US/12 en un paso
python train_pca.py data/training_pose_recording.csv --run-pipeline-first

# Solo PCA si ya tienes Min-Max
python train_pca.py output/03_distances_minmax.csv
```

## Selección de componentes

Por defecto se conservan los componentes necesarios para alcanzar **≥ 95 %** de varianza explicada acumulada (mínimo 2).

Parámetros:

```bash
python train_pca.py output/03_distances_minmax.csv --variance-threshold 0.95 --min-components 2
```

## Salidas (`pipeline/models/`)

| Archivo | Uso |
|---------|-----|
| `pca_model.joblib` | Inferencia en Python (`apply_pca.py`) |
| `pca_production.json` | App iOS — `PosePCAReducer.swift` |
| `pca_report.json` | Documentación / entrega (varianza explicada) |
| `tensor_pca_reduced.npy` | Tensor (n_muestras, n_componentes) |

## Producción en iOS (Swift)

Copia `pca_production.json` al target TSL (bundle). Uso:

```swift
let pca = try PosePCAReducer()
let reduced = try pca.transform(features: distances11) // 11 distancias Min-Max
```

Fórmula: \( \mathbf{z} = (\mathbf{x} - \boldsymbol{\mu}) \mathbf{W}^T \) donde `components` en JSON son las filas de \( \mathbf{W} \).

## Inferencia Python

```bash
python apply_pca.py output/03_distances_minmax.csv
```

## Notas

- Entrena con **más grabaciones reales** (`pose_recording.csv` de la app) para mejorar la varianza explicada.
- Con pocas muestras, sklearn limita componentes a `min(n_samples, n_features)`.
