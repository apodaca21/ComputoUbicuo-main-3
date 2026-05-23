# US-13 — Entrenamiento y validación del modelo KNN

## Checklist

| # | Tarea | Implementación |
|---|--------|----------------|
| 1 | Split 80/10/10 | `knn_trainer.split_dataset()` estratificado |
| 2 | Entrenar KNN (80 %) | `KNeighborsClassifier` en subconjunto train |
| 3 | Ajustar K (10 % val) | Grid sobre `k_candidates`; mejor F1 en validación |
| 4 | Evaluar test (10 %) | Accuracy y F1-Score en conjunto aislado |

## Flujo

```bash
cd pipeline
python3 -m pip install -r requirements.txt

# Dataset etiquetado (real + sintético)
python3 generate_labeled_dataset.py

# Opcional: PCA antes de KNN
python3 train_pca.py data/training_pose_recording.csv --run-pipeline-first

# Entrenar KNN
python3 train_knn.py --generate-dataset
python3 train_knn.py --use-pca   # features PCA (recomendado)
```

## Etiquetas

| Label | Significado |
|-------|-------------|
| `derecho` | Postura correcta |
| `inclinado_jorobado` | Postura incorrecta |

## Salidas (`pipeline/models/`)

| Archivo | Contenido |
|---------|-----------|
| `knn_model.joblib` | Modelo sklearn |
| `knn_report.json` | Métricas train/val/test, K elegido |
| `knn_production.json` | Inferencia iOS (`PoseKNNClassifier`) |

## Métricas (ejemplo con dataset por defecto)

Tras entrenar verás en consola:

- **Accuracy** y **F1-Score** en test (10 %)
- Tabla `classification_report` del split test
- F1 por cada valor de K en validación

## Split

```
Total ──► 80% Train ──► entrena KNN
       ├─► 10% Validation ──► elige K
       └─► 10% Test ──► métricas finales (no usado en tuning)
```

## iOS

Copia `knn_production.json` al bundle TSL:

```swift
let knn = try PoseKNNClassifier()
let pred = try knn.predict(features: vectorReducido)
// pred.label: "derecho" | "inclinado_jorobado"
```

Si entrenaste con `--use-pca`, el vector de entrada debe ser la salida de `PosePCAReducer` (3 componentes). Si no, las 11 distancias Min-Max.

## Integración con el proyecto

```
CSV pose → US/11 Min-Max → US/12 PCA (opcional) → US/13 KNN → clase de postura
```

El MOT con Kalman (README ComputoUbicuo) estabiliza las articulaciones antes de calcular distancias para el KNN.
