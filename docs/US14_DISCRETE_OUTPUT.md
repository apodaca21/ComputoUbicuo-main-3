# US-14 — Salida discreta (clasificación binaria)

## Subtareas

| # | Tarea | Implementación |
|---|--------|----------------|
| 1 | Salida entera 0/1 | `DiscretePostureClass`: **0 = Seguro**, **1 = Inseguro** |
| 2 | Integrar con lógica de negocio | `PoseMLPostureEngine` sustituye distancias/heurística estática por KNN+PCA |
| 3 | Guardar KNN (.pkl) | `pipeline/models/knn_model.pkl` (+ `knn_model.joblib`) |

## Flujo en la app (iOS)

```
Kalman (articulaciones)
    → 11 distancias tren superior
    → PCA (3 componentes)
    → KNN
    → entero 0 | 1
    → semáforo verde | rojo
```

Si falla la carga del modelo, **fallback** a reglas geométricas (`PostureAnalyzerLegacy`).

## Mapeo discreto

| Valor | Significado | UI |
|-------|-------------|-----|
| `0` | Seguro | Verde — postura correcta |
| `1` | Inseguro | Rojo — postura incorrecta |
| `-1` | Sin persona | Gris |

## Archivos clave

| Archivo | Rol |
|---------|-----|
| `TSL/TSL/PoseMLPostureEngine.swift` | Orquesta distancias → PCA → KNN → 0/1 |
| `TSL/TSL/PoseKNNClassifier.swift` | Voto KNN + `discreteClass` |
| `TSL/TSL/PosePCAReducer.swift` | Reducción de dimensionalidad |
| `TSL/TSL/pca_production.json` | Pesos PCA en bundle |
| `TSL/TSL/knn_production.json` | KNN + datos de entrenamiento en bundle |
| `pipeline/models/knn_model.pkl` | Migración / Python |

## Re-entrenar y actualizar bundle

```bash
cd pipeline
python3 train_knn.py --generate-dataset --use-pca
cp models/knn_production.json ../TSL/TSL/
cp models/pca_production.json ../TSL/TSL/
```

## Logs

En el registro de postura (campana):

```
discreto=0 (derecho) ml=true conf=0.92
discreto=1 (inclinado_jorobado) ml=true conf=0.88
```
