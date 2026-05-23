# Pipeline US/11 — Procesamiento y normalización de coordenadas

Implementa las subtareas de la historia de usuario:

1. **Cargar CSV** — `loader.py` + `process_pose_csv.py`
2. **Filtrado espacial** — elimina rostro, rodillas y tobillos (`spatial_filter.py`)
3. **Min-Max Scaling** — normaliza distancias del tren superior (`minmax_scaling.py`)
4. **Documentación de tensores** — [`../docs/TENSORES_ENTRADA.md`](../docs/TENSORES_ENTRADA.md)

## Requisitos

- Python 3.10+
- `pip install -r requirements.txt`

## Uso rápido

```bash
python process_pose_csv.py data/sample_pose_recording.csv
```

Con tu propia grabación:

```bash
python process_pose_csv.py /ruta/a/pose_recording.csv --output-dir output
```

## Estructura

```
pipeline/
├── config.py              # Articulaciones, aristas, columnas CSV
├── loader.py              # Carga y validación del CSV
├── spatial_filter.py      # Subset selection
├── features.py            # Distancias + tensor (S, T, F)
├── minmax_scaling.py      # Min-Max por feature
├── process_pose_csv.py    # Script principal (CLI)
├── data/
│   └── sample_pose_recording.csv
└── output/                # Generado al ejecutar (gitignored)
```

## US/12 — PCA (reducción de dimensionalidad)

```bash
python train_pca.py data/training_pose_recording.csv --run-pipeline-first
python apply_pca.py output/03_distances_minmax.csv
```

Documentación: [`../docs/US12_PCA.md`](../docs/US12_PCA.md)

Copia `models/pca_production.json` a `TSL/TSL/` para usar `PosePCAReducer` en iOS.

## US/13 — KNN (clasificación de postura)

```bash
python3 generate_labeled_dataset.py
python3 train_knn.py --generate-dataset
python3 train_knn.py --use-pca
```

Documentación: [`../docs/US13_KNN.md`](../docs/US13_KNN.md)

Copia `models/knn_production.json` a `TSL/TSL/` para `PoseKNNClassifier`.

## Exportar CSV desde la app iOS

La app TSL puede guardar `pose_recording.csv` en Documentos del dispositivo (ver `PoseCSVRecorder.swift`). Comparte el archivo por AirDrop o Files y pásalo a este pipeline.
