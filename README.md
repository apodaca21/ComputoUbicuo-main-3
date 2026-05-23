# ComputoUbicuo
The silent coach project - Computo Ubicuo

## Pipeline US/11 (CSV → tensor)

```bash
cd pipeline
pip install -r requirements.txt
python process_pose_csv.py data/sample_pose_recording.csv
```

Documentación del tensor de entrada: [`docs/TENSORES_ENTRADA.md`](docs/TENSORES_ENTRADA.md)

La app iOS guarda `pose_recording.csv` al cerrar la cámara (registro en la campana).

## US-09 — Streaming en vivo iPad → iPhone

- **iPad:** Transmitir cámara (Multipeer)
- **iPhone:** Ver transmisión + medidor de latencia (&lt; 300 ms)

Documentación: [`docs/US09_LIVE_STREAMING.md`](docs/US09_LIVE_STREAMING.md)

## US-12 — PCA (reducción de dimensionalidad)

```bash
cd pipeline && python train_pca.py data/training_pose_recording.csv --run-pipeline-first
```

Documentación: [`docs/US12_PCA.md`](docs/US12_PCA.md)

## US-13 — KNN (postura)

```bash
cd pipeline && python3 train_knn.py --generate-dataset --use-pca
```

Documentación: [`docs/US13_KNN.md`](docs/US13_KNN.md)

## US-14 — Salida discreta 0/1 en app

Clasificación binaria integrada: **0 = Seguro**, **1 = Inseguro**.

Documentación: [`docs/US14_DISCRETE_OUTPUT.md`](docs/US14_DISCRETE_OUTPUT.md)

Guía de prueba en dispositivo: [`docs/PRUEBA_EN_IPHONE.md`](docs/PRUEBA_EN_IPHONE.md)
