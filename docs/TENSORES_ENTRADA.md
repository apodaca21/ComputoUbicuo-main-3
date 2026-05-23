# Formato de tensores de entrada (US/11)

Pipeline: **CSV grabado → filtrado espacial → distancias tren superior → Min-Max → tensor NumPy**.

## 1. CSV de entrada (datos grabados)

| Columna      | Tipo   | Descripción |
|-------------|--------|-------------|
| `timestamp` | float  | Segundos desde inicio de la grabación (PTS del frame) |
| `frame_id`  | int    | Índice de frame (0, 1, 2, …) |
| `track_id`  | int    | Identificador de persona (MOT / Kalman) |
| `joint`     | string | Nombre de articulación (snake_case, ver lista abajo) |
| `x`         | float  | Coordenada X normalizada [0, 1] |
| `y`         | float  | Coordenada Y normalizada [0, 1] |
| `confidence`| float  | Confianza de la detección [0, 1] |

**Una fila = una articulación en un frame para una pista.**

### Nombres de articulaciones (compatibles con Vision / app TSL)

`nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `neck`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `root`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`

---

## 2. Filtrado espacial (Subset Selection)

Se **eliminan** coordenadas irrelevantes:

| Grupo   | Articulaciones eliminadas |
|---------|---------------------------|
| Rostro  | `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear` |
| Piernas | `left_knee`, `right_knee`, `left_ankle`, `right_ankle` |

Se **conservan** (tren superior):

`neck`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `root`

Filas con `confidence < 0.2` (configurable) también se descartan.

---

## 3. Features: distancias entre articulaciones

Por cada `(frame_id, track_id)` se calculan **11 distancias euclidianas** en el plano normalizado:

| Índice | Feature | Par de articulaciones |
|--------|---------|------------------------|
| 0 | `dist_neck__left_shoulder` | cuello – hombro izq. |
| 1 | `dist_neck__right_shoulder` | cuello – hombro der. |
| 2 | `dist_left_shoulder__right_shoulder` | hombros |
| 3 | `dist_left_shoulder__left_elbow` | hombro – codo izq. |
| 4 | `dist_left_elbow__left_wrist` | codo – muñeca izq. |
| 5 | `dist_right_shoulder__right_elbow` | hombro – codo der. |
| 6 | `dist_right_elbow__right_wrist` | codo – muñeca der. |
| 7 | `dist_left_shoulder__left_hip` | hombro – cadera izq. |
| 8 | `dist_right_shoulder__right_hip` | hombro – cadera der. |
| 9 | `dist_left_hip__right_hip` | caderas |
| 10 | `dist_neck__root` | cuello – root |

Fórmula: \( d = \sqrt{(x_a - x_b)^2 + (y_a - y_b)^2} \)

Si falta alguna articulación del par → `NaN` en esa feature.

---

## 4. Min-Max Scaling

Por cada feature (columna de distancias), sobre **todo el dataset**:

\[
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min} + \varepsilon}
\]

con \( \varepsilon = 10^{-8} \), resultado en **[0, 1]**.

Parámetros guardados en `output/minmax_scaler.json` para reproducir la escala en inferencia.

---

## 5. Tensor de entrada final

| Propiedad | Valor |
|-----------|--------|
| **Archivo** | `output/tensor_input.npy` |
| **dtype** | `float64` |
| **Forma** | `(S, T, F)` |
| **S** | Número de secuencias (= cantidad de `track_id` distintos) |
| **T** | Máximo de frames de la pista más larga (frames cortos rellenados con `NaN` al final) |
| **F** | 11 features (distancias normalizadas) |

### Semántica de ejes

```
tensor_input[s, t, f]

  s → persona / pista (track_id)
  t → tiempo (frame_id ordenado ascendente)
  f → distancia normalizada entre un par del tren superior
```

### Ejemplo de carga en Python

```python
import numpy as np

X = np.load("pipeline/output/tensor_input.npy")  # shape (S, T, 11)
# X[s, t, :] = vector de postura del tren superior en el instante t de la persona s
```

### Uso típico downstream

- **Clasificación de postura:** `X` → modelo (MLP, LSTM, etc.) por secuencia `s`.
- **Tasa de decisiones:** usar `timestamp` del CSV o índice `t` con el scheduler de la app iOS.
- **Batch ML:** apilar secuencias del mismo `T` o usar padding/máscara sobre `NaN`.

---

## 6. US/12 — Tensor tras PCA (opcional)

Tras Min-Max, PCA reduce **11 → k** componentes (p. ej. k=3 con 95 % varianza):

| Archivo | Forma |
|---------|--------|
| `models/tensor_pca_reduced.npy` | `(n_muestras, k)` |
| `models/pca_production.json` | Parámetros para Swift (`PosePCAReducer`) |

Ver [`US12_PCA.md`](US12_PCA.md).

---

## 7. Ejecución del pipeline

```bash
cd pipeline
pip install -r requirements.txt
python process_pose_csv.py data/sample_pose_recording.csv
```

Salidas en `pipeline/output/`:

| Archivo | Contenido |
|---------|-----------|
| `01_spatial_filtered.csv` | Coordenadas tras subset selection |
| `02_distances_raw.csv` | Distancias sin escalar |
| `03_distances_minmax.csv` | Distancias en [0, 1] |
| `tensor_raw.npy` | Tensor antes de Min-Max |
| `tensor_input.npy` | **Tensor de entrada final** |
| `minmax_scaler.json` | Parámetros Min-Max |
| `tensor_metadata.json` | Metadatos y forma del tensor |
