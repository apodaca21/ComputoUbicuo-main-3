# Prueba en iPhone — checklist rápida

## Antes de abrir Xcode

1. Conecta el **iPhone físico** (no simulador para cámara).
2. En Xcode: target **TSL** → **Signing** con tu equipo.
3. Verifica que en el bundle estén estos JSON (carpeta TSL):
   - `minmax_scaler.json`
   - `pca_production.json`
   - `knn_production.json`

## Al abrir la cámara

En el registro (campana) debe aparecer:

- `ML listo (MinMax+PCA+KNN) K=1` → clasificación ML activa
- `ML no disponible — modo heurístico` → solo reglas geométricas (revisa JSON en bundle)

## Qué probar

| Prueba | Qué deberías ver |
|--------|------------------|
| **1 persona** | Esqueleto verde, `Personas: 1`, semáforo verde si estás derecho |
| **Postura mala** | Inclínate / joroba → semáforo **rojo**, log `Inseguro (clase=1)` |
| **Postura buena** | De pie frente a cámara → semáforo **verde**, `discreto=0` en logs |
| **2+ personas** | `Personas: 2` (o más), aviso en log; semáforo evalúa la del **centro** |
| **Movimiento brazos** | El esqueleto sigue tu brazo (mismo lado que tu cuerpo en espejo) |
| **Cambiar cámara** | Botón arriba derecha; frontal y trasera |

## Si algo falla

- **Siempre rojo o siempre verde:** abre la campana y mira `ml=true/false`. Si `ml=false`, faltan modelos en el bundle.
- **Esqueleto desfasado:** prueba cámara frontal; buena luz; cuerpo completo en cuadro.
- **No detecta 2 personas:** separa a las personas en el plano; ambos visibles de cintura para arriba.

## Re-entrenar modelos (opcional)

```bash
cd pipeline
python3 train_knn.py --generate-dataset --use-pca
cp models/{minmax_scaler,pca_production,knn_production}.json ../TSL/TSL/
```

Luego **Clean Build** en Xcode (⇧⌘K) e instala de nuevo.
