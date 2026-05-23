# US-09 — Live Video Streaming iPad → iPhone

## Historia de usuario

Como usuario, quiero ver la transmisión en vivo de la cámara del iPad en mi iPhone para monitorear el entorno de forma remota.

## Criterios de aceptación

| Criterio | Implementación |
|----------|----------------|
| Delay &lt; 300 ms | `LiveStreamLatencyTracker` muestra latencia en iPhone; badge verde si &lt; 300 ms |
| Stream estable | Multipeer `.unreliable` + descarte de frames tardíos; reconexión al perder peer |
| Preview continuo | `LiveStreamViewerView` actualiza `latestFrame` en cada paquete recibido |

## Tareas técnicas

| Tarea | Estado | Archivo |
|-------|--------|---------|
| Streaming Multipeer | ✅ | `LiveStreamMultipeerService.swift` |
| Optimizar compresión | ✅ | `LiveStreamConfig` + `LiveStreamFrameCodec` (640px, JPEG 0.52, 15 FPS) |
| Validar &lt; 300 ms | ✅ | `LiveStreamLatencyTracker` + ping/pong RTT |
| Pruebas de estabilidad | ✅ | Contadores frames enviados/perdidos; ver sección Pruebas |

## Arquitectura

```
[iPad Host]  AVCapture → JPEG → MCSession (unreliable)
                    ↓
            Wi-Fi / Bluetooth (Multipeer)
                    ↓
[iPhone Viewer]  decode UIImage → preview + latencia
```

## Uso

1. **iPad:** abrir app → **Transmitir cámara (iPad)** → aceptar permiso red local si iOS lo pide.
2. **iPhone:** misma app → **Ver transmisión en vivo** → se conecta automáticamente al iPad.
3. En iPhone revisar badge **Latencia estimada** (objetivo &lt; 300 ms).

## Parámetros de compresión (ajustables)

```swift
// LiveStreamConfig.swift
maxFrameLongSide = 640
jpegQuality = 0.52
maxSendFPS = 15
targetLatencyMs = 300
```

## Pruebas de estabilidad (manual)

1. Mantener transmisión **5 minutos** — el preview no debe congelarse.
2. Alejar iPhone del iPad y volver — debe reconectar o mostrar “Buscando iPad…”.
3. Anotar en iPhone: latencia media, fps y frames perdidos.
4. Si latencia &gt; 300 ms: bajar `maxFrameLongSide` a 480 o `jpegQuality` a 0.45.

## Permisos Info.plist

- `NSLocalNetworkUsageDescription`
- `NSBonjourServices`: `_tsl-live._tcp`
- `NSCameraUsageDescription` (host en iPad)

## Notas

- Ambos dispositivos deben tener la **misma app** instalada y estar en la **misma red Wi‑Fi** (o Bluetooth cercano).
- WebRTC no se usa; Multipeer es nativo y suficiente para demo en LAN con baja latencia.
