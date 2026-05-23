//
//  LiveStreamConfig.swift
//  TSL — US-09 parámetros de compresión y latencia
//

import Foundation

enum LiveStreamConfig {
    /// Tipo de servicio Multipeer (≤15 caracteres, minúsculas).
    static let serviceType = "tsl-live"

    /// Criterio de aceptación US-09.
    static let targetLatencyMs: Double = 300

    /// FPS máximo de envío (reduce carga y ayuda a mantener <300 ms).
    static let maxSendFPS: Double = 15

    /// Lado largo máximo del frame JPEG (compresión espacial).
    static let maxFrameLongSide: CGFloat = 640

    /// Calidad JPEG 0…1 (optimización de compresión).
    static let jpegQuality: CGFloat = 0.52

    /// Intervalo de ping para estimar offset de reloj entre dispositivos (s).
    static let pingIntervalSeconds: TimeInterval = 2.0
}
