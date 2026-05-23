//
//  LiveStreamLatencyTracker.swift
//  TSL — Validación de delay <300 ms (US-09)
//

import Combine
import Foundation

@MainActor
final class LiveStreamLatencyTracker: ObservableObject {
    @Published private(set) var estimatedLatencyMs: Double = 0
    @Published private(set) var decodeDisplayMs: Double = 0
    @Published private(set) var meetsTarget: Bool = false
    @Published private(set) var framesReceived: Int = 0
    @Published private(set) var framesDropped: Int = 0
    @Published private(set) var averageFps: Double = 0
    @Published private(set) var lastRttMs: Double = 0

    private var pingSentAtViewer: [UInt32: TimeInterval] = [:]
    private var lastFrameWallTime: TimeInterval = 0
    private var fpsSamples: [Double] = []

    private let emaAlpha = 0.25

    func recordPingSent(id: UInt32) {
        pingSentAtViewer[id] = Date().timeIntervalSince1970
    }

    func recordPong(id: UInt32) {
        guard let sentAt = pingSentAtViewer.removeValue(forKey: id) else { return }
        lastRttMs = emaAlpha * ((Date().timeIntervalSince1970 - sentAt) * 1000)
            + (1 - emaAlpha) * lastRttMs
    }

    /// Latencia estimada = decode/UI en iPhone + mitad del RTT (ida de red).
    func recordFrameReceived(hostSentEpochMs: UInt64, decodeMs: Double) {
        framesReceived += 1
        let viewerNowMs = Date().timeIntervalSince1970 * 1000
        let wallE2e = viewerNowMs - Double(hostSentEpochMs)
        let networkEstimate = decodeMs + (lastRttMs * 0.5)
        let e2e = min(wallE2e, networkEstimate + 80) // acota relojes desincronizados

        estimatedLatencyMs = emaAlpha * e2e + (1 - emaAlpha) * estimatedLatencyMs
        decodeDisplayMs = emaAlpha * decodeMs + (1 - emaAlpha) * decodeDisplayMs
        meetsTarget = estimatedLatencyMs < LiveStreamConfig.targetLatencyMs

        let now = Date().timeIntervalSince1970
        if lastFrameWallTime > 0 {
            let dt = now - lastFrameWallTime
            if dt > 0 {
                fpsSamples.append(1.0 / dt)
                if fpsSamples.count > 30 { fpsSamples.removeFirst() }
                averageFps = fpsSamples.reduce(0, +) / Double(fpsSamples.count)
            }
        }
        lastFrameWallTime = now
    }

    func recordDroppedFrame() {
        framesDropped += 1
    }

    func reset() {
        estimatedLatencyMs = 0
        decodeDisplayMs = 0
        meetsTarget = false
        framesReceived = 0
        framesDropped = 0
        averageFps = 0
        lastRttMs = 0
        pingSentAtViewer.removeAll()
        lastFrameWallTime = 0
        fpsSamples.removeAll()
    }
}
