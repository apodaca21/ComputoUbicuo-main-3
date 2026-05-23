//
//  PoseKalmanMOT.swift
//  TSL
//
//  Seguimiento multi-persona (MOT) + Kalman por articulación.
//  Durante oclusión: posición anclada (sin predicción) para inferencia Core ML estable.
//

import AVFoundation
import CoreGraphics
import Foundation
import Vision

/// Articulaciones que actualizamos con Kalman (orden fijo).
private let allPoseJoints: [VNHumanBodyPoseObservation.JointName] = [
    .nose, .leftEye, .rightEye, .leftEar, .rightEar,
    .neck,
    .leftShoulder, .rightShoulder,
    .leftElbow, .rightElbow,
    .leftWrist, .rightWrist,
    .leftHip, .rightHip,
    .root,
    .leftKnee, .rightKnee,
    .leftAnkle, .rightAnkle
]

// MARK: - Kalman 2D (velocidad constante; congelamiento explícito)

/// Estado [px, py, vx, vy] en espacio normalizado Vision.
private struct KalmanFilter2D {
    private var x: [Double] = [0, 0, 0, 0]
    private var P: [[Double]]
    private var initialized = false

    private let qPos: Double
    private let qVel: Double
    private let rMeas: Double

    init(qPos: Double = 2e-4, qVel: Double = 5e-3, rMeas: Double = 4e-4) {
        self.qPos = qPos
        self.qVel = qVel
        self.rMeas = rMeas
        P = (0..<4).map { i in (0..<4).map { j in i == j ? (i < 2 ? 1.0 : 10.0) : 0.0 } }
    }

    /// Predicción estándar (solo articulaciones con medición activa en este frame).
    mutating func predict(dt: Double) {
        guard initialized else { return }
        let dt2 = dt * dt
        let dt3 = dt2 * dt
        let dt4 = dt3 * dt

        let f: [[Double]] = [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        x = matVecMul(f, x)

        let q00 = qPos * (dt4 / 4.0)
        let q02 = qPos * (dt3 / 2.0)
        let q22 = qPos * dt2
        let q33 = qVel * dt2

        let Q: [[Double]] = [
            [q00, 0, q02, 0],
            [0, q00, 0, q02],
            [q02, 0, q22, 0],
            [0, q02, 0, q33]
        ]
        let FPFT = matMul(matMul(f, P), transpose(f))
        P = matAdd(FPFT, Q)
    }

    mutating func update(measurement mx: Double, _ my: Double) {
        let z = [mx, my]
        if !initialized {
            x = [mx, my, 0, 0]
            P = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 10, 0],
                [0, 0, 0, 10]
            ]
            initialized = true
            return
        }

        let H: [[Double]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]
        let R: [[Double]] = [
            [rMeas, 0],
            [0, rMeas]
        ]

        let Hx = matVecMul(H, x)
        var y = [Double](repeating: 0, count: 2)
        for i in 0..<2 { y[i] = z[i] - Hx[i] }

        let HP = matMul(H, P)
        let HPT = matMul(HP, transpose(H))
        let S = matAdd(HPT, R)
        let Sinv = invert2x2(S) ?? [[1.0 / rMeas, 0], [0, 1.0 / rMeas]]

        let PHT = matMul(P, transpose(H))
        let K = matMul(PHT, Sinv)

        let Ky = matVecMul(K, y)
        for i in 0..<4 { x[i] += Ky[i] }

        let KH = matMul(K, H)
        let I = identity4()
        let IKH = matSub(I, KH)
        P = matMul(IKH, P)
    }

    /// Innovación limitada para evitar teletransporte al recuperar Vision.
    mutating func updateCapped(measurement mx: Double, _ my: Double, maxStep: Double) {
        guard initialized else {
            update(measurement: mx, my)
            return
        }
        let dx = mx - x[0]
        let dy = my - x[1]
        let dist = hypot(dx, dy)
        if dist > maxStep, dist > 1e-9 {
            let s = maxStep / dist
            update(measurement: x[0] + dx * s, x[1] + dy * s)
        } else {
            update(measurement: mx, my)
        }
    }

    func position() -> CGPoint {
        CGPoint(x: x[0], y: x[1])
    }

    mutating func forceZeroVelocity() {
        guard initialized else { return }
        x[2] = 0
        x[3] = 0
    }

    mutating func setPosition(_ px: Double, _ py: Double) {
        if !initialized {
            x = [px, py, 0, 0]
            initialized = true
            return
        }
        x[0] = px
        x[1] = py
        x[2] = 0
        x[3] = 0
    }

    func velocityMagnitude() -> Double {
        hypot(x[2], x[3])
    }

    var hasEstimate: Bool { initialized }
}

// MARK: - Estado por articulación

private struct JointTrackState {
    var filter: KalmanFilter2D
    var framesSinceMeasurement: Int = 0
    var lastMeasuredConfidence: Float = 0.5
    /// Última medición fuerte — ancla para Core ML / overlay durante oclusión.
    var lastStrongPosition: CGPoint?
    var frozenAnchor: CGPoint?
    var isFrozen: Bool = false
    var recoveryFrames: Int = 0
}

// MARK: - Álgebra mínima

private func identity4() -> [[Double]] {
    (0..<4).map { i in (0..<4).map { j in i == j ? 1.0 : 0.0 } }
}

private func transpose(_ a: [[Double]]) -> [[Double]] {
    guard !a.isEmpty else { return a }
    let r = a.count, c = a[0].count
    return (0..<c).map { j in (0..<r).map { i in a[i][j] } }
}

private func matVecMul(_ a: [[Double]], _ v: [Double]) -> [Double] {
    a.map { row in zip(row, v).map(*).reduce(0, +) }
}

private func matMul(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
    let ar = a.count, ac = a[0].count, bc = b[0].count
    var out = [[Double]](repeating: [Double](repeating: 0, count: bc), count: ar)
    for i in 0..<ar {
        for j in 0..<bc {
            var s = 0.0
            for k in 0..<ac { s += a[i][k] * b[k][j] }
            out[i][j] = s
        }
    }
    return out
}

private func matAdd(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
    zip(a, b).map { zip($0, $1).map(+) }
}

private func matSub(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
    zip(a, b).map { zip($0, $1).map(-) }
}

private func invert2x2(_ m: [[Double]]) -> [[Double]]? {
    let a = m[0][0], b = m[0][1], c = m[1][0], d = m[1][1]
    let det = a * d - b * c
    guard abs(det) > 1e-12 else { return nil }
    return [
        [d / det, -b / det],
        [-c / det, a / det]
    ]
}

// MARK: - Medición de pose (salida hacia Core ML y overlay)

struct PoseJointSample {
    let location: CGPoint
    let confidence: Float
    /// true = coordenada anclada (sin predicción Kalman); inferencia debe confiar en location.
    let isFrozenForInference: Bool
}

// MARK: - Pista corporal

private final class BodyPoseTrack {
    let id: Int
    private var jointStates: [VNHumanBodyPoseObservation.JointName: JointTrackState] = [:]
    var missedFrames: Int = 0

    private let maxHoldFrames = 50
    private let heldDisplayConfidence: Float = 0.28
    /// Velocidad normalizada bajo la cual se considera “casi estático”.
    private let staticVelocityThreshold = 0.007
    /// Paso máximo por frame al recuperar Vision (evita salto brusco al modelo).
    private let recoveryMaxStepPerFrame = 0.022
    private let recoveryFramesNeeded = 5

    init(id: Int) {
        self.id = id
    }

    /// Se invoca DESPUÉS de `update` (evita un frame de predicción tras perder Vision).
    func advanceMotion(dt: Double) {
        for joint in Array(jointStates.keys) {
            guard var state = jointStates[joint] else { continue }

            if state.isFrozen || state.framesSinceMeasurement > 0 {
                applyStrictFreeze(&state)
                jointStates[joint] = state
                continue
            }

            state.filter.predict(dt: dt)
            if state.filter.velocityMagnitude() < staticVelocityThreshold {
                state.filter.forceZeroVelocity()
            }
            jointStates[joint] = state
        }
    }

    func update(from observation: VNHumanBodyPoseObservation, confidenceThreshold: Float) {
        missedFrames = 0
        var measured = Set<VNHumanBodyPoseObservation.JointName>()

        for joint in allPoseJoints {
            guard let p = try? observation.recognizedPoint(joint) else { continue }

            let conf = p.confidence
            let isStrong = conf >= confidenceThreshold
            guard isStrong else { continue }

            measured.insert(joint)
            let lx = Double(p.location.x)
            let ly = Double(p.location.y)
            let measuredPoint = CGPoint(x: p.location.x, y: p.location.y)

            if jointStates[joint] == nil {
                jointStates[joint] = JointTrackState(filter: KalmanFilter2D())
            }
            var state = jointStates[joint]!

            if state.isFrozen {
                applySmoothRecovery(
                    state: &state,
                    measurement: measuredPoint,
                    lx: lx,
                    ly: ly,
                    confidence: conf
                )
            } else {
                state.filter.update(measurement: lx, ly)
                if state.filter.velocityMagnitude() < staticVelocityThreshold {
                    state.filter.forceZeroVelocity()
                }
                state.recoveryFrames = 0
            }

            state.framesSinceMeasurement = 0
            state.lastMeasuredConfidence = conf
            state.lastStrongPosition = measuredPoint
            jointStates[joint] = state
        }

        for joint in Array(jointStates.keys) where !measured.contains(joint) {
            guard var state = jointStates[joint] else { continue }

            if state.framesSinceMeasurement == 0 {
                beginStrictFreeze(&state)
            } else {
                applyStrictFreeze(&state)
            }

            state.framesSinceMeasurement += 1
            if state.framesSinceMeasurement > maxHoldFrames {
                jointStates.removeValue(forKey: joint)
            } else {
                jointStates[joint] = state
            }
        }
    }

    /// Al perder el punto: velocidad → 0 y ancla en última posición fuerte.
    private func beginStrictFreeze(_ state: inout JointTrackState) {
        let wasStatic = state.filter.velocityMagnitude() < staticVelocityThreshold
        state.filter.forceZeroVelocity()

        let anchor = state.lastStrongPosition ?? state.filter.position()
        state.frozenAnchor = anchor
        state.isFrozen = true
        state.recoveryFrames = 0

        if wasStatic {
            state.filter.forceZeroVelocity()
        }
        state.filter.setPosition(Double(anchor.x), Double(anchor.y))
    }

    /// Cada frame sin medición: reescribe estado Kalman al ancla (cero deriva).
    private func applyStrictFreeze(_ state: inout JointTrackState) {
        guard let anchor = state.frozenAnchor ?? state.lastStrongPosition else { return }
        state.filter.setPosition(Double(anchor.x), Double(anchor.y))
        state.filter.forceZeroVelocity()
        state.isFrozen = true
    }

    /// Ramp hacia la nueva medición (sin teletransporte para Core ML).
    private func applySmoothRecovery(
        state: inout JointTrackState,
        measurement: CGPoint,
        lx: Double,
        ly: Double,
        confidence: Float
    ) {
        let anchor = state.frozenAnchor ?? state.lastStrongPosition ?? state.filter.position()
        state.recoveryFrames += 1
        let t = min(1.0, 0.32 + 0.14 * Double(state.recoveryFrames))
        let tx = Double(anchor.x) + t * (lx - Double(anchor.x))
        let ty = Double(anchor.y) + t * (ly - Double(anchor.y))

        state.filter.updateCapped(measurement: tx, ty, maxStep: recoveryMaxStepPerFrame)
        state.filter.forceZeroVelocity()

        let settled = hypot(measurement.x - anchor.x, measurement.y - anchor.y) < 0.012
        if state.recoveryFrames >= recoveryFramesNeeded || settled {
            state.isFrozen = false
            state.frozenAnchor = nil
            state.recoveryFrames = 0
            state.filter.updateCapped(measurement: lx, ly, maxStep: recoveryMaxStepPerFrame)
            state.lastMeasuredConfidence = confidence
        }
    }

    func associationCentroid() -> CGPoint? {
        let keys: [VNHumanBodyPoseObservation.JointName] = [
            .root, .leftHip, .rightHip, .leftShoulder, .rightShoulder, .neck, .nose
        ]
        var sx = 0.0, sy = 0.0, n = 0.0
        for k in keys {
            guard let state = jointStates[k] else { continue }
            let c = inferencePosition(for: state)
            sx += Double(c.x)
            sy += Double(c.y)
            n += 1
        }
        guard n > 0 else { return nil }
        return CGPoint(x: sx / n, y: sy / n)
    }

    /// Coordenadas estables para Core ML, geometría y dibujo (ancla si está congelado).
    func smoothedJoints() -> [VNHumanBodyPoseObservation.JointName: PoseJointSample] {
        var out: [VNHumanBodyPoseObservation.JointName: PoseJointSample] = [:]
        for (joint, state) in jointStates where state.filter.hasEstimate || state.frozenAnchor != nil {
            let pt = inferencePosition(for: state)
            let frozen = state.isFrozen || state.framesSinceMeasurement > 0
            let conf: Float
            if !frozen {
                conf = max(state.lastMeasuredConfidence, heldDisplayConfidence)
            } else {
                conf = heldDisplayConfidence
            }
            out[joint] = PoseJointSample(
                location: pt,
                confidence: conf,
                isFrozenForInference: frozen
            )
        }
        return out
    }

    private func inferencePosition(for state: JointTrackState) -> CGPoint {
        if state.isFrozen || state.framesSinceMeasurement > 0 {
            return state.frozenAnchor ?? state.lastStrongPosition ?? state.filter.position()
        }
        if state.filter.velocityMagnitude() < staticVelocityThreshold,
           let strong = state.lastStrongPosition {
            return strong
        }
        return state.filter.position()
    }

    func markMissed() {
        missedFrames += 1
    }
}

// MARK: - Rastreador multi-objeto

final class MultiBodyPoseKalmanTracker {
    private var tracks: [BodyPoseTrack] = []
    private var nextId = 1
    private var lastTimestamp: CFTimeInterval?
    private let associationThreshold: CGFloat = 0.22
    private let maxMissedFrames = 18
    private let maxTracks = 6
    private let jointConfidenceForUpdate: Float = 0.18

    func reset() {
        tracks.removeAll()
        nextId = 1
        lastTimestamp = nil
    }

    func update(
        observations: [VNHumanBodyPoseObservation],
        sampleBuffer: CMSampleBuffer
    ) -> [[VNHumanBodyPoseObservation.JointName: PoseJointSample]] {
        let now = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let dt: Double
        if let last = lastTimestamp, now > last {
            dt = min(now - last, 0.25)
        } else {
            dt = 1.0 / 30.0
        }
        lastTimestamp = now

        let obsCentroids: [CGPoint] = observations.map { centroidForAssociation($0) }
        var matchedTrack = Set<Int>()
        var matchedObs = Set<Int>()

        var pairs: [(cost: CGFloat, ti: Int, oi: Int)] = []
        for (ti, track) in tracks.enumerated() {
            guard let pc = track.associationCentroid() else { continue }
            for (oi, oc) in obsCentroids.enumerated() {
                let dx = pc.x - oc.x, dy = pc.y - oc.y
                let dist = sqrt(dx * dx + dy * dy)
                if dist < associationThreshold {
                    pairs.append((dist, ti, oi))
                }
            }
        }
        pairs.sort { $0.cost < $1.cost }
        for p in pairs {
            if matchedTrack.contains(p.ti) || matchedObs.contains(p.oi) { continue }
            matchedTrack.insert(p.ti)
            matchedObs.insert(p.oi)
            tracks[p.ti].update(from: observations[p.oi], confidenceThreshold: jointConfidenceForUpdate)
        }

        for (oi, obs) in observations.enumerated() where !matchedObs.contains(oi) {
            guard tracks.count < maxTracks else { break }
            let t = BodyPoseTrack(id: nextId)
            nextId += 1
            t.update(from: obs, confidenceThreshold: jointConfidenceForUpdate)
            tracks.append(t)
        }

        for (ti, _) in tracks.enumerated() where !matchedTrack.contains(ti) {
            tracks[ti].markMissed()
        }

        tracks.removeAll { $0.missedFrames > maxMissedFrames }

        for t in tracks {
            t.advanceMotion(dt: dt)
        }

        return tracks.map { $0.smoothedJoints() }
    }
}

// MARK: - Utilidades públicas

func centroidForAssociation(_ observation: VNHumanBodyPoseObservation) -> CGPoint {
    let keys: [VNHumanBodyPoseObservation.JointName] = [
        .root, .leftHip, .rightHip, .leftShoulder, .rightShoulder, .neck, .nose
    ]
    var sx = 0.0, sy = 0.0, n = 0.0
    for k in keys {
        guard let p = try? observation.recognizedPoint(k), p.confidence > Float(0.2) else { continue }
        sx += Double(p.location.x)
        sy += Double(p.location.y)
        n += 1
    }
    if n == 0 {
        return CGPoint(x: 0.5, y: 0.5)
    }
    return CGPoint(x: sx / n, y: sy / n)
}

func primaryTrackIndex(_ tracks: [[VNHumanBodyPoseObservation.JointName: PoseJointSample]]) -> Int {
    guard !tracks.isEmpty else { return 0 }
    let center = CGPoint(x: 0.5, y: 0.5)
    var best = 0
    var bestDist = CGFloat.greatestFiniteMagnitude
    for (i, joints) in tracks.enumerated() {
        let pts = joints.values.map(\.location)
        guard !pts.isEmpty else { continue }
        let mx = pts.map(\.x).reduce(0, +) / CGFloat(pts.count)
        let my = pts.map(\.y).reduce(0, +) / CGFloat(pts.count)
        let c = CGPoint(x: mx, y: my)
        let d = hypot(c.x - center.x, c.y - center.y)
        if d < bestDist {
            bestDist = d
            best = i
        }
    }
    return best
}
