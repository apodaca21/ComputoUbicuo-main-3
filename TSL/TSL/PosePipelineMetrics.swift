//
//  PosePipelineMetrics.swift
//  TSL
//
//  Métricas del pipeline: tiempos por fase, delay frame→decisión, tasa de decisión
//  discretizada según carga, estimación de puntos/clase y detección de multitudes.
//

import AVFoundation
import CoreGraphics
import Foundation
import Vision

// MARK: - Clase de postura (estimación)

enum PostureClassEstimate: String {
    case none
    case good
    case bad

    var label: String {
        switch self {
        case .none: return "sin_persona"
        case .good: return "derecho"
        case .bad: return "inclinado_jorobado"
        }
    }
}

struct PostureClassificationResult {
    let postureClass: PostureClassEstimate
    /// US-14: 0 = Seguro, 1 = Inseguro (sin persona → -1).
    let discreteClass: Int
    let confidence: Float
    let estimatedJointCount: Int
    let usedMLModel: Bool
    /// Puntos a marcar en amarillo (hombros, cuello, etc.).
    let problemJoints: Set<VNHumanBodyPoseObservation.JointName>

    static let noPersonDiscrete = -1
}

// MARK: - Coordinadas Vision → pantalla (un solo camino, sin doble espejo)

enum PoseCoordinateMapper {
    /// Vision usa origen abajo-izquierda; el preview ya aplica espejo en cámara frontal vía `layerRectConverted`.
    static func visionToPreviewLayer(
        _ normalized: CGPoint,
        previewLayer: AVCaptureVideoPreviewLayer
    ) -> CGPoint {
        let metadataOrigin = CGPoint(x: normalized.x, y: 1.0 - normalized.y)
        let metadataRect = CGRect(origin: metadataOrigin, size: .zero)
        let layerRect = previewLayer.layerRectConverted(fromMetadataOutputRect: metadataRect)
        return CGPoint(x: layerRect.midX, y: layerRect.midY)
    }
}

// MARK: - Estimación de clase (KNN + PCA, US-14)

enum PosePipelineClassifier {
    private static let minJointsForDecision = 5

    static func classify(
        tracks: [[VNHumanBodyPoseObservation.JointName: PoseJointSample]],
        primaryIndex: Int
    ) -> PostureClassificationResult {
        guard !tracks.isEmpty, tracks.indices.contains(primaryIndex) else {
            return PostureClassificationResult(
                postureClass: .none,
                discreteClass: PostureClassificationResult.noPersonDiscrete,
                confidence: 0,
                estimatedJointCount: 0,
                usedMLModel: false,
                problemJoints: []
            )
        }

        let joints = tracks[primaryIndex]
        guard joints.count >= minJointsForDecision else {
            return PostureClassificationResult(
                postureClass: .none,
                discreteClass: PostureClassificationResult.noPersonDiscrete,
                confidence: 0,
                estimatedJointCount: joints.count,
                usedMLModel: false,
                problemJoints: []
            )
        }

        let ml = PoseMLPostureEngine.shared.classify(joints: joints)
        let postureClass: PostureClassEstimate = ml.isPoorPosture ? .bad : .good

        return PostureClassificationResult(
            postureClass: postureClass,
            discreteClass: ml.discreteClass,
            confidence: ml.confidence,
            estimatedJointCount: ml.estimatedJointCount,
            usedMLModel: ml.usedMLModel,
            problemJoints: ml.problemJoints
        )
    }
}

// MARK: - Tiempos de fase y scheduler de decisiones

struct PosePhaseTimings {
    var visionSeconds: Double = 0
    var trackerSeconds: Double = 0
    var postureSeconds: Double = 0
    var frameDelaySeconds: Double = 0

    var heaviestPhaseSeconds: Double {
        max(visionSeconds, trackerSeconds, postureSeconds)
    }
}

/// Discretiza el intervalo entre decisiones de postura según el tiempo de las fases pesadas.
final class PoseDecisionScheduler {
    /// Bucket mínimo de discretización (50 ms).
    private let discretizationStep: TimeInterval = 0.05
    private let safetyFactor: Double = 1.25
    private let minInterval: TimeInterval = 1.0 / 30.0
    private let maxInterval: TimeInterval = 0.5

    private(set) var decisionInterval: TimeInterval = 1.0 / 15.0
    private var lastDecisionWallTime: CFAbsoluteTime = 0
    private var emaHeavyPhase: Double = 1.0 / 30.0
    private let emaAlpha = 0.2

    var decisionsPerSecond: Double {
        guard decisionInterval > 0 else { return 0 }
        return 1.0 / decisionInterval
    }

    func recordPhaseTimings(_ timings: PosePhaseTimings) {
        let heavy = timings.heaviestPhaseSeconds
        emaHeavyPhase = emaAlpha * heavy + (1 - emaAlpha) * emaHeavyPhase

        let bucketed = ceil(emaHeavyPhase / discretizationStep) * discretizationStep
        let raw = bucketed * safetyFactor
        decisionInterval = min(maxInterval, max(minInterval, raw))
    }

    func shouldEmitPostureDecision(at wallTime: CFAbsoluteTime) -> Bool {
        if lastDecisionWallTime == 0 || wallTime - lastDecisionWallTime >= decisionInterval {
            lastDecisionWallTime = wallTime
            return true
        }
        return false
    }

    func reset() {
        lastDecisionWallTime = 0
        decisionInterval = 1.0 / 15.0
        emaHeavyPhase = 1.0 / 30.0
    }
}

// MARK: - Multitudes y registro periódico

final class PosePipelineLogger {
    private let crowdThreshold: Int
    private var lastMetricsLogTime: CFAbsoluteTime = 0
    private let metricsLogInterval: TimeInterval = 8
    private var lastCrowdLogged = false

    init(crowdThreshold: Int = 2) {
        self.crowdThreshold = crowdThreshold
    }

    func reset() {
        lastMetricsLogTime = 0
        lastCrowdLogged = false
    }

    func process(
        bridge: PoseCameraBridge?,
        trackCount: Int,
        timings: PosePhaseTimings,
        scheduler: PoseDecisionScheduler,
        classification: PostureClassificationResult,
        at wallTime: CFAbsoluteTime
    ) {
        guard let bridge else { return }

        if trackCount >= crowdThreshold {
            if !lastCrowdLogged {
                lastCrowdLogged = true
                bridge.appendLog(
                    "Varias personas detectadas (\(trackCount)) — postura evaluada solo en la pista principal"
                )
            }
        } else {
            lastCrowdLogged = false
        }

        guard wallTime - lastMetricsLogTime >= metricsLogInterval else { return }
        lastMetricsLogTime = wallTime

        let delayMs = Int(timings.frameDelaySeconds * 1000)
        let visionMs = Int(timings.visionSeconds * 1000)
        let trackerMs = Int(timings.trackerSeconds * 1000)
        let postureMs = Int(timings.postureSeconds * 1000)
        let intervalMs = Int(scheduler.decisionInterval * 1000)
        let rate = String(format: "%.1f", scheduler.decisionsPerSecond)

        bridge.appendLog(
            """
            Pipeline | delay=\(delayMs)ms vision=\(visionMs)ms tracker=\(trackerMs)ms posture=\(postureMs)ms \
            | decisión cada \(intervalMs)ms (~\(rate)/s) \
            | puntos=\(classification.estimatedJointCount) discreto=\(classification.discreteClass) \
            (\(classification.postureClass.label)) ml=\(classification.usedMLModel) \
            conf=\(String(format: "%.2f", classification.confidence)) personas=\(trackCount)
            """
        )
    }
}
