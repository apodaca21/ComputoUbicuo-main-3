//
//  PoseMLPostureEngine.swift
//  TSL — US/14 salida discreta binaria (0 = Seguro, 1 = Inseguro)
//  Override geométrico: |neck.x − spineRoot.x| (sin ángulos, sin Core ML)
//

import CoreGraphics
import Foundation
import Vision

/// Salida discreta del modelo (US-14).
enum DiscretePostureClass: Int {
    case seguro = 0
    case inseguro = 1

    var isUnsafe: Bool { self == .inseguro }

    var displayName: String {
        switch self {
        case .seguro: return "Seguro"
        case .inseguro: return "Inseguro"
        }
    }
}

struct DiscretePostureResult {
    /// Entero 0 o 1 (requerimiento US-14).
    let discreteClass: Int
    let posture: DiscretePostureClass
    let confidence: Float
    let estimatedJointCount: Int
    let usedMLModel: Bool
    let problemJoints: Set<VNHumanBodyPoseObservation.JointName>

    var isPoorPosture: Bool { posture == .inseguro }
}

// MARK: - Override geométrico de vector (simetría forzada con abs)

/// Única regla del semáforo: desviación horizontal cuello–cadera en espacio Vision normalizado.
enum PoseSpineVectorSafetyOverride {
    /// 15 % del ancho normalizado de la vista (0…1).
    static let safetyThresholdX: CGFloat = 0.15

    private static let minNeckConfidence: Float = 0.10
    private static let minRootConfidence: Float = 0.10
    private static let minHipConfidence: Float = 0.10

    /// `neck.x − spineRoot.x` (Vision, origen abajo-izquierda; sin espejo de preview).
    static func lateralDeviationX(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> CGFloat? {
        guard let neck = joints[.neck], neck.confidence >= minNeckConfidence,
              let spineRoot = resolveSpineRoot(joints: joints) else {
            return nil
        }
        return neck.location.x - spineRoot.location.x
    }

    /// Simetría L/R: solo importa la magnitud horizontal, no el signo.
    static func isPoorPosture(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> Bool {
        guard let deviationX = lateralDeviationX(joints: joints) else { return false }
        return abs(deviationX) > safetyThresholdX
    }

    static func classify(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> DiscretePostureResult {
        let jointCount = joints.count
        let deviationX = lateralDeviationX(joints: joints)
        let isUnsafe = !isPoorPosture(joints: joints)

        let discrete = isUnsafe
            ? DiscretePostureClass.inseguro.rawValue
            : DiscretePostureClass.seguro.rawValue

        let confidence: Float
        if let dx = deviationX {
            let margin = abs(dx) - safetyThresholdX
            confidence = isUnsafe ? min(0.92, 0.72 + Float(margin) * 2.5) : 0.85
        } else {
            confidence = 0.5
        }

        return DiscretePostureResult(
            discreteClass: discrete,
            posture: DiscretePostureClass(rawValue: discrete) ?? .seguro,
            confidence: confidence,
            estimatedJointCount: jointCount,
            usedMLModel: false,
            problemJoints: isUnsafe ? problemJoints(joints: joints) : []
        )
    }

    private static func resolveSpineRoot(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> PoseJointSample? {
        if let root = joints[.root], root.confidence >= minRootConfidence {
            return root
        }
        guard let lh = joints[.leftHip], let rh = joints[.rightHip],
              lh.confidence >= minHipConfidence, rh.confidence >= minHipConfidence else {
            return nil
        }
        return PoseJointSample(
            location: CGPoint(
                x: (lh.location.x + rh.location.x) / 2,
                y: (lh.location.y + rh.location.y) / 2
            ),
            confidence: min(lh.confidence, rh.confidence),
            isFrozenForInference: lh.isFrozenForInference && rh.isFrozenForInference
        )
    }

    static func problemJoints(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> Set<VNHumanBodyPoseObservation.JointName> {
        var problems: Set<VNHumanBodyPoseObservation.JointName> = [.neck]
        if joints[.root] != nil { problems.insert(.root) }
        if joints[.leftHip] != nil { problems.insert(.leftHip) }
        if joints[.rightHip] != nil { problems.insert(.rightHip) }
        return problems
    }
}

// MARK: - Motor de postura (API del pipeline / semáforo)

final class PoseMLPostureEngine {
    static let shared = PoseMLPostureEngine()

    private(set) var statusMessage: String =
        "Postura: |neck.x − cadera.x| > 15 % (vector simétrico)"

    private init() {}

    /// Semáforo UI: únicamente `abs(neck.x − spineRoot.x) > safetyThresholdX`.
    func classify(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> DiscretePostureResult {
        PoseSpineVectorSafetyOverride.classify(joints: joints)
    }
}

// MARK: - Zonas problemáticas (resaltado amarillo)

enum PostureProblemDetector {
    static func problemJoints(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> Set<VNHumanBodyPoseObservation.JointName> {
        PoseSpineVectorSafetyOverride.problemJoints(joints: joints)
    }
}
