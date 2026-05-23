//
//  PoseMLPostureEngine.swift
//  TSL — US/14 salida discreta binaria (0 = Seguro, 1 = Inseguro) vía KNN+PCA
//

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

// MARK: - Distancias tren superior (mismo orden que pipeline/config.py)

private enum UpperBodyDistanceEdges {
    static let pairs: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
        (.neck, .leftShoulder),
        (.neck, .rightShoulder),
        (.leftShoulder, .rightShoulder),
        (.leftShoulder, .leftElbow),
        (.leftElbow, .leftWrist),
        (.rightShoulder, .rightElbow),
        (.rightElbow, .rightWrist),
        (.leftShoulder, .leftHip),
        (.rightShoulder, .rightHip),
        (.leftHip, .rightHip),
        (.neck, .root)
    ]
}

enum PoseUpperBodyDistanceExtractor {
    private static let minConfidence: Float = 0.10
    private static let minValidEdges = 6

    /// 11 distancias en espacio Vision; imputa aristas faltantes si hay ≥7 válidas.
    static func extract(from joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]) -> [Double]? {
        var distances: [Double] = []
        var valid: [Double] = []

        for (a, b) in UpperBodyDistanceEdges.pairs {
            if let pa = joints[a], let pb = joints[b],
               pa.confidence >= minConfidence, pb.confidence >= minConfidence {
                let dx = Double(pa.location.x - pb.location.x)
                let dy = Double(pa.location.y - pb.location.y)
                let d = hypot(dx, dy)
                distances.append(d)
                valid.append(d)
            } else {
                distances.append(-1)
            }
        }

        guard valid.count >= minValidEdges else { return nil }

        let fill = valid.reduce(0, +) / Double(valid.count)
        return distances.map { $0 < 0 ? fill : $0 }
    }
}

// MARK: - Motor ML (KNN + PCA) con fallback heurístico

final class PoseMLPostureEngine {
    static let shared = PoseMLPostureEngine()

    private let minMax: PoseMinMaxScaler?
    private let pca: PosePCAReducer?
    private let knn: PoseKNNClassifier?
    private(set) var loadedML: Bool = false
    private(set) var statusMessage: String = ""

    private init() {
        var minMaxScaler: PoseMinMaxScaler?
        var pcaReducer: PosePCAReducer?
        var knnClassifier: PoseKNNClassifier?

        minMaxScaler = try? PoseMinMaxScaler()
        pcaReducer = try? PosePCAReducer()
        knnClassifier = try? PoseKNNClassifier()

        minMax = minMaxScaler
        pca = pcaReducer
        knn = knnClassifier

        let needsPCA = knnClassifier?.usesPCA == true
        loadedML = minMaxScaler != nil && knnClassifier != nil && (!needsPCA || pcaReducer != nil)

        if PostureCoreMLClassifier.isAvailable {
            statusMessage = "ML CoreML (TSLPostureModel.mlmodel) + geometría"
        } else if loadedML {
            statusMessage = "ML listo (MinMax+PCA+KNN) K=\(knnClassifier?.bestK ?? 0)"
        } else {
            statusMessage = "ML no disponible — modo heurístico (geometría)"
        }
    }

    /// Clasificación: Core ML + geometría + KNN (cualquiera marca inseguro si detecta joroba).
    func classify(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> DiscretePostureResult {
        let jointCount = joints.count
        let legacyPoor = PostureAnalyzerLegacy.isHunchedOrPoorPosture(joints: joints)
        let problems = PostureAnalyzerLegacy.problemJoints(joints: joints)

        var corePoor = false
        var coreConf: Float = 0
        var usedCore = false
        if let core = PostureCoreMLClassifier.predict(joints: joints) {
            usedCore = true
            corePoor = core.isPoorPosture
            coreConf = core.confidence
        }

        var knnPoor = false
        var knnConf: Float = 0
        var usedKnn = false
        if loadedML, let distances = PoseUpperBodyDistanceExtractor.extract(from: joints),
           let knnResult = classifyWithML(distances: distances, jointCount: jointCount) {
            usedKnn = true
            knnPoor = knnResult.isPoorPosture
            knnConf = knnResult.confidence
        }

        let isPoor = legacyPoor || corePoor || knnPoor
        let discrete = isPoor ? DiscretePostureClass.inseguro.rawValue : DiscretePostureClass.seguro.rawValue
        let confidence = max(coreConf, knnConf, isPoor ? 0.72 : 0.68)

        return DiscretePostureResult(
            discreteClass: discrete,
            posture: DiscretePostureClass(rawValue: discrete) ?? .seguro,
            confidence: confidence,
            estimatedJointCount: jointCount,
            usedMLModel: usedCore || usedKnn,
            problemJoints: isPoor ? problems : []
        )
    }

    private func classifyWithML(
        distances: [Double],
        jointCount: Int
    ) -> DiscretePostureResult? {
        guard let knn else { return nil }

        do {
            guard let minMax else { return nil }
            let scaled = try minMax.transform(distances)

            let features: [Double]
            if knn.usesPCA, let pca {
                features = try pca.transform(features: scaled)
            } else {
                features = scaled
            }

            let pred = try knn.predict(features: features)
            let discrete = pred.discreteClass

            return DiscretePostureResult(
                discreteClass: discrete,
                posture: DiscretePostureClass(rawValue: discrete) ?? .seguro,
                confidence: Float(pred.confidence),
                estimatedJointCount: jointCount,
                usedMLModel: true,
                problemJoints: []
            )
        } catch {
            return nil
        }
    }
}

// MARK: - Zonas problemáticas (resaltado amarillo)

enum PostureProblemDetector {
    static func problemJoints(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> Set<VNHumanBodyPoseObservation.JointName> {
        PostureAnalyzerLegacy.problemJoints(joints: joints)
    }
}

// MARK: - Reglas heurísticas (solo fallback; antes era PostureAnalyzer)

private enum PostureAnalyzerLegacy {
    private static let minHip: Float = 0.08
    private static let minShoulder: Float = 0.08
    private static let minNeck: Float = 0.08
    private static let minNose: Float = 0.08

    static func isHunchedOrPoorPosture(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> Bool {
        guard let lh = joints[.leftHip], let rh = joints[.rightHip],
              lh.confidence > minHip, rh.confidence > minHip else { return false }

        let hipX = (lh.location.x + rh.location.x) / 2
        let hipY = (lh.location.y + rh.location.y) / 2

        let upX: CGFloat
        let upY: CGFloat
        if let neck = joints[.neck], neck.confidence > minNeck {
            upX = neck.location.x
            upY = neck.location.y
        } else if let ls = joints[.leftShoulder], let rs = joints[.rightShoulder],
                  ls.confidence > minShoulder, rs.confidence > minShoulder {
            upX = (ls.location.x + rs.location.x) / 2
            upY = (ls.location.y + rs.location.y) / 2
        } else {
            return false
        }

        let vx = upX - hipX
        let vy = upY - hipY
        guard vy > 0.011 else { return false }

        let torsoLen = hypot(vx, vy)
        let trunkLeanFromVertical = abs(atan2(vx, vy))

        var likelySideProfile = false
        if let ls = joints[.leftShoulder], let rs = joints[.rightShoulder],
           ls.confidence > minShoulder, rs.confidence > minShoulder {
            let sw = hypot(rs.location.x - ls.location.x, rs.location.y - ls.location.y)
            if sw < 0.055 {
                likelySideProfile = true
            }
        }

        let leanLimit: CGFloat = likelySideProfile ? 0.24 : 0.12
        if trunkLeanFromVertical > leanLimit { return true }

        if torsoLen > 0.05 {
            let horizontalRatio = abs(vx) / torsoLen
            if horizontalRatio > 0.22 { return true }
        }

        if !likelySideProfile,
           let ls = joints[.leftShoulder], let rs = joints[.rightShoulder],
           ls.confidence > minShoulder, rs.confidence > minShoulder {
            let sw = hypot(rs.location.x - ls.location.x, rs.location.y - ls.location.y)
            if sw > 0.035 {
                let shoulderTilt = abs(ls.location.y - rs.location.y) / sw
                if shoulderTilt > 0.42 { return true }
            }
        }

        if let nose = joints[.nose], let neck = joints[.neck],
           nose.confidence > minNose, neck.confidence > minNeck, torsoLen > 0.03 {
            let headForward = abs(nose.location.x - neck.location.x) / torsoLen
            if headForward > 0.14 { return true }
            if nose.location.y < neck.location.y + 0.08 { return true }
        }

        if let ls = joints[.leftShoulder], let rs = joints[.rightShoulder],
           ls.confidence > minShoulder, rs.confidence > minShoulder {
            let shoulderMidX = (ls.location.x + rs.location.x) / 2
            let shoulderMidY = (ls.location.y + rs.location.y) / 2
            let hipDist = hypot(shoulderMidX - hipX, shoulderMidY - hipY)
            if hipDist > 0.04 && shoulderMidY < hipY + 0.14 { return true }
        }

        return false
    }

    /// Articulaciones a resaltar en amarillo cuando la postura es incorrecta.
    static func problemJoints(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> Set<VNHumanBodyPoseObservation.JointName> {
        var problems = Set<VNHumanBodyPoseObservation.JointName>()
        guard let lh = joints[.leftHip], let rh = joints[.rightHip],
              lh.confidence > minHip, rh.confidence > minHip else { return problems }

        let hipX = (lh.location.x + rh.location.x) / 2
        let hipY = (lh.location.y + rh.location.y) / 2

        let upX: CGFloat
        let upY: CGFloat
        if let neck = joints[.neck], neck.confidence > minNeck {
            upX = neck.location.x
            upY = neck.location.y
        } else if let ls = joints[.leftShoulder], let rs = joints[.rightShoulder],
                  ls.confidence > minShoulder, rs.confidence > minShoulder {
            upX = (ls.location.x + rs.location.x) / 2
            upY = (ls.location.y + rs.location.y) / 2
        } else {
            return problems
        }

        let vx = upX - hipX
        let vy = upY - hipY
        guard vy > 0.011 else { return problems }

        let torsoLen = hypot(vx, vy)
        let trunkLean = abs(atan2(vx, vy))

        var likelySideProfile = false
        if let ls = joints[.leftShoulder], let rs = joints[.rightShoulder],
           ls.confidence > minShoulder, rs.confidence > minShoulder {
            let sw = hypot(rs.location.x - ls.location.x, rs.location.y - ls.location.y)
            if sw < 0.055 { likelySideProfile = true }
        }

        let leanLimit: CGFloat = likelySideProfile ? 0.24 : 0.12
        if trunkLean > leanLimit {
            problems.formUnion([.neck, .root, .leftHip, .rightHip])
            if joints[.root] == nil { problems.remove(.root) }
        }

        if !likelySideProfile,
           let ls = joints[.leftShoulder], let rs = joints[.rightShoulder],
           ls.confidence > minShoulder, rs.confidence > minShoulder {
            let sw = hypot(rs.location.x - ls.location.x, rs.location.y - ls.location.y)
            if sw > 0.042 {
                let shoulderTilt = abs(ls.location.y - rs.location.y) / sw
                if shoulderTilt > 0.52 {
                    problems.formUnion([.leftShoulder, .rightShoulder, .neck])
                }
            }
        }

        if !likelySideProfile,
           let nose = joints[.nose], let neck = joints[.neck],
           nose.confidence > minNose, neck.confidence > minNeck, torsoLen > 0.038 {
            let headForward = abs(nose.location.x - neck.location.x) / torsoLen
            if headForward > 0.27 && nose.location.y < neck.location.y + 0.05 {
                problems.formUnion([.nose, .neck])
            }
        }

        if isHunchedOrPoorPosture(joints: joints) && problems.isEmpty {
            problems.formUnion([.neck, .leftShoulder, .rightShoulder, .nose])
        }

        return problems
    }
}
