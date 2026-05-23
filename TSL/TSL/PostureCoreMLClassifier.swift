//
//  PostureCoreMLClassifier.swift
//  TSL — Clasificador Core ML (coordenadas x,y de 19 articulaciones)
//

import CoreML
import Foundation
import Vision

struct PostureCoreMLPrediction {
    let label: String
    let confidence: Float
    let isPoorPosture: Bool
}

enum PostureCoreMLClassifier {
    private static let badLabels: Set<String> = [
        "inclinado_jorobado",
        "inseguro",
        "unsafe",
        "bad",
        "1"
    ]

    private static let jointOrder: [(VNHumanBodyPoseObservation.JointName, String, String)] = [
        (.nose, "nose_x", "nose_y"),
        (.leftEye, "leftEye_x", "leftEye_y"),
        (.rightEye, "rightEye_x", "rightEye_y"),
        (.leftEar, "leftEar_x", "leftEar_y"),
        (.rightEar, "rightEar_x", "rightEar_y"),
        (.neck, "neck_x", "neck_y"),
        (.leftShoulder, "leftShoulder_x", "leftShoulder_y"),
        (.rightShoulder, "rightShoulder_x", "rightShoulder_y"),
        (.leftElbow, "leftElbow_x", "leftElbow_y"),
        (.rightElbow, "rightElbow_x", "rightElbow_y"),
        (.leftWrist, "leftWrist_x", "leftWrist_y"),
        (.rightWrist, "rightWrist_x", "rightWrist_y"),
        (.leftHip, "leftHip_x", "leftHip_y"),
        (.rightHip, "rightHip_x", "rightHip_y"),
        (.root, "root_x", "root_y"),
        (.leftKnee, "leftKnee_x", "leftKnee_y"),
        (.rightKnee, "rightKnee_x", "rightKnee_y"),
        (.leftAnkle, "leftAnkle_x", "leftAnkle_y"),
        (.rightAnkle, "rightAnkle_x", "rightAnkle_y")
    ]

    private static let loadedModel: MLModel? = {
        let bundle = Bundle.main
        if let compiled = bundle.url(forResource: "TSLPostureModel", withExtension: "mlmodelc"),
           let model = try? MLModel(contentsOf: compiled) {
            return model
        }
        if let raw = bundle.url(forResource: "TSLPostureModel", withExtension: "mlmodel"),
           let compiled = try? MLModel.compileModel(at: raw),
           let model = try? MLModel(contentsOf: compiled) {
            return model
        }
        return nil
    }()

    static var isAvailable: Bool { loadedModel != nil }

    /// Inferencia directa: vector A (19 articulaciones × x,y) → TSLPostureModel.
    static func predict(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> PostureCoreMLPrediction? {
        guard let model = loadedModel else { return nil }
        guard let vector = buildFeatureVector(joints: joints) else { return nil }

        do {
            var features: [String: MLFeatureValue] = [:]
            var idx = 0
            for (_, xName, yName) in jointOrder {
                features[xName] = MLFeatureValue(double: vector[idx].doubleValue)
                idx += 1
                features[yName] = MLFeatureValue(double: vector[idx].doubleValue)
                idx += 1
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: features)
            let out = try model.prediction(from: provider)

            let label = readLabel(from: out) ?? "derecho"
            let confidence = readConfidence(from: out, label: label)
            let poor = isUnsafeLabel(label)

            return PostureCoreMLPrediction(
                label: label,
                confidence: confidence,
                isPoorPosture: poor
            )
        } catch {
            return nil
        }
    }

    private static func buildFeatureVector(
        joints: [VNHumanBodyPoseObservation.JointName: PoseJointSample]
    ) -> MLMultiArray? {
        guard let array = try? MLMultiArray(shape: [38], dataType: .double) else { return nil }
        var idx = 0
        for (joint, _, _) in jointOrder {
            let sample = joints[joint]
            array[idx] = NSNumber(value: sample.map { Double($0.location.x) } ?? 0.5)
            idx += 1
            array[idx] = NSNumber(value: sample.map { Double($0.location.y) } ?? 0.5)
            idx += 1
        }
        return array
    }

    private static func isUnsafeLabel(_ label: String) -> Bool {
        let lower = label.lowercased()
        return badLabels.contains(lower)
            || lower.contains("inclin")
            || lower.contains("jorob")
            || lower.contains("insegur")
    }

    private static func readLabel(from features: MLFeatureProvider) -> String? {
        if let s = features.featureValue(for: "label")?.stringValue { return s }
        if let s = features.featureValue(for: "classLabel")?.stringValue { return s }
        return nil
    }

    private static func readConfidence(from features: MLFeatureProvider, label: String) -> Float {
        let probKeys = ["labelProbability", "classProbability", "classLabelProbs"]
        for key in probKeys {
            if let dict = features.featureValue(for: key)?.dictionaryValue {
                return confidence(from: dict, label: label)
            }
        }
        return 0.55
    }

    private static func confidence(from dict: [AnyHashable: Any], label: String) -> Float {
        for (key, value) in dict {
            let k = key as? String ?? "\(key)"
            if k == label, let num = value as? NSNumber {
                return num.floatValue
            }
        }
        if let best = dict.max(by: {
            ($0.value as? NSNumber)?.floatValue ?? 0 < ($1.value as? NSNumber)?.floatValue ?? 0
        }) {
            return (best.value as? NSNumber)?.floatValue ?? 0.5
        }
        return 0.55
    }
}
