//
//  PoseKNNClassifier.swift
//  TSL — US/13 inferencia KNN en producción (iOS)
//

import Foundation

struct PoseKNNProductionModel: Codable {
    let version: Int
    let k: Int
    let metric: String
    let classLabels: [String]
    let usePca: Bool
    let nFeatures: Int
    let trainingVectors: [[Double]]
    let trainingLabels: [Int]
    let testAccuracy: Double
    let testF1Score: Double

    enum CodingKeys: String, CodingKey {
        case version, k, metric
        case classLabels = "class_labels"
        case usePca = "use_pca"
        case nFeatures = "n_features"
        case trainingVectors = "training_vectors"
        case trainingLabels = "training_labels"
        case testAccuracy = "test_accuracy"
        case testF1Score = "test_f1_score"
    }
}

struct PoseKNNPrediction {
    /// Índice de clase entrenada (orden en `class_labels` del JSON).
    let labelIndex: Int
    let label: String
    let confidence: Double

    /// US-14: salida discreta binaria — 0 = Seguro, 1 = Inseguro.
    var discreteClass: Int {
        if labelIndex == 0 || labelIndex == 1 {
            return labelIndex
        }
        if label == "inclinado_jorobado" || label.lowercased().contains("insegur") {
            return DiscretePostureClass.inseguro.rawValue
        }
        return DiscretePostureClass.seguro.rawValue
    }
}

/// Clasificador KNN exportado desde `train_knn.py`.
final class PoseKNNClassifier {
    private let model: PoseKNNProductionModel

    init(bundle: Bundle = .main, filename: String = "knn_production") throws {
        guard let url = bundle.url(forResource: filename, withExtension: "json") else {
            throw PoseKNNError.modelNotFound(filename)
        }
        let data = try Data(contentsOf: url)
        model = try JSONDecoder().decode(PoseKNNProductionModel.self, from: data)
    }

    var bestK: Int { model.k }
    var classLabels: [String] { model.classLabels }
    var usesPCA: Bool { model.usePca }

    func predict(features: [Double]) throws -> PoseKNNPrediction {
        guard features.count == model.nFeatures else {
            throw PoseKNNError.invalidFeatureCount(expected: model.nFeatures, got: features.count)
        }
        guard !features.contains(where: { $0.isNaN }) else {
            throw PoseKNNError.nanInFeatures
        }

        let trainX = model.trainingVectors
        let trainY = model.trainingLabels
        guard !trainX.isEmpty else {
            throw PoseKNNError.emptyTrainingSet
        }

        var distances: [(dist: Double, label: Int)] = []
        distances.reserveCapacity(trainX.count)

        for (i, vec) in trainX.enumerated() {
            var sum = 0.0
            for j in 0..<model.nFeatures {
                let d = features[j] - vec[j]
                sum += d * d
            }
            distances.append((sqrt(sum), trainY[i]))
        }

        distances.sort { $0.dist < $1.dist }
        let k = min(model.k, distances.count)
        let neighbors = distances.prefix(k)

        var votes: [Int: Int] = [:]
        for n in neighbors {
            votes[n.label, default: 0] += 1
        }

        guard let best = votes.max(by: { $0.value < $1.value }) else {
            throw PoseKNNError.predictionFailed
        }

        let confidence = Double(best.value) / Double(k)
        let label = model.classLabels.indices.contains(best.key)
            ? model.classLabels[best.key]
            : "unknown"

        return PoseKNNPrediction(
            labelIndex: best.key,
            label: label,
            confidence: confidence
        )
    }
}

enum PoseKNNError: LocalizedError {
    case modelNotFound(String)
    case invalidFeatureCount(expected: Int, got: Int)
    case nanInFeatures
    case emptyTrainingSet
    case predictionFailed

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "No se encontró \(name).json. Ejecuta train_knn.py."
        case .invalidFeatureCount(let expected, let got):
            return "KNN espera \(expected) features, recibió \(got)."
        case .nanInFeatures:
            return "KNN no acepta NaN."
        case .emptyTrainingSet:
            return "Conjunto de entrenamiento vacío en el modelo."
        case .predictionFailed:
            return "No se pudo clasificar con KNN."
        }
    }
}
