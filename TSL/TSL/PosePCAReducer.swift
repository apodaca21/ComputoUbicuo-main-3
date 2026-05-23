//
//  PosePCAReducer.swift
//  TSL — US/12 inferencia PCA en producción (Swift / CoreML-ready JSON)
//

import Foundation

/// Modelo PCA exportado desde `pipeline/train_pca.py` → `pca_production.json`.
struct PosePCAProductionModel: Codable {
    let version: Int
    let featureNames: [String]
    let nFeaturesIn: Int
    let nComponents: Int
    let mean: [Double]
    let components: [[Double]]
    let explainedVarianceRatio: [Double]
    let totalExplainedVariance: Double

    enum CodingKeys: String, CodingKey {
        case version
        case featureNames = "feature_names"
        case nFeaturesIn = "n_features_in"
        case nComponents = "n_components"
        case mean
        case components
        case explainedVarianceRatio = "explained_variance_ratio"
        case totalExplainedVariance = "total_explained_variance"
    }
}

/// Reduce el vector de 11 distancias normalizadas a `n_components` (producción).
final class PosePCAReducer {
    private let model: PosePCAProductionModel

    init(bundle: Bundle = .main, filename: String = "pca_production") throws {
        guard let url = bundle.url(forResource: filename, withExtension: "json") else {
            throw PosePCAError.modelNotFound(filename)
        }
        let data = try Data(contentsOf: url)
        model = try JSONDecoder().decode(PosePCAProductionModel.self, from: data)
    }

    init(model: PosePCAProductionModel) {
        self.model = model
    }

    var nComponents: Int { model.nComponents }
    var totalExplainedVariance: Double { model.totalExplainedVariance }

    /// `features` debe tener 11 valores en el mismo orden que `feature_names` del JSON.
    func transform(features: [Double]) throws -> [Double] {
        guard features.count == model.nFeaturesIn else {
            throw PosePCAError.invalidFeatureCount(expected: model.nFeaturesIn, got: features.count)
        }
        guard !features.contains(where: { $0.isNaN }) else {
            throw PosePCAError.nanInFeatures
        }

        var projected = [Double](repeating: 0, count: model.nComponents)
        for (c, component) in model.components.enumerated() {
            var dot = 0.0
            for i in 0..<model.nFeaturesIn {
                dot += (features[i] - model.mean[i]) * component[i]
            }
            projected[c] = dot
        }
        return projected
    }
}

enum PosePCAError: LocalizedError {
    case modelNotFound(String)
    case invalidFeatureCount(expected: Int, got: Int)
    case nanInFeatures

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "No se encontró \(name).json en el bundle. Ejecuta train_pca.py y copia el modelo."
        case .invalidFeatureCount(let expected, let got):
            return "PCA espera \(expected) features, recibió \(got)."
        case .nanInFeatures:
            return "PCA no acepta NaN en las distancias."
        }
    }
}
