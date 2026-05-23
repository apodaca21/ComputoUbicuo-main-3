//
//  PoseMinMaxScaler.swift
//  TSL — Misma escala Min-Max que US/11 (requerido antes de PCA/KNN)
//

import Foundation

struct PoseMinMaxScalerModel: Codable {
    let featureNames: [String]
    let mins: [Double]
    let maxs: [Double]
}

final class PoseMinMaxScaler {
    private let model: PoseMinMaxScalerModel
    private let epsilon = 1e-8

    init(bundle: Bundle = .main, filename: String = "minmax_scaler") throws {
        guard let url = bundle.url(forResource: filename, withExtension: "json") else {
            throw PoseMinMaxError.modelNotFound(filename)
        }
        model = try JSONDecoder().decode(PoseMinMaxScalerModel.self, from: Data(contentsOf: url))
    }

    func transform(_ features: [Double]) throws -> [Double] {
        guard features.count == model.mins.count else {
            throw PoseMinMaxError.invalidCount(expected: model.mins.count, got: features.count)
        }
        return zip(zip(features, model.mins), model.maxs).map { pair, maxVal in
            let (value, minVal) = pair
            let span = max(maxVal - minVal, epsilon)
            return min(1, max(0, (value - minVal) / span))
        }
    }
}

enum PoseMinMaxError: LocalizedError {
    case modelNotFound(String)
    case invalidCount(expected: Int, got: Int)

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Falta \(name).json en el bundle."
        case .invalidCount(let expected, let got):
            return "MinMax espera \(expected) valores, recibió \(got)."
        }
    }
}
