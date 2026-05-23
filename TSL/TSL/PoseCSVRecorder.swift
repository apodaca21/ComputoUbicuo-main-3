//
//  PoseCSVRecorder.swift
//  TSL
//
//  Graba articulaciones por frame en CSV (formato US/11 pipeline).
//

import Foundation
import Vision

/// Nombres de articulación en snake_case para el CSV del pipeline Python.
enum PoseJointCSVName: String {
    case nose, leftEye = "left_eye", rightEye = "right_eye"
    case leftEar = "left_ear", rightEar = "right_ear"
    case neck
    case leftShoulder = "left_shoulder", rightShoulder = "right_shoulder"
    case leftElbow = "left_elbow", rightElbow = "right_elbow"
    case leftWrist = "left_wrist", rightWrist = "right_wrist"
    case leftHip = "left_hip", rightHip = "right_hip"
    case root
    case leftKnee = "left_knee", rightKnee = "right_knee"
    case leftAnkle = "left_ankle", rightAnkle = "right_ankle"

    init?(visionJoint: VNHumanBodyPoseObservation.JointName) {
        switch visionJoint {
        case .nose: self = .nose
        case .leftEye: self = .leftEye
        case .rightEye: self = .rightEye
        case .leftEar: self = .leftEar
        case .rightEar: self = .rightEar
        case .neck: self = .neck
        case .leftShoulder: self = .leftShoulder
        case .rightShoulder: self = .rightShoulder
        case .leftElbow: self = .leftElbow
        case .rightElbow: self = .rightElbow
        case .leftWrist: self = .leftWrist
        case .rightWrist: self = .rightWrist
        case .leftHip: self = .leftHip
        case .rightHip: self = .rightHip
        case .root: self = .root
        case .leftKnee: self = .leftKnee
        case .rightKnee: self = .rightKnee
        case .leftAnkle: self = .leftAnkle
        case .rightAnkle: self = .rightAnkle
        default: return nil
        }
    }
}

final class PoseCSVRecorder {
    private var buffer = "timestamp,frame_id,track_id,joint,x,y,confidence\n"
    private var frameIndex = 0
    private let minConfidence: Float = 0.15

    func appendFrame(
        timestamp: Double,
        tracks: [[VNHumanBodyPoseObservation.JointName: PoseJointSample]]
    ) {
        for (trackIdx, joints) in tracks.enumerated() {
            let trackId = trackIdx + 1
            for (joint, sample) in joints where sample.confidence >= minConfidence {
                guard let name = PoseJointCSVName(visionJoint: joint) else { continue }
                buffer += String(
                    format: "%.6f,%d,%d,%@,%.6f,%.6f,%.4f\n",
                    timestamp,
                    frameIndex,
                    trackId,
                    name.rawValue,
                    sample.location.x,
                    sample.location.y,
                    sample.confidence
                )
            }
        }
        frameIndex += 1
    }

    func exportToDocuments(filename: String = "pose_recording.csv") -> URL? {
        guard frameIndex > 0,
              let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        else { return nil }
        let url = docs.appendingPathComponent(filename)
        do {
            try buffer.write(to: url, atomically: true, encoding: .utf8)
            return url
        } catch {
            return nil
        }
    }

    func reset() {
        buffer = "timestamp,frame_id,track_id,joint,x,y,confidence\n"
        frameIndex = 0
    }
}
