//
//  PoseCameraRotation.swift
//  TSL — Orientación Vision + rotación AVFoundation (solo cámara frontal)
//

import AVFoundation
import CoreGraphics
import UIKit
import Vision

/// Contexto del frame de vídeo para orientación Vision y mapeo al preview (selfie).
struct PoseCameraFrameContext {
    let interfaceOrientation: UIInterfaceOrientation
    let videoRotationAngle: CGFloat
    let isVideoMirrored: Bool
    let pixelBufferWidth: Int
    let pixelBufferHeight: Int
}

enum PoseCameraRotation {
    // MARK: - Vision (VNImageRequestHandler)

    /// Orientación EXIF para el `CVPixelBuffer` que entrega `AVCaptureVideoDataOutput`.
    static func visionImageOrientation(
        interface: UIInterfaceOrientation,
        connection: AVCaptureConnection?,
        pixelBuffer: CVPixelBuffer
    ) -> CGImagePropertyOrientation {
        let context = PoseCameraFrameContext(
            interfaceOrientation: interface,
            videoRotationAngle: connection?.videoRotationAngle ?? 0,
            isVideoMirrored: connection?.isVideoMirrored ?? false,
            pixelBufferWidth: CVPixelBufferGetWidth(pixelBuffer),
            pixelBufferHeight: CVPixelBufferGetHeight(pixelBuffer)
        )
        return visionImageOrientation(context: context)
    }

    static func visionImageOrientation(context: PoseCameraFrameContext) -> CGImagePropertyOrientation {
        if #available(iOS 17.0, *), isHorizonLevelCaptureBuffer(context) {
            return context.isVideoMirrored ? .upMirrored : .up
        }
        return legacyFrontCameraVisionOrientation(interface: context.interfaceOrientation)
    }

    /// Buffer ya rotado por `connection.videoRotationAngle` (RotationCoordinator).
    @available(iOS 17.0, *)
    private static func isHorizonLevelCaptureBuffer(_ context: PoseCameraFrameContext) -> Bool {
        context.videoRotationAngle != 0
    }

    /// Cámara frontal sin rotación en la conexión (pre‑iOS 17 o sin coordinator).
    private static func legacyFrontCameraVisionOrientation(
        interface: UIInterfaceOrientation
    ) -> CGImagePropertyOrientation {
        switch interface {
        case .portrait:
            return .leftMirrored
        case .portraitUpsideDown:
            return .rightMirrored
        case .landscapeLeft:
            return .downMirrored
        case .landscapeRight:
            return .upMirrored
        default:
            return .leftMirrored
        }
    }

    // MARK: - Vision → AVCaptureVideoPreviewLayer (overlay / esqueleto)

    /// Convierte un punto normalizado de Vision (origen abajo-izquierda) a coordenadas del `previewLayer`.
    static func mapVisionPointToPreview(
        _ normalized: CGPoint,
        previewLayer: AVCaptureVideoPreviewLayer,
        isVideoMirrored: Bool
    ) -> CGPoint {
        var nx = normalized.x
        var ny = 1.0 - normalized.y

        if isVideoMirrored {
            nx = 1.0 - nx
        }

        let metadataRect = CGRect(x: nx, y: ny, width: 0.001, height: 0.001)
        let layerRect = previewLayer.layerRectConverted(fromMetadataOutputRect: metadataRect)
        let h = previewLayer.bounds.height
        guard h > 0 else {
            return CGPoint(x: layerRect.midX, y: layerRect.midY)
        }
        return CGPoint(x: layerRect.midX, y: h - layerRect.midY)
    }

    // MARK: - AVCaptureConnection (RotationCoordinator)

    static func applyRotation(
        coordinator: AVCaptureDevice.RotationCoordinator,
        previewConnection: AVCaptureConnection?,
        videoOutputConnection: AVCaptureConnection?
    ) {
        let previewAngle = coordinator.videoRotationAngleForHorizonLevelPreview
        let captureAngle = coordinator.videoRotationAngleForHorizonLevelCapture

        if let previewConnection,
           previewConnection.isVideoRotationAngleSupported(previewAngle) {
            previewConnection.videoRotationAngle = previewAngle
        }

        if let videoOutputConnection,
           videoOutputConnection.isVideoRotationAngleSupported(captureAngle) {
            videoOutputConnection.videoRotationAngle = captureAngle
        }
    }

    static func configureFrontVideoConnection(_ connection: AVCaptureConnection?) {
        guard let connection else { return }
        if connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = true
        }
    }
}

/// Gestiona `AVCaptureDevice.RotationCoordinator` y observa cambios de ángulo.
final class PoseCameraRotationController {
    private var coordinator: AVCaptureDevice.RotationCoordinator?
    private var observation: NSKeyValueObservation?

    func configure(device: AVCaptureDevice, previewLayer: AVCaptureVideoPreviewLayer, onChange: @escaping () -> Void) {
        teardown()
        let newCoordinator = AVCaptureDevice.RotationCoordinator(
            device: device,
            previewLayer: previewLayer
        )
        coordinator = newCoordinator
        observation = newCoordinator.observe(
            \.videoRotationAngleForHorizonLevelPreview,
            options: [.new]
        ) { _, _ in
            onChange()
        }
        onChange()
    }

    func currentCoordinator() -> AVCaptureDevice.RotationCoordinator? {
        coordinator
    }

    func teardown() {
        observation?.invalidate()
        observation = nil
        coordinator = nil
    }
}
