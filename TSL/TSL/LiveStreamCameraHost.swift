//
//  LiveStreamCameraHost.swift
//  TSL — Captura cámara iPad y envío comprimido (US-09 host)
//

import AVFoundation
import CoreVideo
import UIKit

final class LiveStreamCameraHost: NSObject {
    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "tsl.livestream.capture")
    private let output = AVCaptureVideoDataOutput()

    private var multipeer: LiveStreamMultipeerService?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private weak var previewView: UIView?
    private let rotationController = PoseCameraRotationController()

    private var isSending = false
    private var frameSequence: UInt32 = 0
    private var lastSendTime: CFAbsoluteTime = 0
    private var droppedFrames = 0
    private var sentFrames = 0

    var onStatsUpdate: ((Int, Int) -> Void)?

    func attach(to view: UIView, multipeer: LiveStreamMultipeerService) {
        self.previewView = view
        self.multipeer = multipeer

        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = view.bounds
        view.layer.insertSublayer(layer, at: 0)
        previewLayer = layer

        sessionQueue.async { [weak self] in
            self?.configureSession()
        }
    }

    func stop() {
        rotationController.teardown()
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }

    func layoutPreview() {
        previewLayer?.frame = previewView?.bounds ?? .zero
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .vga640x480

        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }

        guard let device = AVCaptureDevice.default(
            .builtInWideAngleCamera,
            for: .video,
            position: .back
        ),
        let input = try? AVCaptureDeviceInput(device: device),
        session.canAddInput(input) else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(output)
        output.setSampleBufferDelegate(self, queue: sessionQueue)

        session.commitConfiguration()
        session.startRunning()

        DispatchQueue.main.async { [weak self] in
            guard let self, let layer = self.previewLayer else { return }
            self.rotationController.configure(device: device, previewLayer: layer) {
                self.sessionQueue.async { self.applyRotation() }
            }
        }
    }

    private func applyRotation() {
        guard let coordinator = rotationController.currentCoordinator(),
              let conn = session.outputs.compactMap({ $0 as? AVCaptureVideoDataOutput }).first?.connection(with: .video)
        else { return }
        PoseCameraRotation.applyRotation(
            coordinator: coordinator,
            previewConnection: previewLayer?.connection,
            videoOutputConnection: conn
        )
    }
}

extension LiveStreamCameraHost: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard !isSending,
              let multipeer,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            droppedFrames += 1
            return
        }

        let minInterval = 1.0 / LiveStreamConfig.maxSendFPS
        let now = CFAbsoluteTimeGetCurrent()
        if now - lastSendTime < minInterval {
            droppedFrames += 1
            return
        }

        isSending = true
        defer { isSending = false }

        guard let compressed = LiveStreamFrameCodec.compressPixelBuffer(pixelBuffer) else {
            droppedFrames += 1
            return
        }

        let (jpeg, width, height) = compressed
        frameSequence &+= 1
        let epochMs = UInt64(Date().timeIntervalSince1970 * 1000)
        let packet = LiveStreamFrameCodec.encodeVideoFrame(
            LiveStreamVideoFrame(
                sequence: frameSequence,
                sentAtEpochMs: epochMs,
                width: UInt16(min(max(width, 0), Int(UInt16.max))),
                height: UInt16(min(max(height, 0), Int(UInt16.max))),
                jpegData: jpeg
            )
        )

        multipeer.sendVideoFrame(packet)
        lastSendTime = now
        sentFrames += 1

        if sentFrames % 30 == 0 {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.onStatsUpdate?(self.sentFrames, self.droppedFrames)
            }
        }
    }
}
