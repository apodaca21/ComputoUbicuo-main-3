//
//  PoseCameraView.swift
//  TSL
//
//  Cámara en vivo + pose corporal con Vision (Kalman MOT + mapeo original de coordenadas).
//

import AVFoundation
import Combine
import SwiftUI
import UIKit
import Vision

// MARK: - Estado de postura y puente de logs

enum PostureState: Equatable {
    case none
    case good
    case bad
}

final class PoseCameraBridge: ObservableObject {
    @Published var postureState: PostureState = .none
    @Published var logs: [String] = []

    private let maxLogs = 300

    func appendLog(_ line: String) {
        let ts = ISO8601DateFormatter().string(from: Date())
        let entry = "[\(ts)] \(line)"
        logs.insert(entry, at: 0)
        if logs.count > maxLogs {
            logs.removeLast()
        }
    }

    func setPosture(_ state: PostureState) {
        postureState = state
    }
}

// MARK: - SwiftUI

struct PoseCameraView: View {
    @Binding var isPresented: Bool
    @StateObject private var bridge = PoseCameraBridge()
    @State private var showLogPanel = false

    var body: some View {
        ZStack(alignment: .topLeading) {
            PoseCameraRepresentable(bridge: bridge, isPresented: $isPresented)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 14) {
                    Button {
                        isPresented = false
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 32))
                            .symbolRenderingMode(.hierarchical)
                            .foregroundStyle(.white)
                            .shadow(radius: 4)
                    }

                    Button {
                        showLogPanel = true
                    } label: {
                        Image(systemName: "bell.fill")
                            .font(.system(size: 26))
                            .foregroundStyle(.white)
                            .padding(10)
                            .background(.black.opacity(0.45))
                            .clipShape(Circle())
                            .shadow(radius: 4)
                    }
                    .accessibilityLabel("Registro de postura")
                }
                .padding(.leading, 20)
                .padding(.top, 16)

                Spacer()
            }

            VStack {
                Spacer()
                Text("Verde = buena postura — rojo/amarillo = zona incorrecta — campana: registro")
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 10)
                    .background(.black.opacity(0.45))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.bottom, 36)
            }
        }
        .sheet(isPresented: $showLogPanel) {
            NavigationStack {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(Array(bridge.logs.enumerated()), id: \.offset) { _, line in
                            Text(line)
                                .font(.caption.monospaced())
                                .foregroundStyle(.primary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding()
                }
                .navigationTitle("Registro de postura")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("Cerrar") { showLogPanel = false }
                    }
                }
            }
        }
        .onReceive(Timer.publish(every: 5, on: .main, in: .common).autoconnect()) { _ in
            guard bridge.postureState == .good else { return }
            bridge.appendLog("Postura correcta")
        }
    }
}

// MARK: - UIViewControllerRepresentable

private struct PoseCameraRepresentable: UIViewControllerRepresentable {
    @ObservedObject var bridge: PoseCameraBridge
    @Binding var isPresented: Bool

    func makeUIViewController(context: Context) -> PoseCameraViewController {
        let vc = PoseCameraViewController()
        vc.bridge = bridge
        vc.onDismissRequest = { isPresented = false }
        return vc
    }

    func updateUIViewController(_ uiViewController: PoseCameraViewController, context: Context) {}
}

// MARK: - Conexiones del esqueleto

private let bodyJointConnections: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName)] = [
    (.neck, .nose),
    (.leftShoulder, .neck),
    (.rightShoulder, .neck),
    (.leftShoulder, .rightShoulder),
    (.leftShoulder, .leftElbow),
    (.leftElbow, .leftWrist),
    (.rightShoulder, .rightElbow),
    (.rightElbow, .rightWrist),
    (.leftShoulder, .leftHip),
    (.rightShoulder, .rightHip),
    (.leftHip, .rightHip),
    (.leftHip, .leftKnee),
    (.leftKnee, .leftAnkle),
    (.rightHip, .rightKnee),
    (.rightKnee, .rightAnkle)
]

// MARK: - Vista de cámara + Vision

private final class PoseCameraViewController: UIViewController {
    var onDismissRequest: (() -> Void)?
    weak var bridge: PoseCameraBridge?

    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "tsl.camera.session")
    private let visionQueue = DispatchQueue(label: "tsl.vision.pose")

    private var previewLayer: AVCaptureVideoPreviewLayer!
    private let overlayView = PoseOverlayView()
    private let trafficLightView = TrafficLightIndicatorView()
    private let poseRequest = VNDetectHumanBodyPoseRequest()
    private let poseTracker = MultiBodyPoseKalmanTracker()

    private var isProcessingFrame = false
    private var captureDevice: AVCaptureDevice?
    private weak var videoDataConnection: AVCaptureConnection?
    private let rotationController = PoseCameraRotationController()
    private var lastInterfaceOrientation: UIInterfaceOrientation = .portrait
    private var lastLoggedBadTransition = false

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black

        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        overlayView.backgroundColor = .clear
        overlayView.isUserInteractionEnabled = false
        view.addSubview(overlayView)

        trafficLightView.translatesAutoresizingMaskIntoConstraints = false
        trafficLightView.isUserInteractionEnabled = false
        trafficLightView.setState(.noPerson)
        view.addSubview(trafficLightView)
        NSLayoutConstraint.activate([
            trafficLightView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 6),
            trafficLightView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            trafficLightView.widthAnchor.constraint(equalToConstant: 56),
            trafficLightView.heightAnchor.constraint(equalToConstant: 92)
        ])

        NotificationCenter.default.addObserver(
            self,
            selector: #selector(orientationChanged),
            name: UIDevice.orientationDidChangeNotification,
            object: nil
        )
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
        overlayView.frame = view.bounds
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        lastLoggedBadTransition = false
        bridge?.appendLog(PoseMLPostureEngine.shared.statusMessage)
        checkPermissionAndConfigure()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }

    deinit {
        rotationController.teardown()
        NotificationCenter.default.removeObserver(self)
    }

    @objc private func orientationChanged() {
        applyRotationFromCoordinator()
    }

    private func currentInterfaceOrientation() -> UIInterfaceOrientation {
        if let scene = view.window?.windowScene {
            return scene.interfaceOrientation
        }
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        if let active = scenes.first(where: { $0.activationState == .foregroundActive }) {
            return active.interfaceOrientation
        }
        if let first = scenes.first {
            return first.interfaceOrientation
        }
        return .portrait
    }

    private func applyRotationFromCoordinator() {
        lastInterfaceOrientation = currentInterfaceOrientation()
        guard let rotationCoordinator = rotationController.currentCoordinator() else { return }
        PoseCameraRotation.applyRotation(
            coordinator: rotationCoordinator,
            previewConnection: previewLayer.connection,
            videoOutputConnection: nil
        )
        sessionQueue.async { [weak self] in
            self?.applyVideoOutputRotation()
        }
    }

    private func applyVideoOutputRotation() {
        guard let rotationCoordinator = rotationController.currentCoordinator() else { return }
        if videoDataConnection == nil {
            videoDataConnection = session.outputs
                .compactMap { $0 as? AVCaptureVideoDataOutput }
                .first?
                .connection(with: .video)
        }
        PoseCameraRotation.applyRotation(
            coordinator: rotationCoordinator,
            previewConnection: nil,
            videoOutputConnection: videoDataConnection
        )
        PoseCameraRotation.configureFrontVideoConnection(videoDataConnection)
    }

    private func setupRotationCoordinator(with device: AVCaptureDevice) {
        captureDevice = device
        rotationController.configure(device: device, previewLayer: previewLayer) { [weak self] in
            self?.applyRotationFromCoordinator()
        }
    }

    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        coordinator.animate(alongsideTransition: { _ in
            self.applyRotationFromCoordinator()
        })
    }

    private func checkPermissionAndConfigure() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            sessionQueue.async { [weak self] in self?.configureSession() }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.sessionQueue.async { self?.configureSession() }
                    } else {
                        self?.onDismissRequest?()
                    }
                }
            }
        default:
            onDismissRequest?()
        }
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .high

        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else {
            session.commitConfiguration()
            DispatchQueue.main.async { [weak self] in self?.onDismissRequest?() }
            return
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]

        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(output)

        let videoConn = output.connection(with: .video)
        videoDataConnection = videoConn
        PoseCameraRotation.configureFrontVideoConnection(videoConn)

        session.commitConfiguration()
        output.setSampleBufferDelegate(self, queue: visionQueue)

        session.startRunning()

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.setupRotationCoordinator(with: device)
            self.sessionQueue.async {
                self.applyVideoOutputRotation()
            }
        }
    }

    private func convertVisionPoint(
        _ normalized: CGPoint,
        previewLayer: AVCaptureVideoPreviewLayer,
        isVideoMirrored: Bool
    ) -> CGPoint {
        PoseCameraRotation.mapVisionPointToPreview(
            normalized,
            previewLayer: previewLayer,
            isVideoMirrored: isVideoMirrored
        )
    }
}

// MARK: - Video frames → Vision

extension PoseCameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard !isProcessingFrame else { return }
        isProcessingFrame = true

        guard CMSampleBufferGetImageBuffer(sampleBuffer) != nil else {
            isProcessingFrame = false
            DispatchQueue.main.async { [weak self] in
                self?.trafficLightView.setState(.noPerson)
                self?.bridge?.setPosture(.none)
            }
            return
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            isProcessingFrame = false
            return
        }

        let orientation = PoseCameraRotation.visionImageOrientation(
            interface: lastInterfaceOrientation,
            connection: connection,
            pixelBuffer: pixelBuffer
        )

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])

        do {
            try handler.perform([poseRequest])
        } catch {
            isProcessingFrame = false
            DispatchQueue.main.async { [weak self] in
                self?.trafficLightView.setState(.noPerson)
                self?.bridge?.setPosture(.none)
            }
            return
        }

        let rawObservations = (poseRequest.results ?? []).compactMap { $0 as? VNHumanBodyPoseObservation }
        let smoothedTracks = poseTracker.update(observations: rawObservations, sampleBuffer: sampleBuffer)

        let primaryIdx = primaryTrackIndex(smoothedTracks)
        let primaryJoints = smoothedTracks.indices.contains(primaryIdx) ? smoothedTracks[primaryIdx] : [:]

        let classification = PosePipelineClassifier.classify(tracks: smoothedTracks, primaryIndex: primaryIdx)
        let isPoorPosture = classification.discreteClass == DiscretePostureClass.inseguro.rawValue
        let problemJoints = classification.problemJoints
        let isVideoMirrored = connection.isVideoMirrored
        let minLineConf = Float(0.22)

        var primarySegments: [(VNHumanBodyPoseObservation.JointName, VNHumanBodyPoseObservation.JointName, CGPoint, CGPoint)] = []
        if !primaryJoints.isEmpty {
            for (a, b) in bodyJointConnections {
                guard let pa = primaryJoints[a], let pb = primaryJoints[b],
                      pa.confidence > minLineConf,
                      pb.confidence > minLineConf else { continue }
                primarySegments.append((a, b, pa.location, pb.location))
            }
        }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            defer {
                self.visionQueue.async { self.isProcessingFrame = false }
            }

            guard let previewLayer = self.previewLayer else { return }

            if smoothedTracks.isEmpty || primaryJoints.isEmpty {
                self.overlayView.clear()
                self.trafficLightView.setState(.noPerson)
                self.bridge?.setPosture(.none)
                return
            }

            if classification.discreteClass == PostureClassificationResult.noPersonDiscrete {
                self.trafficLightView.setState(.noPerson)
                self.bridge?.setPosture(.none)
            } else if isPoorPosture {
                self.trafficLightView.setState(.badPosture)
                self.bridge?.setPosture(.bad)
                if !self.lastLoggedBadTransition {
                    self.lastLoggedBadTransition = true
                    self.bridge?.appendLog(
                        "Inseguro conf=\(String(format: "%.2f", classification.confidence)) ml=\(classification.usedMLModel)"
                    )
                }
            } else {
                self.trafficLightView.setState(.goodPosture)
                self.bridge?.setPosture(.good)
                self.lastLoggedBadTransition = false
            }

            var lineSegments: [(CGPoint, CGPoint)] = []
            var jointDraws: [PoseJointDraw] = []
            for (ja, jb, la, lb) in primarySegments {
                let p1 = self.convertVisionPoint(la, previewLayer: previewLayer, isVideoMirrored: isVideoMirrored)
                let p2 = self.convertVisionPoint(lb, previewLayer: previewLayer, isVideoMirrored: isVideoMirrored)
                lineSegments.append((p1, p2))
                jointDraws.append(PoseJointDraw(point: p1, joint: ja, isProblem: problemJoints.contains(ja)))
                jointDraws.append(PoseJointDraw(point: p2, joint: jb, isProblem: problemJoints.contains(jb)))
            }

            if lineSegments.isEmpty {
                self.overlayView.clear()
                return
            }
            self.overlayView.update(
                lines: lineSegments,
                joints: jointDraws,
                isPoorPosture: isPoorPosture
            )
        }
    }
}

// MARK: - Semáforo verde / rojo

private enum TrafficLightState {
    case noPerson
    case goodPosture
    case badPosture
}

private final class TrafficLightIndicatorView: UIView {
    private let housing = UIView()
    private let lampView = UIView()

    override init(frame: CGRect) {
        super.init(frame: frame)
        isAccessibilityElement = true
        accessibilityTraits.insert(.updatesFrequently)
        accessibilityLabel = "Estado de postura"

        housing.translatesAutoresizingMaskIntoConstraints = false
        housing.backgroundColor = UIColor(white: 0.12, alpha: 0.92)
        housing.layer.cornerRadius = 14
        housing.layer.borderWidth = 1.5
        housing.layer.borderColor = UIColor.white.withAlphaComponent(0.35).cgColor

        lampView.translatesAutoresizingMaskIntoConstraints = false
        lampView.layer.cornerRadius = 20
        lampView.backgroundColor = .systemRed
        lampView.layer.shadowOpacity = 0.85
        lampView.layer.shadowRadius = 10
        lampView.layer.shadowOffset = .zero
        lampView.layer.masksToBounds = false

        addSubview(housing)
        housing.addSubview(lampView)

        NSLayoutConstraint.activate([
            housing.topAnchor.constraint(equalTo: topAnchor),
            housing.leadingAnchor.constraint(equalTo: leadingAnchor),
            housing.trailingAnchor.constraint(equalTo: trailingAnchor),
            housing.bottomAnchor.constraint(equalTo: bottomAnchor),
            lampView.centerXAnchor.constraint(equalTo: housing.centerXAnchor),
            lampView.centerYAnchor.constraint(equalTo: housing.centerYAnchor),
            lampView.widthAnchor.constraint(equalToConstant: 40),
            lampView.heightAnchor.constraint(equalToConstant: 40)
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func setState(_ state: TrafficLightState) {
        let color: UIColor
        let value: String
        switch state {
        case .noPerson:
            color = UIColor(white: 0.28, alpha: 1)
            value = "Sin persona"
        case .goodPosture:
            color = .systemGreen
            value = "Postura correcta (derecho)"
        case .badPosture:
            color = .systemRed
            value = "Postura incorrecta (inclinado o jorobado)"
        }
        accessibilityValue = value

        UIView.animate(withDuration: 0.18, delay: 0, options: [.curveEaseInOut, .beginFromCurrentState]) {
            self.lampView.backgroundColor = color
            self.lampView.layer.shadowColor = color.cgColor
        }
    }
}

// MARK: - Capa de dibujo del esqueleto (verde / rojo + amarillo en zona mala)

private struct PoseJointDraw {
    let point: CGPoint
    let joint: VNHumanBodyPoseObservation.JointName
    let isProblem: Bool
}

private final class PoseOverlayView: UIView {
    private var lineSegments: [(CGPoint, CGPoint)] = []
    private var joints: [PoseJointDraw] = []
    private var isPoorPosture = false

    func update(lines: [(CGPoint, CGPoint)], joints: [PoseJointDraw], isPoorPosture: Bool) {
        self.lineSegments = lines
        self.joints = joints
        self.isPoorPosture = isPoorPosture
        setNeedsDisplay()
    }

    func clear() {
        lineSegments = []
        joints = []
        isPoorPosture = false
        setNeedsDisplay()
    }

    override func draw(_ rect: CGRect) {
        guard let ctx = UIGraphicsGetCurrentContext() else { return }

        let lineColor: UIColor = isPoorPosture
            ? UIColor(red: 1, green: 0.22, blue: 0.2, alpha: 0.95)
            : UIColor(red: 0.4, green: 1.0, blue: 0.55, alpha: 0.95)

        ctx.setLineWidth(4)
        ctx.setStrokeColor(lineColor.cgColor)
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)

        for (p1, p2) in lineSegments {
            ctx.move(to: p1)
            ctx.addLine(to: p2)
            ctx.strokePath()
        }

        let r: CGFloat = 7
        for joint in joints {
            let fill: UIColor
            if isPoorPosture && joint.isProblem {
                fill = UIColor(red: 1, green: 0.92, blue: 0.1, alpha: 1)
            } else if isPoorPosture {
                fill = UIColor(red: 1, green: 0.45, blue: 0.42, alpha: 0.95)
            } else {
                fill = UIColor(red: 0.9, green: 1.0, blue: 0.95, alpha: 0.95)
            }
            ctx.setFillColor(fill.cgColor)
            ctx.fillEllipse(in: CGRect(x: joint.point.x - r, y: joint.point.y - r, width: r * 2, height: r * 2))
            if isPoorPosture && joint.isProblem {
                ctx.setStrokeColor(UIColor.systemOrange.cgColor)
                ctx.setLineWidth(2)
                ctx.strokeEllipse(in: CGRect(x: joint.point.x - r - 2, y: joint.point.y - r - 2, width: (r + 2) * 2, height: (r + 2) * 2))
            }
        }
    }
}
