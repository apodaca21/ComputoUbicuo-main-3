//
//  LiveStreamMultipeerService.swift
//  TSL — Multipeer Connectivity (US-09 streaming iPad → iPhone)
//

import Combine
import Foundation
import MultipeerConnectivity
import UIKit

enum LiveStreamRole {
    case host   // iPad: publica cámara
    case viewer // iPhone: recibe y muestra
}

@MainActor
final class LiveStreamSessionModel: ObservableObject {
    @Published var role: LiveStreamRole
    @Published var connectionState: String = "Desconectado"
    @Published var connectedPeerName: String?
    @Published var latestFrame: UIImage?
    @Published var isHosting = false
    @Published var isBrowsing = false
    @Published var lastError: String?

    let latency = LiveStreamLatencyTracker()

    init(role: LiveStreamRole) {
        self.role = role
    }
}

// MARK: - Multipeer (callbacks fuera del MainActor → re-dispatch)

final class LiveStreamMultipeerService: NSObject {
    let model: LiveStreamSessionModel
    private let peerID: MCPeerID
    private var session: MCSession?
    private var advertiser: MCNearbyServiceAdvertiser?
    private var browser: MCNearbyServiceBrowser?
    private var pingTimer: Timer?
    private var pingSequence: UInt32 = 0

    var onVideoFrameForHost: ((Data) -> Void)?

    init(model: LiveStreamSessionModel) {
        self.model = model
        self.peerID = MCPeerID(displayName: UIDevice.current.name)
        super.init()
    }

    func start() {
        let session = MCSession(peer: peerID, securityIdentity: nil, encryptionPreference: .none)
        session.delegate = self
        self.session = session

        switch model.role {
        case .host:
            startAdvertising(session: session)
        case .viewer:
            startBrowsing(session: session)
        }
    }

    func stop() {
        pingTimer?.invalidate()
        pingTimer = nil
        advertiser?.stopAdvertisingPeer()
        browser?.stopBrowsingForPeers()
        session?.disconnect()
        advertiser = nil
        browser = nil
        session = nil
        reconnectWorkItem?.cancel()
        reconnectWorkItem = nil
        Task { @MainActor in
            model.isHosting = false
            model.isBrowsing = false
            model.connectionState = "Desconectado"
            model.connectedPeerName = nil
        }
    }

    private var reconnectWorkItem: DispatchWorkItem?

    func sendVideoFrame(_ packet: Data) {
        guard let session, !session.connectedPeers.isEmpty else { return }
        do {
            try session.send(packet, toPeers: session.connectedPeers, with: .unreliable)
        } catch {
            Task { @MainActor in model.lastError = error.localizedDescription }
        }
    }

    private func startAdvertising(session: MCSession) {
        let advertiser = MCNearbyServiceAdvertiser(
            peer: peerID,
            discoveryInfo: ["role": "host"],
            serviceType: LiveStreamConfig.serviceType
        )
        advertiser.delegate = self
        advertiser.startAdvertisingPeer()
        self.advertiser = advertiser
        Task { @MainActor in
            model.isHosting = true
            model.connectionState = "Esperando iPhone…"
        }
    }

    private func startBrowsing(session: MCSession) {
        let browser = MCNearbyServiceBrowser(peer: peerID, serviceType: LiveStreamConfig.serviceType)
        browser.delegate = self
        browser.startBrowsingForPeers()
        self.browser = browser
        Task { @MainActor in
            model.isBrowsing = true
            model.connectionState = "Buscando iPad…"
        }
    }

    private func connectedPeers() -> [MCPeerID] {
        session?.connectedPeers ?? []
    }

    private func updateConnectionState() {
        Task { @MainActor in
            guard let session else { return }
            if session.connectedPeers.isEmpty {
                model.connectionState = model.role == .host ? "Esperando iPhone…" : "Buscando iPad…"
                model.connectedPeerName = nil
                pingTimer?.invalidate()
                pingTimer = nil
                scheduleReconnectIfNeeded()
            } else {
                model.connectionState = "Conectado"
                model.connectedPeerName = session.connectedPeers.first?.displayName
                if model.role == .viewer {
                    startPingTimerOnMain()
                }
            }
        }
    }

    private func scheduleReconnectIfNeeded() {
        reconnectWorkItem?.cancel()
        Task { @MainActor [weak self] in
            guard let self else { return }
            let role = self.model.role
            let work = DispatchWorkItem { [weak self] in
                guard let self else { return }
                switch role {
                case .viewer:
                    self.browser?.stopBrowsingForPeers()
                    self.browser?.startBrowsingForPeers()
                case .host:
                    self.advertiser?.stopAdvertisingPeer()
                    self.advertiser?.startAdvertisingPeer()
                }
            }
            self.reconnectWorkItem = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.5, execute: work)
        }
    }

    @MainActor
    private func startPingTimerOnMain() {
        pingTimer?.invalidate()
        pingTimer = Timer.scheduledTimer(withTimeInterval: LiveStreamConfig.pingIntervalSeconds, repeats: true) { [weak self] _ in
            self?.sendPing()
        }
    }

    private func sendPing() {
        guard let session, let peer = session.connectedPeers.first else { return }
        pingSequence &+= 1
        let epochMs = UInt64(Date().timeIntervalSince1970 * 1000)
        let data = LiveStreamFrameCodec.encodePing(sentAtEpochMs: epochMs, pingId: pingSequence)
        Task { @MainActor in
            model.latency.recordPingSent(id: pingSequence)
        }
        try? session.send(data, toPeers: [peer], with: .reliable)
    }

    private func handleReceived(data: Data, from peer: MCPeerID) {
        guard let type = LiveStreamFrameCodec.messageType(of: data) else { return }

        switch type {
        case .videoFrame:
            guard let frame = LiveStreamFrameCodec.decodeVideoFrame(from: data) else { return }
            let t0 = CFAbsoluteTimeGetCurrent()
            guard let image = UIImage(data: frame.jpegData) else { return }
            let decodeMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            Task { @MainActor in
                model.latestFrame = image
                model.latency.recordFrameReceived(hostSentEpochMs: frame.sentAtEpochMs, decodeMs: decodeMs)
            }

        case .ping:
            guard let decoded = LiveStreamFrameCodec.decodePingPong(from: data),
                  let session, model.role == .host else { return }
            let pong = LiveStreamFrameCodec.encodePong(sentAtEpochMs: decoded.epochMs, pingId: decoded.id)
            try? session.send(pong, toPeers: [peer], with: .reliable)

        case .pong:
            guard let decoded = LiveStreamFrameCodec.decodePingPong(from: data) else { return }
            Task { @MainActor in
                model.latency.recordPong(id: decoded.id)
            }
        }
    }
}

// MARK: - MCSessionDelegate

extension LiveStreamMultipeerService: MCSessionDelegate {
    func session(_ session: MCSession, peer peerID: MCPeerID, didChange state: MCSessionState) {
        Task { @MainActor in
            switch state {
            case .connected:
                updateConnectionState()
            case .notConnected:
                updateConnectionState()
            case .connecting:
                model.connectionState = "Conectando…"
            @unknown default:
                break
            }
        }
    }

    func session(_ session: MCSession, didReceive data: Data, fromPeer peerID: MCPeerID) {
        handleReceived(data: data, from: peerID)
    }

    func session(
        _ session: MCSession,
        didReceive stream: InputStream,
        withName streamName: String,
        fromPeer peerID: MCPeerID
    ) {}

    func session(
        _ session: MCSession,
        didStartReceivingResourceWithName resourceName: String,
        fromPeer peerID: MCPeerID,
        with progress: Progress
    ) {}

    func session(
        _ session: MCSession,
        didFinishReceivingResourceWithName resourceName: String,
        fromPeer peerID: MCPeerID,
        at localURL: URL?,
        withError error: Error?
    ) {}
}

// MARK: - Advertiser / Browser

extension LiveStreamMultipeerService: MCNearbyServiceAdvertiserDelegate {
    func advertiser(
        _ advertiser: MCNearbyServiceAdvertiser,
        didReceiveInvitationFromPeer peerID: MCPeerID,
        withContext context: Data?,
        invitationHandler: @escaping (Bool, MCSession?) -> Void
    ) {
        invitationHandler(true, session)
    }
}

extension LiveStreamMultipeerService: MCNearbyServiceBrowserDelegate {
    func browser(
        _ browser: MCNearbyServiceBrowser,
        foundPeer peerID: MCPeerID,
        withDiscoveryInfo info: [String: String]?
    ) {
        guard info?["role"] == "host", let session else { return }
        browser.invitePeer(peerID, to: session, withContext: nil, timeout: 15)
        Task { @MainActor in
            model.connectionState = "Invitando a \(peerID.displayName)…"
        }
    }

    func browser(_ browser: MCNearbyServiceBrowser, lostPeer peerID: MCPeerID) {}
}
