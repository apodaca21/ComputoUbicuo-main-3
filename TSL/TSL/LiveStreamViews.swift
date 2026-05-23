//
//  LiveStreamViews.swift
//  TSL — UI US-09: host (iPad) y viewer (iPhone)
//

import SwiftUI
import UIKit

// MARK: - Host (iPad transmite)

struct LiveStreamHostView: View {
    @Binding var isPresented: Bool
    @StateObject private var model = LiveStreamSessionModel(role: .host)
    @State private var service: LiveStreamMultipeerService?
    @State private var cameraHost = LiveStreamCameraHost()
    @State private var sentCount = 0
    @State private var droppedCount = 0

    var body: some View {
        ZStack {
            LiveStreamHostPreview(
                cameraHost: cameraHost,
                onViewReady: { view in
                    if service == nil {
                        let svc = LiveStreamMultipeerService(model: model)
                        service = svc
                        svc.start()
                        cameraHost.attach(to: view, multipeer: svc)
                        cameraHost.onStatsUpdate = { sent, dropped in
                            sentCount = sent
                            droppedCount = dropped
                        }
                    }
                }
            )
            .ignoresSafeArea()

            VStack {
                HStack {
                    Button { close() } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 32))
                            .foregroundStyle(.white)
                            .shadow(radius: 4)
                    }
                    Spacer()
                }
                .padding()

                LiveStreamStatusBadge(
                    connectionState: model.connectionState,
                    peerName: model.connectedPeerName,
                    extra: "Enviados: \(sentCount) · Descartados: \(droppedCount)"
                )

                Spacer()

                Text("Modo iPad — transmisión en vivo")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.white)
                    .padding(10)
                    .background(.black.opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .padding(.bottom, 32)
            }
        }
        .onDisappear {
            cameraHost.stop()
            service?.stop()
        }
    }

    private func close() {
        cameraHost.stop()
        service?.stop()
        isPresented = false
    }
}

private struct LiveStreamHostPreview: UIViewRepresentable {
    let cameraHost: LiveStreamCameraHost
    let onViewReady: (UIView) -> Void

    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: .zero)
        view.backgroundColor = .black
        DispatchQueue.main.async {
            onViewReady(view)
        }
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        cameraHost.layoutPreview()
    }
}

// MARK: - Viewer (iPhone recibe)

struct LiveStreamViewerView: View {
    @Binding var isPresented: Bool
    @StateObject private var model = LiveStreamSessionModel(role: .viewer)
    @State private var service: LiveStreamMultipeerService?

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let image = model.latestFrame {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                VStack(spacing: 12) {
                    ProgressView()
                        .tint(.white)
                    Text(model.connectionState)
                        .foregroundStyle(.white.opacity(0.9))
                }
            }

            VStack {
                HStack {
                    Button { close() } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 32))
                            .foregroundStyle(.white)
                            .shadow(radius: 4)
                    }
                    Spacer()
                }
                .padding()

                LiveStreamStatusBadge(
                    connectionState: model.connectionState,
                    peerName: model.connectedPeerName,
                    extra: nil
                )

                LiveStreamLatencyBadge(latency: model.latency)

                Spacer()
            }
        }
        .onAppear {
            if service == nil {
                let svc = LiveStreamMultipeerService(model: model)
                service = svc
                svc.start()
            }
        }
        .onDisappear {
            service?.stop()
            model.latency.reset()
        }
    }

    private func close() {
        service?.stop()
        isPresented = false
    }
}

// MARK: - Componentes UI

private struct LiveStreamStatusBadge: View {
    let connectionState: String
    let peerName: String?
    let extra: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Circle()
                    .fill(peerName != nil ? Color.green : Color.orange)
                    .frame(width: 8, height: 8)
                Text(connectionState)
                    .font(.caption.weight(.semibold))
            }
            if let peerName {
                Text(peerName)
                    .font(.caption2)
            }
            if let extra {
                Text(extra)
                    .font(.caption2.monospacedDigit())
            }
        }
        .foregroundStyle(.white)
        .padding(10)
        .background(.black.opacity(0.55))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal)
    }
}

private struct LiveStreamLatencyBadge: View {
    @ObservedObject var latency: LiveStreamLatencyTracker

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Latencia estimada")
                    .font(.caption.weight(.semibold))
                Spacer()
                Text(latency.meetsTarget ? "OK <300ms" : "ALERTA")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(latency.meetsTarget ? .green : .red)
            }
            Text("\(Int(latency.estimatedLatencyMs)) ms")
                .font(.title2.weight(.bold).monospacedDigit())
            Text("Decode+UI: \(Int(latency.decodeDisplayMs)) ms · \(String(format: "%.1f", latency.averageFps)) fps")
                .font(.caption2.monospacedDigit())
            Text("Frames: \(latency.framesReceived) · perdidos: \(latency.framesDropped)")
                .font(.caption2)
        }
        .foregroundStyle(.white)
        .padding(12)
        .background(.black.opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }
}
