//
//  LiveStreamFrameCodec.swift
//  TSL — Paquetes de video JPEG + mensajes de control (ping/pong)
//

import CoreGraphics
import CoreVideo
import Foundation
import UIKit

enum LiveStreamMessageType: UInt8 {
    case videoFrame = 1
    case ping = 2
    case pong = 3
}

struct LiveStreamVideoFrame {
    let sequence: UInt32
    let sentAtEpochMs: UInt64
    let width: UInt16
    let height: UInt16
    let jpegData: Data
}

enum LiveStreamFrameCodec {
    private static let headerSize = 19 // type(1) + seq(4) + time(8) + w(2) + h(2)

    static func encodeVideoFrame(_ frame: LiveStreamVideoFrame) -> Data {
        var data = Data(capacity: headerSize + frame.jpegData.count)
        data.append(LiveStreamMessageType.videoFrame.rawValue)
        var seq = frame.sequence.bigEndian
        var ts = frame.sentAtEpochMs.bigEndian
        var w = frame.width.bigEndian
        var h = frame.height.bigEndian
        withUnsafeBytes(of: &seq) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &ts) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &w) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &h) { data.append(contentsOf: $0) }
        data.append(frame.jpegData)
        return data
    }

    static func decodeVideoFrame(from data: Data) -> LiveStreamVideoFrame? {
        guard data.count > headerSize,
              data[data.startIndex] == LiveStreamMessageType.videoFrame.rawValue else { return nil }
        let seq = data.subdata(in: 1 ..< 5).withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        let ts = data.subdata(in: 5 ..< 13).withUnsafeBytes { $0.load(as: UInt64.self).bigEndian }
        let w = data.subdata(in: 13 ..< 15).withUnsafeBytes { $0.load(as: UInt16.self).bigEndian }
        let h = data.subdata(in: 15 ..< 17).withUnsafeBytes { $0.load(as: UInt16.self).bigEndian }
        let jpeg = data.subdata(in: headerSize ..< data.count)
        return LiveStreamVideoFrame(
            sequence: seq,
            sentAtEpochMs: ts,
            width: w,
            height: h,
            jpegData: jpeg
        )
    }

    static func encodePing(sentAtEpochMs: UInt64, pingId: UInt32) -> Data {
        var data = Data(capacity: 13)
        data.append(LiveStreamMessageType.ping.rawValue)
        var ts = sentAtEpochMs.bigEndian
        var id = pingId.bigEndian
        withUnsafeBytes(of: &ts) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &id) { data.append(contentsOf: $0) }
        return data
    }

    static func encodePong(sentAtEpochMs: UInt64, pingId: UInt32) -> Data {
        var data = Data(capacity: 13)
        data.append(LiveStreamMessageType.pong.rawValue)
        var ts = sentAtEpochMs.bigEndian
        var id = pingId.bigEndian
        withUnsafeBytes(of: &ts) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &id) { data.append(contentsOf: $0) }
        return data
    }

    static func messageType(of data: Data) -> LiveStreamMessageType? {
        guard let first = data.first else { return nil }
        return LiveStreamMessageType(rawValue: first)
    }

    static func decodePingPong(from data: Data) -> (epochMs: UInt64, id: UInt32)? {
        guard data.count >= 13 else { return nil }
        let ts = data.subdata(in: 1 ..< 9).withUnsafeBytes { $0.load(as: UInt64.self).bigEndian }
        let id = data.subdata(in: 9 ..< 13).withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        return (ts, id)
    }

    /// Redimensiona y comprime un CVPixelBuffer vía UIImage para controlar tamaño de red.
    static func compressPixelBuffer(
        _ pixelBuffer: CVPixelBuffer,
        maxLongSide: CGFloat = LiveStreamConfig.maxFrameLongSide,
        quality: CGFloat = LiveStreamConfig.jpegQuality
    ) -> (Data, Int, Int)? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        var image = UIImage(cgImage: cgImage)

        let w = image.size.width
        let h = image.size.height
        let longSide = max(w, h)
        if longSide > maxLongSide {
            let scale = maxLongSide / longSide
            let newSize = CGSize(width: w * scale, height: h * scale)
            UIGraphicsBeginImageContextWithOptions(newSize, true, 1.0)
            image.draw(in: CGRect(origin: .zero, size: newSize))
            if let resized = UIGraphicsGetImageFromCurrentImageContext() {
                image = resized
            }
            UIGraphicsEndImageContext()
        }

        guard let jpeg = image.jpegData(compressionQuality: quality) else { return nil }
        return (jpeg, Int(image.size.width), Int(image.size.height))
    }
}
