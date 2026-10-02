// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Standard rotary position encoding (half-split layout) over the first
/// `dimensions` of the last axis, at arbitrary, possibly fractional,
/// positions. A negative position undoes a rotation, so cached keys can be
/// made position-free and re-rotated elsewhere.
///
/// For text this equals Qwen3.5's interleaved multimodal rotary, because all
/// three position axes carry the same position.
///
/// - Parameters:
///   - x: [..., n, D]
///   - positions: [n]
/// - Returns: float32, same shape as `x`.
public func rotaryRotate(_ x: MLXArray, positions: MLXArray, dimensions: Int, base: Float) -> MLXArray {
    let half = dimensions / 2
    let exponents = MLXArray(stride(from: 0, to: dimensions, by: 2).map { Float($0) / Float(dimensions) })
    let inverse = 1.0 / pow(MLXArray(base), exponents)  // [half]
    let theta = positions.asType(.float32).reshaped(-1, 1) * inverse.reshaped(1, -1)  // [n, half]
    let c = cos(theta)
    let s = sin(theta)
    let xf = x.asType(.float32)
    let x1 = xf[.ellipsis, ..<half]
    let x2 = xf[.ellipsis, half ..< dimensions]
    var parts = [x1 * c - x2 * s, x2 * c + x1 * s]
    if dimensions < x.dim(-1) {
        parts.append(xf[.ellipsis, dimensions...])
    }
    return concatenated(parts, axis: -1)
}
