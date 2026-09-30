import Foundation
import MLX
import Testing

@testable import MLXLMCommon
@testable import MLXVLM

/// `rotaryRotate` must reproduce the model's own rotary encoding for text
/// positions, and undo it exactly.
@Test func rotaryRotateMatchesQwen35TextRotary() {
    let dimensions = 64
    let base: Float = 10_000_000
    let x = MLXRandom.normal([1, 4, 5, 256], key: MLXRandom.key(11))
    let positions = MLXArray([Int32(0), 17, 18, 400, 5000])
    let rope = Qwen35Language.RotaryEmbedding(dim: dimensions, base: base, mropeSection: [11, 11, 10])
    let (c, s) = rope(x: x, positionIds: positions.reshaped(1, 5))
    let (_, reference) = Qwen35Language.applyMultimodalRotaryPosEmb(q: x, k: x, cos: c, sin: s)
    let rotated = rotaryRotate(x, positions: positions, dimensions: dimensions, base: base)
    #expect(abs(rotated - reference.asType(.float32)).max().item(Float.self) < 1e-3)
    let undone = rotaryRotate(rotated, positions: -positions, dimensions: dimensions, base: base)
    #expect(abs(undone - x).max().item(Float.self) < 1e-3)
}

/// A query tap keeps the last `keep` positions across calls and does not
/// change attention.
@Test func queryTapKeepsTheLastPositions() {
    let cache = KVCacheSimple()
    let tap = QueryTap(keep: 3)
    cache.queryTap = tap
    let key = MLXRandom.key(5)
    let parts = MLXRandom.split(key: key, into: 6)
    for i in 0 ..< 2 {
        let q = MLXRandom.normal([1, 8, 2, 16], key: parts[3 * i])
        let k = MLXRandom.normal([1, 2, 2, 16], key: parts[3 * i + 1])
        let v = MLXRandom.normal([1, 2, 2, 16], key: parts[3 * i + 2])
        _ = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: cache, scale: 0.25, mask: .none)
    }
    #expect(tap.queries?.shape == [1, 8, 3, 16])
    let lastCall = MLXRandom.normal([1, 8, 2, 16], key: parts[3])
    #expect(abs(tap.queries![0..., 0..., 1..., 0...] - lastCall).max().item(Float.self) < 1e-6)
}
