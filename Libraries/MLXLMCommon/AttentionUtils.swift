import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let turboCache = cache as? TurboQuantKVCache {
        let L = queries.dim(2)
        if L > 1 && !turboCache.isCompressed {
            // Prefill (L>1) on a raw cache: plain update + standard SDPA, // zero overhead; compression is deferred to the first decode step.
            let (cachedKeys, cachedValues) = turboCache.update(keys: keys, values: values)
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: cachedKeys, values: cachedValues,
                scale: scale, mask: mask
            )
        }
        // Decode (L=1) or any call once the cache is compressed (speculative
        // verify chunks, multi-turn re-prefill): the compressed path. The raw
        // update() path is invalid after compression, its raw buffers are
        // gone. First decode call triggers compressRawCache().
        return turboCache.compressedAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask
        )
    } else if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        if let gate = (cache as? BaseKVCache)?.slotGate, gate.count > 0 {
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: .array(gatedMask(gate, mask: mask, queries: queries, keyCount: cachedKeys.dim(2)))
            )
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }
}

/// The additive mask for a gated cache: the caller's mask as 0 / -1e9, plus
/// the gate's per-head bias on its columns. Shape [B or 1, heads, L, keys].
private func gatedMask(
    _ gate: SlotGate, mask: MLXFast.ScaledDotProductAttentionMaskMode, queries: MLXArray, keyCount: Int
) -> MLXArray {
    let heads = queries.dim(1)
    let length = queries.dim(2)
    let columns = MLXArray(Int32(0) ..< Int32(keyCount)).reshaped(1, 1, 1, keyCount)
    let base: MLXArray
    switch mask {
    case .none:
        base = MLXArray.zeros([1, 1, length, keyCount], dtype: .float32)
    case .causal:
        let rows = MLXArray(Int32(0) ..< Int32(length)).reshaped(1, 1, length, 1)
        base = which(columns .<= (rows + Int32(keyCount - length)), Float(0), Float(-1e9))
    case .array(let array):
        base = array.dtype == .bool ? which(array, Float(0), Float(-1e9)) : array.asType(.float32)
    case .arrays(let arrays):
        let array = arrays[0]
        base = array.dtype == .bool ? which(array, Float(0), Float(-1e9)) : array.asType(.float32)
    }
    let inGate = (columns .>= Int32(gate.start)) .&& (columns .< Int32(gate.start + gate.count))
    let bias = which(inGate, Float(1), Float(0)) * gate.bias.asType(.float32).reshaped(1, heads, 1, 1)
    return (base + bias).asType(queries.dtype)
}
