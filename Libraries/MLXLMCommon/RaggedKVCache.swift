// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXFast

/// A key/value cache for a batch of sequences that share a prefix and then
/// diverge by different amounts per round — the shape batched speculative
/// decoding produces, where every row verifies the same block width but
/// keeps a different number of its tokens.
///
/// Rows share one physical buffer `[B, heads, T, D]`; each row has its own
/// logical `lengths[r]`. A write of `L` positions lands at
/// `lengths[r] ..< lengths[r] + L` for every row (a scatter, not a
/// concatenation), and the attention mask lets a query at row position
/// `lengths[r] + j` see physical columns `..< lengths[r] + j + 1` only, so
/// whatever an earlier round left beyond a row's logical end is invisible
/// and is overwritten by the next write.
///
/// Positions must be supplied to the model per row (it cannot derive them
/// from `offset`, which here is the physical length).
public final class RaggedKVCache: BaseKVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    /// Logical length per row.
    public private(set) var lengths: [Int]
    /// The lengths before the last `update`, so `rewind(keep:)` can keep a
    /// per-row part of what that update wrote.
    private var lengthsBeforeLastWrite: [Int]
    public var step = 256

    public init(rows: Int) {
        lengths = Array(repeating: 0, count: rows)
        lengthsBeforeLastWrite = lengths
        super.init()
    }

    /// Every row starts as a copy of a single-row cache.
    public convenience init(expanding single: KVCache, rows: Int) {
        self.init(rows: rows)
        let state = single.state
        guard state.count == 2 else {
            fatalError("RaggedKVCache: cannot expand a cache with \(state.count) state arrays")
        }
        let length = single.offset
        keys = repeated(state[0][.ellipsis, ..<length, 0...], count: rows, axis: 0)
        values = repeated(state[1][.ellipsis, ..<length, 0...], count: rows, axis: 0)
        lengths = Array(repeating: length, count: rows)
        lengthsBeforeLastWrite = lengths
        offset = length
    }

    public var rows: Int { lengths.count }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            return [keys[.ellipsis, ..<offset, 0...], values[.ellipsis, ..<offset, 0...]]
        }
        set {
            guard newValue.count == 2 else { fatalError("RaggedKVCache state must be [keys, values]") }
            keys = newValue[0]
            values = newValue[1]
        }
    }

    public override var isTrimmable: Bool { true }

    /// Uniform trim (every row loses `n`), for callers that treat the cache
    /// as a single sequence.
    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let n = min(n, lengths.min() ?? 0)
        lengths = lengths.map { $0 - n }
        lengthsBeforeLastWrite = lengthsBeforeLastWrite.map { max(0, $0 - n) }
        offset = lengths.max() ?? 0
        return n
    }

    public override func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let B = newKeys.dim(0)
        precondition(B == rows, "RaggedKVCache: \(B) rows written into a \(rows)-row cache")
        let L = newKeys.dim(2)
        let needed = (lengths.max() ?? 0) + L
        ensureCapacity(needed, like: newKeys, values: newValues)

        // Scatter each row's block at its own logical end.
        let starts = MLXArray(lengths.map { Int32($0) }).reshaped(B, 1, 1, 1)
        let index = starts + MLXArray(Int32(0) ..< Int32(L)).reshaped(1, 1, L, 1)
        let keyIndex = broadcast(index, to: newKeys.shape)
        let valueIndex = broadcast(index, to: newValues.shape)
        keys = putAlong(keys!, keyIndex, values: newKeys.asType(keys!.dtype), axis: 2)
        values = putAlong(values!, valueIndex, values: newValues.asType(values!.dtype), axis: 2)

        lengthsBeforeLastWrite = lengths
        lengths = lengths.map { $0 + L }
        offset = needed
        return (keys![.ellipsis, ..<offset, 0...], values![.ellipsis, ..<offset, 0...])
    }

    private func ensureCapacity(_ needed: Int, like keyTemplate: MLXArray, values valueTemplate: MLXArray) {
        let current = keys?.dim(2) ?? 0
        guard needed > current else { return }
        let grow = ((needed - current + step - 1) / step) * step
        let kShape = [rows, keyTemplate.dim(1), grow, keyTemplate.dim(3)]
        let vShape = [rows, valueTemplate.dim(1), grow, valueTemplate.dim(3)]
        let newK = MLXArray.zeros(kShape, dtype: keys?.dtype ?? keyTemplate.dtype)
        let newV = MLXArray.zeros(vShape, dtype: values?.dtype ?? valueTemplate.dtype)
        if let keys, let values {
            self.keys = concatenated([keys, newK], axis: 2)
            self.values = concatenated([values, newV], axis: 2)
        } else {
            keys = newK
            values = newV
        }
    }

    /// Keep `keep[r]` of the positions the last `update` wrote for row `r`
    /// and drop the rest (the rejected drafts).
    public func rewind(keep: [Int]) {
        precondition(keep.count == rows)
        lengths = zip(lengthsBeforeLastWrite, keep).map { $0 + $1 }
        offset = lengths.max() ?? 0
    }

    /// Keep only the given rows.
    public func filter(rows kept: [Int]) {
        let index = MLXArray(kept.map { Int32($0) })
        keys = keys?[index]
        values = values?[index]
        lengths = kept.map { lengths[$0] }
        lengthsBeforeLastWrite = kept.map { lengthsBeforeLastWrite[$0] }
        offset = lengths.max() ?? 0
    }

    /// The mask for `n` new positions per row against the physical columns
    /// the next `update` will expose: row `r`'s query `j` sees columns
    /// `..< lengths[r] + j + 1`. Always an array; a single position still
    /// has stale columns to hide.
    public override func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let columns = (lengths.max() ?? 0) + n
        let rowLengths = MLXArray(lengths.map { Int32($0) }).reshaped(rows, 1, 1, 1)
        let query = MLXArray(Int32(0) ..< Int32(n)).reshaped(1, 1, n, 1)
        let column = MLXArray(Int32(0) ..< Int32(columns)).reshaped(1, 1, 1, columns)
        var mask = column .<= (rowLengths + query)
        if let windowSize {
            mask = mask .&& (column .> (rowLengths + query - Int32(windowSize)))
        }
        return .array(mask)
    }

    public override func copy() -> any KVCache {
        let new = RaggedKVCache(rows: rows)
        new.keys = keys
        new.values = values
        new.lengths = lengths
        new.lengthsBeforeLastWrite = lengthsBeforeLastWrite
        new.offset = offset
        new.step = step
        return new
    }
}
