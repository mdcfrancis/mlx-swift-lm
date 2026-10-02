import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// A cache holding `prefix` keys/values, ready for `length` new positions.
private func cache(keys: MLXArray, values: MLXArray) -> KVCacheSimple {
    let cache = KVCacheSimple()
    cache.state = [keys, values]
    return cache
}

private func causal(length: Int, prefix: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let rows = MLXArray(Int32(0) ..< Int32(length)).reshaped(1, 1, length, 1)
    let columns = MLXArray(Int32(0) ..< Int32(prefix + length)).reshaped(1, 1, 1, prefix + length)
    return .array(columns .<= (rows + Int32(prefix)))
}

private struct Fixture {
    let heads = 8
    let kvHeads = 2
    let dim = 16
    let length = 3
    let roster = 4
    let slots = 3
    let rest = 5
    let queries: MLXArray
    let newKeys: MLXArray
    let newValues: MLXArray
    let prefixKeys: MLXArray
    let prefixValues: MLXArray

    init() {
        let key = MLXRandom.key(7)
        let parts = MLXRandom.split(key: key, into: 5)
        queries = MLXRandom.normal([1, heads, length, dim], key: parts[0])
        newKeys = MLXRandom.normal([1, kvHeads, length, dim], key: parts[1])
        newValues = MLXRandom.normal([1, kvHeads, length, dim], key: parts[2])
        prefixKeys = MLXRandom.normal([1, kvHeads, roster + slots + rest, dim], key: parts[3])
        prefixValues = MLXRandom.normal([1, kvHeads, roster + slots + rest, dim], key: parts[4])
    }

    var prefix: Int { roster + slots + rest }
    var scale: Float { 1 / Float(dim).squareRoot() }

    func attend(gateBias: Float?) -> MLXArray {
        let c = cache(keys: prefixKeys, values: prefixValues)
        if let gateBias {
            c.slotGate = SlotGate(start: roster, count: slots, bias: MLXArray(Array(repeating: gateBias, count: heads)))
        }
        return attentionWithCacheUpdate(
            queries: queries, keys: newKeys, values: newValues, cache: c, scale: scale,
            mask: causal(length: length, prefix: prefix))
    }

    /// The same attention with the slot columns removed from the prefix.
    func attendWithoutSlots() -> MLXArray {
        let keep = MLXArray((0 ..< prefix).filter { $0 < roster || $0 >= roster + slots }.map { Int32($0) })
        let c = cache(keys: prefixKeys[0..., 0..., keep, 0...], values: prefixValues[0..., 0..., keep, 0...])
        return attentionWithCacheUpdate(
            queries: queries, keys: newKeys, values: newValues, cache: c, scale: scale,
            mask: causal(length: length, prefix: prefix - slots))
    }
}

@Test func slotGateAtZeroMatchesUngatedAttention() {
    let f = Fixture()
    let difference = abs(f.attend(gateBias: 0) - f.attend(gateBias: nil)).max().item(Float.self)
    #expect(difference < 1e-5)
}

@Test func slotGateFarNegativeMatchesRemovingTheSlots() {
    let f = Fixture()
    let difference = abs(f.attend(gateBias: -1e4) - f.attendWithoutSlots()).max().item(Float.self)
    #expect(difference < 1e-5)
}

@Test func slotGateChangesAttentionInBetween() {
    let f = Fixture()
    let difference = abs(f.attend(gateBias: -2) - f.attend(gateBias: nil)).max().item(Float.self)
    #expect(difference > 1e-3)
}

@Test func raggedCacheCarriesTheSlotGate() {
    let f = Fixture()
    let single = cache(keys: f.prefixKeys, values: f.prefixValues)
    single.slotGate = SlotGate(start: f.roster, count: f.slots, bias: MLXArray.zeros([f.heads]))
    let ragged = RaggedKVCache(expanding: single, rows: 2)
    #expect(ragged.slotGate?.start == f.roster)
    #expect(ragged.slotGate?.count == f.slots)
    #expect((ragged.copy() as? RaggedKVCache)?.slotGate?.count == f.slots)
    #expect(ragged.extract(row: 1).slotGate?.count == f.slots)
    #expect((single.copy() as? KVCacheSimple)?.slotGate?.count == f.slots)
}
