// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@_spi(Testing) @testable import MLXLMCommon

private func dflashConfigJSON(
    layers: Int = 2, hidden: Int = 32, heads: Int = 4, kvHeads: Int = 2, headDim: Int = 8,
    intermediate: Int = 64, vocab: Int = 64, block: Int = 4, taps: [Int] = [1, 3],
    selector: Bool = true, conv: Bool = true
) -> String {
    var dflash = """
        "block_size": \(block), "target_layer_ids": \(taps), "mask_token_id": 63
        """
    if conv { dflash += ", \"conv_kernel_size\": 2, \"conv_group_size\": 8" }
    if selector { dflash += ", \"selector_rank\": 8, \"selector_top_k\": 4" }
    return """
        {
          "architectures": ["DFlash2DraftModel"],
          "model_type": "qwen3",
          "hidden_size": \(hidden), "num_hidden_layers": \(layers), "intermediate_size": \(intermediate),
          "num_attention_heads": \(heads), "num_key_value_heads": \(kvHeads), "head_dim": \(headDim),
          "rms_norm_eps": 1e-6, "vocab_size": \(vocab),
          "rope_parameters": { "rope_theta": 10000 },
          "sliding_window": 16, "layer_types": ["sliding_attention", "sliding_attention"],
          "is_causal": false, "attention_bias": false,
          "dflash_config": { \(dflash) }
        }
        """
}

@Test
func testDFlashConfigurationDecodes() throws {
    let config = try JSONDecoder.json5().decode(
        DFlashDraftConfiguration.self, from: Data(dflashConfigJSON().utf8))
    #expect(config.hiddenLayers == 2)
    #expect(config.dflash.blockSize == 4)
    #expect(config.dflash.targetLayerIds == [1, 3])
    #expect(config.ropeTheta == 10000)
    #expect(config.hasConvolutions)
    #expect(config.hasSelector)
    #expect(config.windowSize(layer: 0) == 16)
    #expect(DFlashDraftConfiguration.describes(Data(dflashConfigJSON().utf8)))
    #expect(!DFlashDraftConfiguration.describes(Data("{\"model_type\": \"qwen3\"}".utf8)))
}

@Test
func testDFlashDraftModelShapesAndParameters() throws {
    let config = try JSONDecoder.json5().decode(
        DFlashDraftConfiguration.self, from: Data(dflashConfigJSON().utf8))
    let draft = DFlashDraftModel(config)
    #expect(draft.blockSize == 4)
    #expect(draft.maximumBlockSize == 4)
    #expect(draft.targetTapLayers == [1, 3])
    #expect(draft.consumesFullContextHidden)
    #expect(draft.requiresGreedySampling)
    #expect(draft.requiresPromptPrefill)

    // The reference checkpoint stores the selector codebooks without `.weight`.
    let sanitized = draft.sanitize(weights: [
        "candidate_selector.predecessor_codebook": MLXArray.zeros([64, 8]),
        "candidate_selector.successor_codebook": MLXArray.zeros([64, 8]),
        "fc.weight": MLXArray.zeros([32, 64]),
    ])
    #expect(sanitized["candidate_selector.predecessor_codebook.weight"] != nil)
    #expect(sanitized["candidate_selector.predecessor_codebook"] == nil)
    #expect(sanitized["fc.weight"] != nil)

    let keys = Set(draft.parameters().flattened().map(\.0))
    #expect(keys.contains("fc.weight"))
    #expect(keys.contains("hidden_norm.weight"))
    #expect(keys.contains("layers.0.attention_conv.base_kernel"))
    #expect(keys.contains("layers.1.mlp_conv.kernel_projection.weight"))
    #expect(keys.contains("candidate_selector.hidden_projection.weight"))
    #expect(keys.contains("layers.0.self_attn.q_norm.weight"))
}

@Test
func testDFlashDraftForwardProducesBlockHidden() throws {
    let config = try JSONDecoder.json5().decode(
        DFlashDraftConfiguration.self, from: Data(dflashConfigJSON().utf8))
    let draft = DFlashDraftModel(config)
    // Two tapped layers of hidden size 32 → 64-wide target features.
    let taps = MLXRandom.normal([1, 5, 64]).asType(.bfloat16)
    let context = draft.debugProjectContext(taps)
    #expect(context.shape == [1, 5, 32])
    let noise = MLXRandom.normal([1, 4, 32]).asType(.bfloat16)
    let hidden = draft.debugForward(context: context, noiseEmbedding: noise)
    eval(hidden)
    #expect(hidden.shape == [1, 4, 32])
    #expect(!isNaN(hidden.asType(.float32)).any().item(Bool.self))

    let logits = MLXRandom.normal([1, 3, 64])
    let selected = try #require(
        draft.debugSelect(
            hidden: hidden[0..., 1..., 0...], logits: logits, anchor: MLXArray([Int32(7)])))
    eval(selected)
    #expect(selected.shape == [1, 3])
    // Every selected token is one of that position's top-k candidates.
    let picks = selected.asArray(Int32.self)
    for (position, pick) in picks.enumerated() {
        let row = logits[0, position].asArray(Float.self)
        let threshold = row.sorted(by: >)[3]
        #expect(row[Int(pick)] >= threshold)
    }
}

@Test
func testDFlashContextCacheKeepsSinkAndWindow() {
    let cache = DFlashContextCache(sinkSize: 2, windowSize: 3)
    #expect(cache.spansToAppend(count: 4) == [0 ..< 4])
    #expect(cache.spansToAppend(count: 10) == [0 ..< 2, 7 ..< 10])
    let keys = MLXRandom.normal([1, 2, 7, 4])
    let values = MLXRandom.normal([1, 2, 7, 4])
    cache.append(keys: keys, values: values, start: 0)
    #expect(cache.length == 5)
    #expect(cache.positions == [0, 1, 4, 5, 6])
    #expect(cache.keys?.shape == [1, 2, 5, 4])
    #expect(cache.spansToAppend(count: 2) == [0 ..< 2])
}

@Test
func testDFlashBlockPolicyWidensWithAcceptance() throws {
    let config = try JSONDecoder.json5().decode(
        DFlashDraftConfiguration.self, from: Data(dflashConfigJSON(block: 8).utf8))
    let draft = DFlashDraftModel(config)
    // Never below 4, never above the maximum, wider as acceptance rises.
    #expect(draft.nextBlockSize(afterAccepting: 0, current: 8, maximum: 8) == 4)
    var block = 4
    for _ in 0 ..< 6 { block = draft.nextBlockSize(afterAccepting: 7, current: block, maximum: 8) }
    #expect(block == 8)
    for _ in 0 ..< 6 { block = draft.nextBlockSize(afterAccepting: 0, current: block, maximum: 8) }
    #expect(block == 4)
}

@Test
func testDFlashRegistrationMatchesOnlyDFlashConfigs() async throws {
    await DFlashDrafterRegistration.register()
    let dflash = Data(dflashConfigJSON().utf8)
    let model = try await MTPDrafterTypeRegistry.shared.createModel(
        configuration: dflash, modelType: "qwen3")
    #expect(model is DFlashDraftModel)
}

@Test
func testSpeculativeOptionsPadVerifyRowsOnlyWhenWorthIt() {
    var options = SpeculativeOptions()
    options.verifyRowMultiples = [4, 16]
    #expect(options.verifyRows(for: 1) == 1)
    #expect(options.verifyRows(for: 2) == 2)
    #expect(options.verifyRows(for: 3) == 4)
    #expect(options.verifyRows(for: 4) == 4)
    #expect(options.verifyRows(for: 5) == 5)
    #expect(options.verifyRows(for: 8) == 8)
    #expect(options.verifyRows(for: 9) == 16)
    #expect(options.verifyRows(for: 16) == 16)
    #expect(options.verifyRows(for: 17) == 17)
    options.verifyRowMultiples = []
    #expect(options.verifyRows(for: 3) == 3)
}

@Test
func testMambaCacheTapeRestoresBookkeeping() {
    let cache = MambaCache()
    cache[0] = MLXArray.zeros([1, 3, 8])
    cache[1] = MLXArray.zeros([1, 2, 4, 4])
    let before: [MLXArray?] = [cache[0], cache[1]]
    cache.recordSpeculativeTape(
        stateBefore: before, positions: 4, operands: ["q": MLXArray.zeros([1, 4, 2, 4])])
    cache.advance(4)
    #expect(cache.speculativeTape?.positions == 4)
    let conv = MLXArray.ones([1, 3, 8])
    let recurrent = MLXArray.ones([1, 2, 4, 4])
    cache.restoreFromSpeculativeTape(keep: 1, convState: conv, recurrentState: recurrent)
    #expect(cache.speculativeTape == nil)
    #expect(cache[0]?.sum().item(Float.self) == 24)
    #expect(cache[1]?.sum().item(Float.self) == 32)
    cache.discardSpeculativeCheckpoint()
    #expect(cache.speculativeTape == nil)
}

@Test
func testQwenMTPSanitizeAcceptsPrefixedKeys() {
    let sanitized = qwenMTPSanitizeWeights(
        weights: [
            "language_model.mtp.fc.weight": MLXArray.zeros([4, 8]),
            "model.mtp.norm.weight": MLXArray.zeros([4]),
            "mtp.layers.0.mlp.down_proj.weight": MLXArray.zeros([4, 4]),
            "language_model.model.layers.0.mlp.down_proj.weight": MLXArray.zeros([4, 4]),
        ],
        mtpNumHiddenLayers: 1, numExperts: 0, shiftNormWeights: false)
    #expect(
        Set(sanitized.keys) == [
            "mtp.fc.weight", "mtp.norm.weight", "mtp.layers.0.mlp.down_proj.weight",
        ])
}
