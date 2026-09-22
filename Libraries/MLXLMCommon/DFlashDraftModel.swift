// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// DFlash speculative decoding (arXiv:2602.06036): a small block-diffusion
// draft proposes a whole block of tokens in one non-causal pass, conditioned
// on hidden states tapped from several of the target's decoder layers; the
// target verifies the block in one pass. DFlash 2 adds two-tap dynamic
// convolutions inside each draft layer and a path selector that picks a
// coherent token sequence through the block's top-k candidates.
//
// The draft is an ``MTPDrafterModel``: the target publishes the tapped
// layers for every position it processes (``MTPDrafterModel/targetTapLayers``),
// the draft keeps them, projected, in a sink-plus-window context cache, and
// each round drafts `blockSize - 1` tokens after the round's bonus token.
// Checkpoints follow the reference `DFlash2DraftModel` layout (z-lab).

/// What a DFlash draft needs from its target.
public protocol DFlashTargetModel: LanguageModel {
    /// The token embedding the draft shares with the target.
    var dflashTokenEmbedding: Embedding { get }
    /// Logits for draft hidden states through the target's output head.
    func dflashLogits(_ hidden: MLXArray) -> MLXArray
    /// Scale the target applies to token embeddings (1 for Qwen, √dim for Gemma).
    var dflashEmbeddingScale: Float { get }
}

extension DFlashTargetModel {
    public var dflashEmbeddingScale: Float { 1 }
}

public struct DFlashDraftConfiguration: Codable, Sendable {
    public struct DFlashConfig: Codable, Sendable {
        public var blockSize: Int
        public var targetLayerIds: [Int]
        public var maskTokenId: Int
        public var convKernelSize: Int?
        public var convGroupSize: Int?
        public var selectorRank: Int?
        public var selectorTopK: Int?
        public var inputEmbeddingScale: Float?

        enum CodingKeys: String, CodingKey {
            case blockSize = "block_size"
            case targetLayerIds = "target_layer_ids"
            case maskTokenId = "mask_token_id"
            case convKernelSize = "conv_kernel_size"
            case convGroupSize = "conv_group_size"
            case selectorRank = "selector_rank"
            case selectorTopK = "selector_top_k"
            case inputEmbeddingScale = "input_embedding_scale"
        }
    }

    public var architectures: [String]?
    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var intermediateSize: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var vocabularySize: Int
    public var ropeTheta: Float
    public var slidingWindow: Int?
    public var layerTypes: [String]?
    public var isCausal: Bool?
    public var attentionBias: Bool
    public var dflash: DFlashConfig

    enum CodingKeys: String, CodingKey {
        case architectures
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case isCausal = "is_causal"
        case attentionBias = "attention_bias"
        case dflash = "dflash_config"
    }

    private struct RopeParameters: Codable {
        var ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / attentionHeads)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        let parameters = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        ropeTheta =
            try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? parameters?.ropeTheta ?? 10000
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        isCausal = try c.decodeIfPresent(Bool.self, forKey: .isCausal)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        dflash = try c.decode(DFlashConfig.self, forKey: .dflash)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(dflash, forKey: .dflash)
    }

    /// True when `config.json` describes a DFlash draft (any base model type).
    public static func describes(_ data: Data) -> Bool {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        return root["dflash_config"] is [String: Any]
    }

    var hasConvolutions: Bool { dflash.convKernelSize != nil && dflash.convGroupSize != nil }
    var hasSelector: Bool { dflash.selectorRank != nil && dflash.selectorTopK != nil }
    func windowSize(layer: Int) -> Int? {
        guard let slidingWindow, slidingWindow > 0 else { return nil }
        if let layerTypes, layer < layerTypes.count {
            return layerTypes[layer] == "sliding_attention" ? slidingWindow : nil
        }
        return slidingWindow
    }
}

// MARK: - Context cache

/// Per-layer keys and values of the target's projected context, kept as a
/// sink of the first positions plus a window of the latest ones. Positions
/// are absolute sequence positions (the keys carry their rotary phase).
public final class DFlashContextCache: BaseKVCache {
    public let sinkSize: Int
    public let windowSize: Int
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    /// One absolute position per cached entry.
    public private(set) var positions: [Int32] = []

    public init(sinkSize: Int = 64, windowSize: Int = 1024) {
        self.sinkSize = sinkSize
        self.windowSize = windowSize
        super.init()
    }

    public var length: Int { positions.count }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public override var state: [MLXArray] {
        get { innerState() }
        set {
            keys = newValue.count > 0 ? newValue[0] : nil
            values = newValue.count > 1 ? newValue[1] : nil
        }
    }

    /// Append rotary-applied keys and values `[B, kvHeads, n, D]` for
    /// absolute positions `start ..< start + n`, then apply the window.
    func append(keys newKeys: MLXArray, values newValues: MLXArray, start: Int) {
        let n = newKeys.dim(2)
        guard n > 0 else { return }
        if let keys, let values {
            self.keys = concatenated([keys, newKeys], axis: 2)
            self.values = concatenated([values, newValues], axis: 2)
        } else {
            keys = newKeys
            values = newValues
        }
        positions.append(contentsOf: (0 ..< n).map { Int32(start + $0) })
        offset = Int(positions.last.map { $0 + 1 } ?? 0)
        let limit = sinkSize + windowSize
        if positions.count > limit, let keys, let values {
            let total = positions.count
            let sink = 0 ..< sinkSize
            let window = (total - windowSize) ..< total
            self.keys = concatenated([keys[0..., 0..., sink], keys[0..., 0..., window]], axis: 2)
            self.values = concatenated(
                [values[0..., 0..., sink], values[0..., 0..., window]], axis: 2)
            positions = Array(positions[sink]) + Array(positions[window])
        }
    }

    /// Which spans of `count` new positions the cache would keep: all of
    /// them, or the sink and the tail when they exceed the capacity.
    func spansToAppend(count: Int) -> [Range<Int>] {
        guard count > 0 else { return [] }
        let capacity = sinkSize + windowSize
        if length == 0 {
            if count <= capacity { return [0 ..< count] }
            var spans: [Range<Int>] = []
            let sinkEnd = min(sinkSize, count)
            if sinkEnd > 0 { spans.append(0 ..< sinkEnd) }
            let tailStart = max(sinkEnd, count - windowSize)
            if tailStart < count { spans.append(tailStart ..< count) }
            return spans
        }
        return count <= windowSize ? [0 ..< count] : [(count - windowSize) ..< count]
    }

    public override func copy() -> any KVCache {
        let new = DFlashContextCache(sinkSize: sinkSize, windowSize: windowSize)
        new.keys = keys
        new.values = values
        new.positions = positions
        new.offset = offset
        return new
    }
}

// MARK: - Modules

final class DFlashAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let window: Int?
    let isCausal: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let rope: RoPE

    init(_ config: DFlashDraftConfiguration, layer: Int) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        window = config.windowSize(layer: layer)
        isCausal = config.isCausal ?? (window != nil)
        let dim = config.hiddenSize
        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(heads * headDim, dim, bias: config.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
        super.init()
    }

    /// Keys and values `[B, kvHeads, n, D]` for context features at absolute
    /// positions starting at `start`.
    func contextKeysValues(_ context: MLXArray, start: Int) -> (MLXArray, MLXArray) {
        let (B, n) = (context.dim(0), context.dim(1))
        var k = kNorm(kProj(context).reshaped(B, n, kvHeads, headDim)).transposed(0, 2, 1, 3)
        k = rope(k, offset: start)
        let v = vProj(context).reshaped(B, n, kvHeads, headDim).transposed(0, 2, 1, 3)
        return (k, v)
    }

    /// Attention for a block of `L` positions starting at absolute
    /// `queryOffset` over the cached context plus the block itself.
    func callAsFunction(_ x: MLXArray, cache: DFlashContextCache?, queryOffset: Int) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
        q = rope(q, offset: queryOffset)
        var k = kNorm(kProj(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        k = rope(k, offset: queryOffset)
        let v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        var keys = k
        var values = v
        var keyPositions = (0 ..< L).map { Int32(queryOffset + $0) }
        if let cache, let cachedKeys = cache.keys, let cachedValues = cache.values {
            keys = concatenated([cachedKeys, k], axis: 2)
            values = concatenated([cachedValues, v], axis: 2)
            keyPositions = cache.positions + keyPositions
        }

        // Context entries within the window are visible; block entries are
        // all visible to each other (non-causal), or causally when configured.
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if keyPositions.count == L, !isCausal {
            mask = .none
        } else {
            let queries = MLXArray((0 ..< L).map { Int32(queryOffset + $0) }).reshaped(-1, 1)
            let keysArray = MLXArray(keyPositions).reshaped(1, -1)
            var context = keysArray .< Int32(queryOffset)
            if let window {
                context = context .&& ((queries - keysArray) .< Int32(window))
            }
            var block = keysArray .>= Int32(queryOffset)
            if isCausal {
                block = block .&& (keysArray .<= queries)
            }
            mask = .array(context .|| block)
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, heads * headDim))
    }
}

final class DFlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: DFlashDraftConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

/// Two-tap causal convolution over the block, per group of channels, with a
/// per-position dynamic kernel added to a learned base kernel (DFlash 2).
final class DFlashGroupedDynamicConv: Module {
    let groupSize: Int
    let kernelSize: Int
    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.groupSize = groupSize
        self.kernelSize = kernelSize
        let groups = hiddenSize / groupSize
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    private func convolve(_ hidden: MLXArray, dynamic: MLXArray, base: MLXArray) -> MLXArray {
        let (B, L, H) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let groups = H / groupSize
        let blocks = hidden.reshaped(B, L, groups, groupSize)
        let dynamic5 = dynamic.reshaped(B, L, kernelSize, groups, 1)
        var output = MLXArray.zeros(blocks.shape, dtype: blocks.dtype)
        for tap in 0 ..< kernelSize {
            let shifted: MLXArray
            if tap == 0 {
                shifted = blocks
            } else if tap >= L {
                continue
            } else {
                let pad = MLXArray.zeros([B, tap, groups, groupSize], dtype: blocks.dtype)
                shifted = concatenated([pad, blocks[0..., ..<(L - tap)]], axis: 1)
            }
            let kernel = base[tap].reshaped(1, 1, groups, groupSize).asType(hidden.dtype)
            output = output + (kernel + dynamic5[0..., 0..., tap]) * shifted
        }
        return output.reshaped(B, L, H)
    }

    /// Convolve `hidden` with the first kernel; returns it and the dynamic
    /// part of the second kernel, for `finish`.
    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let (B, L, H) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let groups = H / groupSize
        let dynamic = kernelProjection(hidden).reshaped(B, L, 2, kernelSize, groups)
        let first = convolve(hidden, dynamic: dynamic[0..., 0..., 0], base: baseKernel[0])
        return (first, dynamic[0..., 0..., 1])
    }

    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        convolve(hidden, dynamic: dynamic, base: baseKernel[1])
    }
}

final class DFlashDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DFlashAttention
    @ModuleInfo(key: "mlp") var mlp: DFlashMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlashGroupedDynamicConv?
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlashGroupedDynamicConv?

    init(_ config: DFlashDraftConfiguration, layer: Int) {
        _attention.wrappedValue = DFlashAttention(config, layer: layer)
        _mlp.wrappedValue = DFlashMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.hasConvolutions, let kernel = config.dflash.convKernelSize,
            let group = config.dflash.convGroupSize
        {
            _attentionConv.wrappedValue = DFlashGroupedDynamicConv(
                hiddenSize: config.hiddenSize, kernelSize: kernel, groupSize: group)
            _mlpConv.wrappedValue = DFlashGroupedDynamicConv(
                hiddenSize: config.hiddenSize, kernelSize: kernel, groupSize: group)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: DFlashContextCache?, queryOffset: Int) -> MLXArray {
        var h = x
        if let attentionConv, let mlpConv {
            let (normed, dynamic) = attentionConv.prepare(inputLayerNorm(h))
            h =
                h
                + attentionConv.finish(
                    attention(normed, cache: cache, queryOffset: queryOffset), dynamic: dynamic)
            let (normed2, dynamic2) = mlpConv.prepare(postAttentionLayerNorm(h))
            return h + mlpConv.finish(mlp(normed2), dynamic: dynamic2)
        }
        h = h + attention(inputLayerNorm(h), cache: cache, queryOffset: queryOffset)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// DFlash 2's path selector: scores each position's top-k candidates by
/// their logit plus a low-rank bilinear affinity with the chosen
/// predecessor, and walks the block greedily.
final class DFlashCandidateSelector: Module {
    let topK: Int
    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    init(_ config: DFlashDraftConfiguration, rank: Int, topK: Int) {
        self.topK = topK
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: rank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: rank)
        _hiddenProjection.wrappedValue = Linear(config.hiddenSize, rank, bias: false)
        super.init()
    }

    /// `hidden` and `logits` are `[B, P, …]` for the P drafted positions;
    /// `anchor` `[B]` is the token before the block. Returns `[B, P]` tokens.
    func select(hidden: MLXArray, logits: MLXArray, anchor: MLXArray) -> MLXArray {
        let vocabulary = logits.dim(-1)
        let k = min(topK, vocabulary)
        let candidates = argPartition(logits, kth: vocabulary - k, axis: -1)[
            .ellipsis, (vocabulary - k)...]
        let unary = takeAlong(logits, candidates, axis: -1).asType(.float32)
        let projected = hiddenProjection(hidden)
        var predecessor = anchor.reshaped(-1)
        var path: [MLXArray] = []
        for position in 0 ..< hidden.dim(1) {
            let slot = candidates[0..., position]
            let edges =
                (predecessorCodebook(predecessor)[0..., .newAxis, 0...]
                * projected[0..., position, .newAxis, 0...]
                * successorCodebook(slot)).sum(axis: -1).asType(.float32)
            let scores = unary[0..., position] + edges
            let selected = argMax(scores, axis: -1)
            predecessor = takeAlong(slot, selected[0..., .newAxis], axis: -1)[0..., 0]
            path.append(predecessor)
        }
        return stacked(path, axis: 1)
    }
}

// MARK: - Draft model

public final class DFlashDraftModel: Module, StatefulMTPDrafterModel {
    public let configuration: DFlashDraftConfiguration
    public var blockSize: Int { configuration.dflash.blockSize }
    public var maximumBlockSize: Int? { configuration.dflash.blockSize }
    public let requiresSharedTargetKV = false
    public let requiresPromptPrefill = true
    public let requiresGreedySampling = true
    public var targetTapLayers: [Int]? { configuration.dflash.targetLayerIds }
    public var consumesFullContextHidden: Bool { true }

    /// Block policy: the verify pass costs about the same up to 4 tokens
    /// and grows past that (on an M5 Max with Qwen3.8-27B 4-bit: 41 ms at
    /// 4, 80 ms at 8), so never go below 4 and widen when recent rounds
    /// accepted enough to pay for the wider pass.
    /// Exponential moving average of accepted drafted tokens per round.
    private var acceptedAverage: Double?

    public func nextBlockSize(afterAccepting accepted: Int, current: Int, maximum: Int) -> Int {
        let average = acceptedAverage.map { 0.7 * $0 + 0.3 * Double(accepted) } ?? Double(accepted)
        acceptedAverage = average
        return max(4, min(maximum, Int(average.rounded()) + 3))
    }
    /// Context kept per layer: the first `sinkSize` positions and the last
    /// `windowSize` (the reference defaults).
    public var sinkSize = 64
    public var windowSize = 1024

    @ModuleInfo(key: "layers") var layers: [DFlashDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var selector: DFlashCandidateSelector?

    public init(_ configuration: DFlashDraftConfiguration) {
        self.configuration = configuration
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map {
            DFlashDecoderLayer(configuration, layer: $0)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _fc.wrappedValue = Linear(
            configuration.dflash.targetLayerIds.count * configuration.hiddenSize,
            configuration.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        if configuration.hasSelector, let rank = configuration.dflash.selectorRank,
            let topK = configuration.dflash.selectorTopK
        {
            _selector.wrappedValue = DFlashCandidateSelector(configuration, rank: rank, topK: topK)
        }
        super.init()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        for name in ["predecessor_codebook", "successor_codebook"] {
            let key = "candidate_selector.\(name)"
            if let value = sanitized.removeValue(forKey: key) {
                sanitized[key + ".weight"] = value
            }
        }
        return sanitized
    }

    /// The target's tapped hidden states `[B, n, taps × H]` as draft context `[B, n, H]`.
    func projectContext(_ targetHidden: MLXArray) -> MLXArray {
        hiddenNorm(fc(targetHidden))
    }

    /// Add `count` new context positions starting at absolute `start` to
    /// every layer's cache, keeping only what the sink and window retain.
    func appendContext(_ targetHidden: MLXArray, start: Int, caches: [KVCache]) {
        let count = targetHidden.dim(1)
        guard count > 0, let first = caches.first as? DFlashContextCache else { return }
        let spans = first.spansToAppend(count: count)
        for span in spans {
            let context = projectContext(targetHidden[0..., span, 0...])
            for (layer, entry) in zip(layers, caches) {
                guard let cache = entry as? DFlashContextCache else { continue }
                let (k, v) = layer.attention.contextKeysValues(
                    context, start: start + span.lowerBound)
                cache.append(keys: k, values: v, start: start + span.lowerBound)
            }
        }
        eval(caches.flatMap { $0.innerState() })
    }

    /// Draft `blockSize - 1` tokens after `anchor` `[B, 1]`.
    func draft(
        anchor: MLXArray, target: any DFlashTargetModel, caches: [KVCache], queryOffset: Int,
        blockSize: Int
    ) -> MLXArray {
        let B = anchor.dim(0)
        let masks = MLXArray.full(
            [B, blockSize - 1], values: MLXArray(Int32(configuration.dflash.maskTokenId)))
        let tokens = concatenated([anchor.asType(.int32), masks], axis: 1)
        var h = target.dflashTokenEmbedding(tokens)
        let scale = target.dflashEmbeddingScale * (configuration.dflash.inputEmbeddingScale ?? 1)
        if scale != 1 { h = h * scale }
        for (layer, entry) in zip(layers, caches) {
            h = layer(h, cache: entry as? DFlashContextCache, queryOffset: queryOffset)
        }
        let hidden = norm(h)[0..., 1..., 0...]
        let logits = target.dflashLogits(hidden)
        if let selector {
            return selector.select(hidden: hidden, logits: logits, anchor: anchor)
        }
        return argMax(logits, axis: -1)
    }

    // MARK: Testing hooks

    /// The projected draft context for target taps `[B, n, taps × H]`.
    @_spi(Testing) public func debugProjectContext(_ targetHidden: MLXArray) -> MLXArray {
        projectContext(targetHidden)
    }

    /// One block forward from fresh caches: `context` `[B, n, H]` at
    /// positions `0 ..< n`, `noiseEmbedding` `[B, block, H]` at positions
    /// `n ..< n + block`. Returns the normalised block hidden states.
    @_spi(Testing) public func debugForward(context: MLXArray, noiseEmbedding: MLXArray) -> MLXArray
    {
        let caches = layers.map { _ in
            DFlashContextCache(sinkSize: sinkSize, windowSize: windowSize)
        }
        for (layer, cache) in zip(layers, caches) {
            let (k, v) = layer.attention.contextKeysValues(context, start: 0)
            cache.append(keys: k, values: v, start: 0)
        }
        var h = noiseEmbedding
        for (layer, cache) in zip(layers, caches) {
            h = layer(h, cache: cache, queryOffset: context.dim(1))
        }
        return norm(h)
    }

    /// Selector choice for debugging: `hidden` `[B, P, H]`, `logits` `[B, P, V]`, `anchor` `[B]`.
    @_spi(Testing) public func debugSelect(hidden: MLXArray, logits: MLXArray, anchor: MLXArray)
        -> MLXArray?
    {
        selector?.select(hidden: hidden, logits: logits, anchor: anchor)
    }

    // MARK: StatefulMTPDrafterModel

    public func makeState(parameters: GenerateParameters?) -> MTPDrafterState {
        MTPDrafterState(
            cache: layers.map { _ in DFlashContextCache(sinkSize: sinkSize, windowSize: windowSize)
            })
    }

    public func prepareDrafterState(
        target: any LanguageModel, promptTokens: MLXArray, targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas: MLXArray?, state: inout MTPDrafterState, sampler: any LogitSampler
    ) {
        // Everything the target has seen so far (a cached prefix's hidden
        // states arrive concatenated before the prompt's).
        appendContext(targetHidden, start: 0, caches: state.cache)
        state.nextPosition = targetHidden.dim(1)
        state.proposalAppended = 0
        acceptedAverage = nil
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?, queryOffset: Int, blockSize: Int, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray {
        guard let target = target as? any DFlashTargetModel else {
            fatalError("DFlashDraftModel needs a DFlashTargetModel target, got \(type(of: target))")
        }
        let anchor = lastToken.ndim == 1 ? lastToken.reshaped(lastToken.dim(0), 1) : lastToken
        let proposed = draft(
            anchor: anchor, target: target, caches: state.cache, queryOffset: state.nextPosition,
            blockSize: blockSize)
        state.proposalAppended = blockSize - 1
        return proposed
    }

    public func draftBlock(
        target: any LanguageModel, lastToken: MLXArray, lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?, queryOffset: Int, blockSize: Int, sampler: any LogitSampler
    ) -> MLXArray {
        var state = makeState(parameters: nil)
        state.nextPosition = queryOffset
        return draftBlock(
            target: target, lastToken: lastToken, lastHidden: lastHidden, sharedKV: sharedKV,
            positionDeltas: positionDeltas, queryOffset: queryOffset, blockSize: blockSize,
            state: &state, sampler: sampler)
    }

    public func commitDrafterState(
        target: any LanguageModel, targetHidden: MLXArray, draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray, positionDeltas: MLXArray?, state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) {
        // The verify pass covered the bonus token and the drafted ones; the
        // bonus and the accepted drafts are now context.
        let committed = min(acceptedCount + 1, targetHidden.dim(1))
        appendContext(
            targetHidden[0..., ..<committed, 0...], start: state.nextPosition, caches: state.cache)
        state.nextPosition += committed
        state.proposalAppended = 0
    }
}

/// Registers DFlash drafts with ``MTPDrafterTypeRegistry`` under the base
/// model types their checkpoints declare (the presence of `dflash_config`
/// tells them apart from an MTP head of the same type).
public enum DFlashDrafterRegistration {
    public static func register(
        modelTypes: [String] = ["qwen3", "qwen3_5", "gemma4", "gemma4_text", "llama"]
    ) async {
        for type in modelTypes {
            await MTPDrafterTypeRegistry.shared.registerModelType(
                type,
                matches: { DFlashDraftConfiguration.describes($0) },
                creator: { data in
                    DFlashDraftModel(
                        try JSONDecoder.json5().decode(DFlashDraftConfiguration.self, from: data))
                })
        }
    }
}
