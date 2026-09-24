// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Speculative decoding for a batch of rows that share a prompt and then
/// diverge: every round drafts a block per row, verifies all rows in one
/// target pass, keeps a different number of tokens per row, and rewinds
/// each row by its own amount ("ragged" batching). The rows sample
/// independently, so at a temperature they are alternative continuations
/// of the same prompt for the price of one verify pass with `rows` times
/// the rows.
///
/// Requires a ``RaggedSpeculativeTarget`` (per-row positions, per-row
/// rewind over ``RaggedKVCache`` and ``MambaCache``) and a
/// ``DFlashDraftModel``.
public final class BatchSpeculativeGenerator {

    public struct Options: Sendable {
        /// Tokens per round (one anchor plus `blockSize - 1` drafts).
        public var blockSize = 8
        public var adaptiveBlock = true
        public var maxTokens = 400
        /// Time each stage of a round (adds synchronisation; profiling only).
        public var timing = false
        /// Keep a finished row's caches (trimmed to what it emitted) so
        /// `append(_:)` can give it more tokens later, instead of dropping
        /// the row for good.
        public var keepFinishedRows = false
        public init() {}
    }

    /// Tokens for one row as they are accepted.
    public struct Emission {
        public let row: Int
        public let tokens: [Int]
        public let finished: Bool
    }

    public struct Report {
        public var rounds = 0
        public var proposed = 0
        public var accepted = 0
        public var generated = 0
        public var blockTotal = 0
        public var averageBlock: Double { rounds > 0 ? Double(blockTotal) / Double(rounds) : 0 }
    }

    private let target: any RaggedSpeculativeTarget
    private let drafter: DFlashDraftModel
    private let sampler: any LogitSampler
    private let samples: Bool
    private let eosTokens: Set<Int>
    private let options: Options

    private var cache: [KVCache]
    private var draftCaches: [KVCache]
    private var state: LMOutput.State
    /// Original row index of every active row.
    private var rowIDs: [Int]
    private var positions: [Int]
    private var anchors: [Int]
    private var produced: [Int]
    private var blockSize: Int
    public private(set) var report = Report()
    /// Each row's first token, emitted with its first round.
    private var firstTokens: [Int]?
    /// Finished rows kept aside (`keepFinishedRows`), by original row.
    private var parked: [Int: (cache: [KVCache], draft: [KVCache], position: Int)] = [:]
    private var totalRows: Int
    /// Milliseconds per stage, summed over rounds (with `timing`).
    public private(set) var stageMilliseconds: [String: Double] = [:]
    private var stageClock = ContinuousClock.now

    private func lap(_ stage: String) {
        guard options.timing else { return }
        let now = ContinuousClock.now
        let d = now - stageClock
        stageMilliseconds[stage, default: 0] += Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
        stageClock = now
    }

    /// - Parameters:
    ///   - cache: the target's single-row cache with the whole prompt
    ///     processed (its last token included).
    ///   - promptState: the state that prompt forward returned, carrying the
    ///     prompt's tapped hidden states under ``mtpLastHiddenStatesKey``.
    ///   - lastLogits: `[1, V]` logits of the prompt's last position; each
    ///     row's first token is drawn from them.
    ///   - promptTokens: the prompt, for the drafter's preparation.
    public init(
        target: any RaggedSpeculativeTarget, drafter: DFlashDraftModel,
        cache: [KVCache], promptState: LMOutput.State, lastLogits: MLXArray, promptTokens: MLXArray,
        rows: Int, parameters: GenerateParameters, eosTokens: Set<Int>, options: Options = Options(),
        firstTokens: [Int]? = nil
    ) {
        precondition(rows >= 1)
        precondition(firstTokens == nil || firstTokens!.count == rows)
        let forcedFirst = firstTokens
        self.target = target
        self.drafter = drafter
        self.sampler = parameters.sampler()
        self.samples = parameters.temperature != 0
        self.eosTokens = eosTokens
        self.options = options
        self.blockSize = max(2, min(options.blockSize, drafter.maximumBlockSize ?? options.blockSize))

        let promptHidden = promptState[mtpLastHiddenStatesKey]!
        let promptLength = promptHidden.dim(1)

        // First token per row from the prompt's last logits (or as given,
        // e.g. the top candidates to force rows apart deterministically).
        let first: MLXArray
        if let forcedFirst {
            first = MLXArray(forcedFirst.map { Int32($0) })
        } else {
            let logits = repeated(lastLogits.reshaped(1, -1), count: rows, axis: 0)
            first = samples
                ? sampler.sample(distribution: sampler.distribution(logits: logits)!)
                : argMax(logits, axis: -1)
        }
        eval(first)
        anchors = first.asArray(Int.self)

        // The drafter's context over the prompt, then every row a copy.
        var drafterState = drafter.makeState(parameters: parameters)
        drafter.prepareDrafterState(
            target: target, promptTokens: promptTokens, targetHidden: promptHidden,
            firstBonus: first[0 ..< 1], positionDeltas: nil, state: &drafterState, sampler: sampler)
        draftCaches = drafter.expandedCaches(drafterState.cache, rows: rows)

        self.cache = target.expandCache(cache, rows: rows)
        var state = promptState
        state[mtpLastHiddenStatesKey] = nil
        self.state = state
        rowIDs = Array(0 ..< rows)
        positions = Array(repeating: promptLength, count: rows)
        produced = Array(repeating: 0, count: rows)
        totalRows = rows
        self.firstTokens = anchors
    }

    /// One prompt per row: each row's prefilled single-row cache, the state
    /// its prompt forward returned (with the prompt's tapped hidden states),
    /// its last-position logits and its prompt tokens.
    public struct RowPrompt {
        public var cache: [KVCache]
        public var promptState: LMOutput.State
        public var lastLogits: MLXArray
        public var promptTokens: MLXArray
        public init(cache: [KVCache], promptState: LMOutput.State, lastLogits: MLXArray, promptTokens: MLXArray) {
            self.cache = cache
            self.promptState = promptState
            self.lastLogits = lastLogits
            self.promptTokens = promptTokens
        }
    }

    /// Rows with different prompts ("experts"): every row is prefilled on
    /// its own and the rows are merged into one ragged batch.
    public init(
        target: any RaggedSpeculativeTarget, drafter: DFlashDraftModel, rows prompts: [RowPrompt],
        parameters: GenerateParameters, eosTokens: Set<Int>, options: Options = Options()
    ) {
        precondition(!prompts.isEmpty)
        self.target = target
        self.drafter = drafter
        self.sampler = parameters.sampler()
        self.samples = parameters.temperature != 0
        self.eosTokens = eosTokens
        self.options = options
        self.blockSize = max(2, min(options.blockSize, drafter.maximumBlockSize ?? options.blockSize))

        var firsts: [Int] = []
        var draftStates: [[KVCache]] = []
        for prompt in prompts {
            let hidden = prompt.promptState[mtpLastHiddenStatesKey]!
            let logits = prompt.lastLogits.reshaped(1, -1)
            let first = samples
                ? sampler.sample(distribution: sampler.distribution(logits: logits)!)
                : argMax(logits, axis: -1)
            eval(first)
            firsts.append(first.asArray(Int.self)[0])
            var drafterState = drafter.makeState(parameters: parameters)
            drafter.prepareDrafterState(
                target: target, promptTokens: prompt.promptTokens, targetHidden: hidden,
                firstBonus: first[0 ..< 1], positionDeltas: nil, state: &drafterState, sampler: sampler)
            draftStates.append(drafterState.cache)
        }
        anchors = firsts
        draftCaches = drafter.mergedCaches(draftStates)
        cache = target.mergeCaches(prompts.map(\.cache))
        // Rope deltas and other per-row state: rows prefilled alone carry
        // their own; text prompts have none, so the first row's state
        // stands for the batch.
        var state = prompts[0].promptState
        state[mtpLastHiddenStatesKey] = nil
        self.state = state
        rowIDs = Array(prompts.indices)
        positions = prompts.map { $0.promptState[mtpLastHiddenStatesKey]!.dim(1) }
        produced = Array(repeating: 0, count: prompts.count)
        totalRows = prompts.count
        firstTokens = firsts
    }

    public var isFinished: Bool { rowIDs.isEmpty }

    /// Give every row (active or parked) more prompt tokens — the next
    /// turn of a conversation — in one ragged prefill, and make every row
    /// active again with a fresh token budget. `suffixes` is indexed by
    /// original row and must be non-empty for every row.
    public func append(_ suffixes: [[Int]]) {
        precondition(suffixes.count == totalRows, "append needs one suffix per row")
        precondition(suffixes.allSatisfy { !$0.isEmpty }, "append needs a non-empty suffix for every row")
        // Gather every row as a single-row cache, in original order.
        var rowCaches = [[KVCache]?](repeating: nil, count: totalRows)
        var rowDrafts = [[KVCache]?](repeating: nil, count: totalRows)
        var rowPositions = [Int](repeating: 0, count: totalRows)
        for (r, id) in rowIDs.enumerated() {
            rowCaches[id] = target.extractRow(cache, row: r)
            rowDrafts[id] = drafter.extractRow(draftCaches, row: r)
            rowPositions[id] = positions[r]
        }
        for (id, entry) in parked {
            rowCaches[id] = entry.cache
            rowDrafts[id] = entry.draft
            rowPositions[id] = entry.position
        }
        precondition(rowCaches.allSatisfy { $0 != nil }, "append: a row is neither active nor parked")
        parked.removeAll()
        cache = target.mergeCaches(rowCaches.map { $0! })
        draftCaches = drafter.mergedCaches(rowDrafts.map { $0! })
        rowIDs = Array(0 ..< totalRows)
        positions = rowPositions
        let B = totalRows

        // One forward over the padded suffixes at each row's positions;
        // the padding is dropped again like rejected drafts.
        let longest = suffixes.map(\.count).max()!
        let padded = suffixes.map { $0 + Array(repeating: $0.last!, count: longest - $0.count) }
        let tokens = MLXArray(padded.flatMap { $0.map { Int32($0) } }).reshaped(B, longest)
        let rowPositionIds = MLXArray(positions.map { Int32($0) }).reshaped(B, 1)
            + MLXArray(Int32(0) ..< Int32(longest)).reshaped(1, longest)
        var prefillState = state
        prefillState[mtpEmitFlagKey] = true
        prefillState[mtpTapLayersKey] = drafter.targetTapLayers
        prefillState[mtpSpeculativeTapeKey] = true
        prefillState[mtpPositionIdsKey] = rowPositionIds
        let result = target(LMInput.Text(tokens: tokens), cache: cache, state: prefillState)
        guard let taps = result.state?[mtpLastHiddenStatesKey] else {
            fatalError("BatchSpeculativeGenerator: the target did not emit tapped hidden states")
        }
        var nextState = result.state ?? state
        nextState[mtpLastHiddenStatesKey] = nil
        nextState[mtpEmitFlagKey] = nil
        nextState[mtpSpeculativeTapeKey] = nil
        nextState[mtpPositionIdsKey] = nil
        state = nextState

        let keep = suffixes.map(\.count)
        target.rewindSpeculativeCache(cache, keepPerRow: keep)
        drafter.commitRagged(taps, keep: keep, starts: positions, caches: draftCaches)

        // Each row's next token comes from its own last real position.
        let lastIndex = MLXArray(keep.map { Int32($0 - 1) }).reshaped(B, 1, 1)
        let lastLogits = takeAlong(result.logits, broadcast(lastIndex, to: [B, 1, result.logits.dim(-1)]), axis: 1)
            .squeezed(axis: 1)
        let first = samples
            ? sampler.sample(distribution: sampler.distribution(logits: lastLogits)!)
            : argMax(lastLogits, axis: -1)
        eval(first, cache.flatMap { $0.innerState() })
        anchors = first.asArray(Int.self)
        firstTokens = anchors
        positions = zip(positions, keep).map { $0 + $1 }
        produced = Array(repeating: 0, count: B)
        report.rounds += 0
    }

    /// One round for every active row. Returns what each row produced.
    public func round() -> [Emission] {
        guard !rowIDs.isEmpty else { return [] }
        let B = rowIDs.count
        let drafts = blockSize - 1
        stageClock = ContinuousClock.now

        // Draft.
        let anchor = MLXArray(anchors.map { Int32($0) }).reshaped(B, 1)
        let proposals = drafter.draftRagged(
            anchor: anchor, target: target, caches: draftCaches, queryOffsets: positions, blockSize: blockSize)
        if options.timing { eval(proposals) }
        lap("draft")

        // Verify all rows in one pass at their own positions.
        let verifyTokens = concatenated([anchor, proposals.asType(.int32)], axis: 1)
        let rowPositions = MLXArray(positions.map { Int32($0) }).reshaped(B, 1)
            + MLXArray(Int32(0) ..< Int32(blockSize)).reshaped(1, blockSize)
        var verifyState = state
        verifyState[mtpEmitFlagKey] = true
        verifyState[mtpTapLayersKey] = drafter.targetTapLayers
        verifyState[mtpSpeculativeTapeKey] = true
        verifyState[mtpPositionIdsKey] = rowPositions
        let result = target(LMInput.Text(tokens: verifyTokens), cache: cache, state: verifyState)
        let logits = result.logits
        if options.timing { eval(logits) }
        lap("verify")
        guard let taps = result.state?[mtpLastHiddenStatesKey] else {
            fatalError("BatchSpeculativeGenerator: the target did not emit tapped hidden states")
        }
        var nextState = result.state ?? state
        nextState[mtpLastHiddenStatesKey] = nil
        nextState[mtpEmitFlagKey] = nil
        nextState[mtpSpeculativeTapeKey] = nil
        nextState[mtpPositionIdsKey] = nil
        state = nextState

        // Accept per row.
        eval(proposals)
        let proposed = proposals.asArray(Int.self)  // [B * drafts]
        var accepted = [Int](repeating: 0, count: B)
        let finals: [Int]
        if samples, let probs = sampler.distribution(logits: logits) {
            let pDraft = takeAlong(probs[0..., ..<drafts, 0...], proposals[.ellipsis, .newAxis], axis: -1).squeezed(axis: -1)
            let keep = MLXRandom.uniform(0 ..< 1, [B, drafts]) .<= pDraft
            eval(keep)
            let keeps = keep.asArray(Bool.self)
            for r in 0 ..< B {
                while accepted[r] < drafts, keeps[r * drafts + accepted[r]] { accepted[r] += 1 }
            }
            // Row r draws from probs[r, accepted[r]]: the residual at the
            // rejection (draft token removed), or the bonus row.
            let index = MLXArray(accepted.map { Int32($0) }).reshaped(B, 1, 1)
            var final = takeAlong(probs, broadcast(index, to: [B, 1, probs.dim(-1)]), axis: 1).squeezed(axis: 1)
            let rejectedToken = MLXArray((0 ..< B).map { r -> Int32 in
                accepted[r] < drafts ? Int32(proposed[r * drafts + accepted[r]]) : -1
            }).reshaped(B, 1)
            let vocabulary = MLXArray.arange(probs.dim(-1)).reshaped(1, -1)
            final = MLX.where(vocabulary .== rejectedToken, MLXArray(Float(0)), final)
            final = final / maximum(final.sum(axis: -1, keepDims: true), MLXArray(Float(1e-30)))
            let token = sampler.sample(distribution: final)
            eval(token)
            finals = token.asArray(Int.self)
        } else {
            let targetTokens = argMax(logits, axis: -1)
            eval(targetTokens)
            let targets = targetTokens.asArray(Int.self)  // [B * blockSize]
            var f = [Int](repeating: 0, count: B)
            for r in 0 ..< B {
                while accepted[r] < drafts, targets[r * blockSize + accepted[r]] == proposed[r * drafts + accepted[r]] {
                    accepted[r] += 1
                }
                f[r] = targets[r * blockSize + accepted[r]]
            }
            finals = f
        }

        lap("accept")
        // What each row emits, and how much of the pass it keeps: a row
        // that finishes (end token, or its budget) keeps only what it
        // emitted, so its cache ends exactly where its text does.
        var emissions: [Emission] = []
        var survivors: [Int] = []
        var keep = accepted.map { $0 + 1 }
        let first = firstTokens
        firstTokens = nil
        var finishedRows: [Int] = []
        for r in 0 ..< B {
            let lead = first.map { [$0[r]] } ?? []
            let fresh = (0 ..< accepted[r]).map { proposed[r * drafts + $0] } + [finals[r]]
            var tokens = lead + fresh
            var finished = false
            if let end = tokens.firstIndex(where: { eosTokens.contains($0) }) {
                tokens = Array(tokens[..<end])
                finished = true
            }
            let room = options.maxTokens - produced[r]
            if tokens.count >= room {
                tokens = Array(tokens.prefix(max(0, room)))
                finished = true
            }
            if finished {
                // Positions of this pass the row keeps: its emitted share
                // of `fresh` (the lead token was already in the cache).
                keep[r] = max(0, tokens.count - lead.count)
                finishedRows.append(r)
            }
            produced[r] += tokens.count
            emissions.append(Emission(row: rowIDs[r], tokens: tokens, finished: finished))
            if !finished { survivors.append(r) }
        }
        target.rewindSpeculativeCache(cache, keepPerRow: keep)
        if options.timing { eval(cache.flatMap { $0.innerState() }) }
        lap("rewind")
        drafter.commitRagged(taps, keep: keep, starts: positions, caches: draftCaches)
        lap("commit")
        for r in 0 ..< B {
            positions[r] += keep[r]
            anchors[r] = finals[r]
        }
        if options.keepFinishedRows {
            for r in finishedRows {
                parked[rowIDs[r]] = (target.extractRow(cache, row: r), drafter.extractRow(draftCaches, row: r), positions[r])
            }
        }
        report.rounds += 1
        report.blockTotal += blockSize
        report.proposed += drafts * B
        report.accepted += accepted.reduce(0, +)
        report.generated += emissions.reduce(0) { $0 + $1.tokens.count }

        if survivors.count < B {
            if !survivors.isEmpty {
                target.filterCache(cache, rows: survivors)
                drafter.filterCaches(draftCaches, rows: survivors)
            }
            rowIDs = survivors.map { rowIDs[$0] }
            positions = survivors.map { positions[$0] }
            anchors = survivors.map { anchors[$0] }
            produced = survivors.map { produced[$0] }
        }
        lap("emit")
        if options.adaptiveBlock, !rowIDs.isEmpty {
            let mean = Int((Double(accepted.reduce(0, +)) / Double(B)).rounded())
            let maximum = drafter.maximumBlockSize ?? options.blockSize
            blockSize = max(2, min(maximum, drafter.nextBlockSize(afterAccepting: mean, current: blockSize, maximum: maximum)))
        }
        return emissions
    }
}
