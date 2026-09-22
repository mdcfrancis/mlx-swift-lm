// Copyright © 2026 Apple Inc.
//
// The verify kernels are ported from dflash-mlx (Copyright 2026 bstnxbt,
// Apache License 2.0), `verify_qmm.py` / `verify_linear.py`.

import Foundation
import MLX
import MLXNN

// Speculative decoding verifies a small block of tokens in one target pass:
// a quantized matmul with 2-16 rows, which MLX serves with its
// matrix-vector kernel (row-looped, compute-bound at that width) or its
// tiled kernels (sized for 32+ rows). Neither is near the weight-bandwidth
// floor at these widths: on an M5 Max, Qwen3.8-27B 4-bit verifies 8 tokens
// in 80 ms against 35 ms for one. These kernels are written for exactly
// 4 or 16 rows — a 16-row tensor-unit (NAX) kernel for Apple GPUs that
// have one, and a 4-row split-K kernel for any Apple GPU — and the
// iterator pads its verify pass to those widths (``SpeculativeOptions``).

/// Which verify kernels to install and where.
public struct VerifyKernelOptions: Sendable {
    /// Row counts to serve with the kernels; other widths use MLX's kernels.
    public var rows: Set<Int> = [4, 16]
    /// Skip linears wider than this (the output head, by default, is kept).
    public var maximumOutputSize = 1 << 20

    public init() {}
}

public enum SpeculativeVerifyKernels {
    /// The 4-row kernel runs on any Apple GPU; the 16-row kernel needs the
    /// tensor units of the M5 generation and macOS 26.2 or later.
    public static var isAvailable: Bool { MLX.GPU.deviceInfo().architecture.lowercased().hasPrefix("applegpu") }

    public static var supportsTensorUnitKernel: Bool {
        guard MLX.GPU.deviceInfo().architecture.lowercased().hasPrefix("applegpu_g17") else { return false }
        let version = ProcessInfo.processInfo.operatingSystemVersion
        return version.majorVersion > 26 || (version.majorVersion == 26 && version.minorVersion >= 2)
    }

    /// Replace every eligible ``QuantizedLinear`` under `model` with a
    /// ``VerifyQuantizedLinear``. Returns how many were replaced.
    @discardableResult
    public static func install(in model: Module, options: VerifyKernelOptions = .init()) -> Int {
        var updates: [(String, Module)] = []
        for (path, module) in model.leafModules().flattened() {
            guard !(module is VerifyQuantizedLinear), let linear = module as? QuantizedLinear,
                VerifyQuantizedLinear.isEligible(linear, options: options)
            else { continue }
            updates.append((path, VerifyQuantizedLinear(linear, options: options)))
        }
        guard !updates.isEmpty else { return 0 }
        model.update(modules: ModuleChildren.unflattened(updates))
        return updates.count
    }

    /// Run each distinct kernel once so shader compilation does not land
    /// inside the first speculative round.
    public static func prewarm(in model: Module, dtype: DType = .bfloat16) {
        var seen: Set<String> = []
        var outputs: [MLXArray] = []
        for (_, module) in model.leafModules().flattened() {
            guard let linear = module as? VerifyQuantizedLinear else { continue }
            let key = "\(linear.inputSize)/\(linear.outputSize)/\(linear.bits)/\(linear.groupSize)"
            guard !seen.contains(key) else { continue }
            seen.insert(key)
            for rows in linear.servedRows {
                outputs.append(linear(MLXArray.zeros([1, rows, linear.inputSize], dtype: dtype)))
            }
        }
        eval(outputs)
    }
}

/// A ``QuantizedLinear`` that serves 4- and 16-row inputs with the verify
/// kernels and everything else with MLX's quantized matmul.
open class VerifyQuantizedLinear: QuantizedLinear {
    let inputSize: Int
    let outputSize: Int
    let servedRows: [Int]
    private let kParts: Int

    static func isEligible(_ linear: QuantizedLinear, options: VerifyKernelOptions) -> Bool {
        guard linear.bits == 4, linear.mode == .affine, [32, 64, 128].contains(linear.groupSize),
            linear.biases != nil
        else { return false }
        let (N, K) = linear.shape
        return N < options.maximumOutputSize && K % 32 == 0 && N % 4 == 0
    }

    init(_ linear: QuantizedLinear, options: VerifyKernelOptions) {
        let (N, K) = linear.shape
        inputSize = K
        outputSize = N
        var rows: [Int] = []
        if options.rows.contains(4) { rows.append(4) }
        if options.rows.contains(16), SpeculativeVerifyKernels.supportsTensorUnitKernel, K % 256 == 0, N % 32 == 0 {
            rows.append(16)
        }
        servedRows = rows
        kParts = N >= 4096 ? 2 : 4
        super.init(
            weight: linear.weight, bias: linear.bias, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits, mode: linear.mode)
        freeze()
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rows = x.size / inputSize
        guard servedRows.contains(rows), x.dtype == .bfloat16 || x.dtype == .float16, let biases else {
            return super.callAsFunction(x)
        }
        let x2 = x.reshaped(rows, inputSize)
        var y: MLXArray
        if rows == 16 {
            let kernel = VerifyKernelCache.shared.tensorUnitKernel(k: inputSize, groupSize: groupSize, dtype: x.dtype)
            y = kernel(
                [x2, weight, scales, biases, MLXArray(Int32(outputSize))],
                template: [("T", x.dtype), ("KCONST", inputSize)],
                grid: (256, outputSize / 32, 1), threadGroup: (256, 1, 1),
                outputShapes: [[16, outputSize]], outputDTypes: [x.dtype])[0]
        } else {
            let kernel = VerifyKernelCache.shared.splitKKernel(groupSize: groupSize, dtype: x.dtype, kParts: kParts)
            y = kernel(
                [x2, weight, scales, biases, MLXArray(Int32(inputSize)), MLXArray(Int32(outputSize))],
                template: [("T", x.dtype)],
                grid: (32 * kParts, outputSize / 4, 1), threadGroup: (32 * kParts, 1, 1),
                outputShapes: [[4, outputSize]], outputDTypes: [x.dtype])[0]
        }
        y = y.reshaped(Array(x.shape.dropLast()) + [outputSize])
        if let bias { y = y + bias }
        return y
    }
}

// MARK: - Kernels

final class VerifyKernelCache: @unchecked Sendable {
    static let shared = VerifyKernelCache()
    private var kernels: [String: MLXFast.MLXFastKernel] = [:]
    private let lock = NSLock()

    private func kernel(_ key: String, make: () -> MLXFast.MLXFastKernel) -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let existing = kernels[key] { return existing }
        let made = make()
        kernels[key] = made
        return made
    }

    private static func tag(_ dtype: DType) -> String { dtype == .float16 ? "fp16" : "bf16" }

    /// 16 rows: eight simdgroups split K, each dequantizes a 16×32 weight
    /// tile into threadgroup memory and multiplies it on the tensor units;
    /// the partial sums are reduced in a fixed order.
    func tensorUnitKernel(k: Int, groupSize: Int, dtype: DType) -> MLXFast.MLXFastKernel {
        kernel("m16_nax_k\(k)_gs\(groupSize)_\(Self.tag(dtype))") {
            MLXFast.metalKernel(
                name: "verify_m16_nax_ktmpl_k\(k)_gs\(groupSize)_\(Self.tag(dtype))",
                inputNames: ["x", "w_q", "scales", "biases", "N_size"],
                outputNames: ["y"],
                source: """
                    using namespace metal;
                    using namespace mpp::tensor_ops;

                    constexpr int BM = 16;
                    constexpr int BN = 32;
                    constexpr int BK = 16;
                    constexpr int NSG = 8;
                    constexpr int GS = \(groupSize);
                    constexpr int K = KCONST;
                    constexpr int K_by_8 = K / 8;
                    constexpr int K_by_gs = K / GS;
                    constexpr int K_chunk = K / NSG;

                    uint tid = thread_position_in_threadgroup.x;
                    uint sg_id = simdgroup_index_in_threadgroup;
                    uint lane = thread_index_in_simdgroup;
                    uint tg_n = threadgroup_position_in_grid.y;
                    int N = int(N_size);
                    int n0 = int(tg_n) * BN;
                    int k_begin = int(sg_id) * K_chunk;
                    int k_end = k_begin + K_chunk;

                    threadgroup T B_tile[NSG][BK * BN];
                    threadgroup float partial[NSG][BM * BN];

                    constexpr auto desc = matmul2d_descriptor(
                        16, 32, 16, false, false, false,
                        matmul2d_descriptor::mode::multiply_accumulate);
                    matmul2d<desc, metal::execution_simdgroup> op;

                    tensor<device T, dextents<int, 2>, tensor_inline> A(
                        (device T*)x, dextents<int, 2>{K, BM}, array<int, 2>{1, K});
                    tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
                        B_tile[sg_id], dextents<int, 2>{BN, BK}, array<int, 2>{1, BN});
                    tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
                        partial[sg_id], dextents<int, 2>{BN, BM}, array<int, 2>{1, BN});

                    auto ct_c = op.template get_destination_cooperative_tensor<
                        tensor<device T, extents<int, 16, 16>, tensor_inline>,
                        tensor<threadgroup T, extents<int, 32, 16>, tensor_inline>,
                        float>();
                    _Pragma("unroll")
                    for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {
                        ct_c[i] = 0.0f;
                    }

                    int n_global = n0 + int(lane);
                    for (int k0 = k_begin; k0 < k_end; k0 += BK) {
                        uint32_t p0 = w_q[n_global * K_by_8 + ((k0 + 0) >> 3)];
                        uint32_t p1 = w_q[n_global * K_by_8 + ((k0 + 8) >> 3)];
                        float s0 = float(scales[n_global * K_by_gs + ((k0 + 0) / GS)]);
                        float s1 = float(scales[n_global * K_by_gs + ((k0 + 8) / GS)]);
                        float b0 = float(biases[n_global * K_by_gs + ((k0 + 0) / GS)]);
                        float b1 = float(biases[n_global * K_by_gs + ((k0 + 8) / GS)]);

                        _Pragma("unroll")
                        for (int ki = 0; ki < 8; ++ki) {
                            uint32_t nib = (p0 >> (ki * 4)) & 0xFu;
                            B_tile[sg_id][ki * BN + int(lane)] = T(float(nib) * s0 + b0);
                        }
                        _Pragma("unroll")
                        for (int ki = 0; ki < 8; ++ki) {
                            uint32_t nib = (p1 >> (ki * 4)) & 0xFu;
                            B_tile[sg_id][(8 + ki) * BN + int(lane)] = T(float(nib) * s1 + b1);
                        }
                        simdgroup_barrier(mem_flags::mem_threadgroup);

                        auto tA = A.template slice<16, 16>(k0, 0);
                        auto tB = B.template slice<32, 16>(0, 0);
                        op.run(tA, tB, ct_c);
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                    }

                    auto tC = C.template slice<32, 16>(0, 0);
                    ct_c.store(tC);
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    for (int off = int(tid); off < BM * BN; off += NSG * 32) {
                        float acc01 = partial[0][off] + partial[1][off];
                        float acc23 = partial[2][off] + partial[3][off];
                        float acc45 = partial[4][off] + partial[5][off];
                        float acc67 = partial[6][off] + partial[7][off];
                        float acc = (acc01 + acc23) + (acc45 + acc67);
                        int row = off / BN;
                        int col = off - row * BN;
                        y[row * N + n0 + col] = T(acc);
                    }
                    """,
                header: "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n")
        }
    }

    /// 4 rows: each threadgroup owns four output columns; its simdgroups
    /// split K, each lane walks packed weights with all four input rows in
    /// registers, and the partials are reduced through threadgroup memory.
    func splitKKernel(groupSize: Int, dtype: DType, kParts: Int) -> MLXFast.MLXFastKernel {
        kernel("m4_ksplit_kp\(kParts)_gs\(groupSize)_\(Self.tag(dtype))") {
            MLXFast.metalKernel(
                name: "verify_m4_ksplit_np_kp\(kParts)_gs\(groupSize)_\(Self.tag(dtype))",
                inputNames: ["x", "w_q", "scales", "biases", "K_size", "N_size"],
                outputNames: ["y"],
                source: """
                    using namespace metal;
                    constexpr int M = 4;
                    constexpr int BN = 4;
                    constexpr int K_PARTS = \(kParts);
                    constexpr int GS = \(groupSize);

                    uint part = simdgroup_index_in_threadgroup;
                    uint lane = thread_index_in_simdgroup;
                    uint tg_n = threadgroup_position_in_grid.y;

                    int K = int(K_size);
                    int N = int(N_size);
                    int K_by_8 = K / 8;
                    int K_by_gs = K / GS;
                    int n0 = int(tg_n) * BN;
                    int packs_per_part = K_by_8 / K_PARTS;
                    int pack_start = int(part) * packs_per_part;
                    int pack_end = (int(part) == K_PARTS - 1) ? K_by_8 : pack_start + packs_per_part;

                    float acc[BN * M];
                    for (int i = 0; i < BN * M; ++i) {
                        acc[i] = 0.0f;
                    }

                    using Vec8 = vec<T, 8>;
                    const device Vec8 *xv = (const device Vec8*)x;

                    for (int pack = pack_start + int(lane); pack < pack_end; pack += 32) {
                        int k_base = pack * 8;
                        Vec8 v0 = xv[(0 * K + k_base) / 8];
                        Vec8 v1 = xv[(1 * K + k_base) / 8];
                        Vec8 v2 = xv[(2 * K + k_base) / 8];
                        Vec8 v3 = xv[(3 * K + k_base) / 8];
                        uint32_t packed[4];
                        float sc[4];
                        float bi[4];
                        for (int j = 0; j < 4; ++j) {
                            packed[j] = w_q[(n0 + j) * K_by_8 + pack];
                            sc[j] = float(scales[(n0 + j) * K_by_gs + (k_base / GS)]);
                            bi[j] = float(biases[(n0 + j) * K_by_gs + (k_base / GS)]);
                        }
                        for (int j = 0; j < 4; ++j) {
                            for (int ki = 0; ki < 8; ++ki) {
                                float wv = float((packed[j] >> (ki * 4)) & 0xFu) * sc[j] + bi[j];
                                acc[j * M + 0] += float(v0[ki]) * wv;
                                acc[j * M + 1] += float(v1[ki]) * wv;
                                acc[j * M + 2] += float(v2[ki]) * wv;
                                acc[j * M + 3] += float(v3[ki]) * wv;
                            }
                        }
                    }

                    for (int i = 0; i < BN * M; ++i) {
                        acc[i] = simd_sum(acc[i]);
                    }

                    threadgroup float partial[K_PARTS * BN * M];
                    if (lane == 0) {
                        for (int i = 0; i < BN * M; ++i) {
                            partial[int(part) * BN * M + i] = acc[i];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    if (part == 0 && lane < BN * M) {
                        float total = 0.0f;
                        for (int p = 0; p < K_PARTS; ++p) {
                            total += partial[p * BN * M + int(lane)];
                        }
                        int j = int(lane) / M;
                        int row = int(lane) - j * M;
                        int n_global = n0 + j;
                        if (n_global < N) {
                            y[row * N + n_global] = T(total);
                        }
                    }
                    """)
        }
    }
}
