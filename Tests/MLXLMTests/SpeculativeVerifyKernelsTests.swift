// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private final class Holder: Module {
    @ModuleInfo var linear: QuantizedLinear
    init(_ linear: QuantizedLinear) {
        _linear.wrappedValue = linear
        super.init()
    }
}

/// The verify kernels must agree with MLX's quantized matmul at the row
/// counts they serve, and leave every other width to it.
@Test
func testVerifyKernelsMatchQuantizedMatmul() throws {
    guard SpeculativeVerifyKernels.isAvailable else { return }
    let (n, k) = (64, 512)
    let base = QuantizedLinear(Linear(k, n, bias: false), groupSize: 64, bits: 4)
    let holder = Holder(
        QuantizedLinear(
            weight: base.weight, bias: nil, scales: base.scales, biases: base.biases,
            groupSize: 64, bits: 4))
    #expect(SpeculativeVerifyKernels.install(in: holder) == 1)
    let installed = try #require(holder.linear as? VerifyQuantizedLinear)
    #expect(installed.servedRows.contains(4))
    #expect(installed.servedRows.contains(16) == SpeculativeVerifyKernels.supportsTensorUnitKernel)

    for rows in [1, 4, 8, 16] {
        let x = MLXRandom.normal([1, rows, k]).asType(.bfloat16)
        let expected = base(x).asType(.float32)
        let actual = holder.linear(x).asType(.float32)
        eval(expected, actual)
        let scale = abs(expected).max().item(Float.self)
        let difference = abs(expected - actual).max().item(Float.self)
        #expect(
            difference <= 0.02 * max(scale, 1),
            "rows \(rows): max |diff| \(difference) at scale \(scale)")
    }
    // A second install is a no-op.
    #expect(SpeculativeVerifyKernels.install(in: holder) == 0)
}

@Test
func testVerifyKernelsSkipIneligibleLinears() {
    // 8-bit and non-multiple shapes stay on MLX.
    let eightBit = Holder(QuantizedLinear(Linear(64, 32, bias: false), groupSize: 64, bits: 8))
    #expect(SpeculativeVerifyKernels.install(in: eightBit) == 0)
    let oddColumns = Holder(QuantizedLinear(Linear(64, 30, bias: false), groupSize: 64, bits: 4))
    #expect(SpeculativeVerifyKernels.install(in: oddColumns) == 0)
}
