import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate

struct GEMMSuite {
    static func makeValues(_ n: Int, seed: Int) -> [Float] {
        var v = [Float]()
        v.reserveCapacity(n)
        for i in 0..<n {
            v.append(Float(((i * 31 + seed * 7 + 17) % 97) - 48) * 0.02)
        }
        return v
    }

    static func snapFp8(_ x: Float) -> Float {
        let grid: [Float] = [0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        var best = grid[0]
        for g in grid where abs(g - abs(x)) < abs(best - abs(x)) { best = g }
        return x < 0 ? -best : best
    }

    static func snapFp4(_ x: Float) -> Float {
        let grid: [Float] = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        var best = grid[0]
        for g in grid where abs(g - abs(x)) < abs(best - abs(x)) { best = g }
        return x < 0 ? -best : best
    }

    static func snapFp8e5m2(_ x: Float) -> Float {
        let a = abs(x)
        var best = Float(0)
        var bestErr = a + 1
        for code in 0...127 {
            let v = fp8e5m2Float(UInt8(code))
            let e = abs(v - a)
            if e < bestErr { bestErr = e; best = v }
        }
        return x < 0 ? -best : best
    }

    static func dtypeValues(_ a: [Float], _ b: [Float], _ dtype: String) -> ([Float], [Float]) {
        switch dtype {
        case "fp8": return (a.map(snapFp8), b.map(snapFp8))
        case "fp8e5m2": return (a.map(snapFp8e5m2), b.map(snapFp8e5m2))
        case "fp4": return (a.map(snapFp4), b.map(snapFp4))
        case "fp16": return (a.map { Float(Float16($0)) }, b.map { Float(Float16($0)) })
        case "bf16":
            let r = { (x: Float) in Float(bitPattern: UInt32(bf16Bits(x)) << 16) }
            return (a.map(r), b.map(r))
        case "int8":
            return (a.map { ($0 / 0.02).rounded() }, b.map { ($0 / 0.02).rounded() })
        default: return (a, b)
        }
    }

    static func cblasRef(_ a: [Float], _ b: [Float], n: Int) -> [Float] {
        var c = [Float](repeating: 0, count: n * n)
        a.withUnsafeBufferPointer { pa in
            b.withUnsafeBufferPointer { pb in
                c.withUnsafeMutableBufferPointer { pc in
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                Int32(n), Int32(n), Int32(n), 1, pa.baseAddress!, Int32(n),
                                pb.baseAddress!, Int32(n), 1, pc.baseAddress!, Int32(n))
                }
            }
        }
        return c
    }

    static func run(device: MTLDevice, sizes: [Int], dtypes: [String], paths: [String],
                    iters: Int) throws -> SuiteSection {
        var section = SuiteSection(id: "s2_gemm", title: "GEMM throughput (GPU: MPS / custom MSL / TensorOps)")
        let queue = device.makeCommandQueue()!

        var gemmLib: MTLLibrary?
        if paths.contains("msl") {
            do { gemmLib = try ShaderLib.makeLibrary(device: device, file: "gemm") }
            catch { section.notes.append("MSL gemm library compile failed: \(error)") }
        }
        var toLib: MTLLibrary?
        if paths.contains("tensorops") {
            do { toLib = try ShaderLib.makeLibrary(device: device, file: "tensorops") }
            catch { section.notes.append("TensorOps library compile failed (MSL tensor types/MPS headers): \(error)") }
        }

        let m4: M4Context? = paths.contains("tensorops") ? (try? M4Context(device: device)) : nil

        // DVFS primer: ramp the GPU to sustained clocks before any measurement
        // (small-n cells alone do not reliably reach steady state; without this,
        // back-to-back full runs drift up to 2x on MPS small-size kernels).
        runPrimer(queue: queue, device: device)
        if let lib = toLib, let m4 {
            runTensorOpsProbe(section: &section, lib: lib, m4: m4, device: device)
        }

        for n in sizes {
            let aF = makeValues(n * n, seed: 1)
            let bF = makeValues(n * n, seed: 2)

            for dtype in dtypes {
                let (aV, bV) = dtypeValues(aF, bF, dtype)
                let ref = cblasRef(aV, bV, n: n)
                let tol: Double
                switch dtype {
                case "fp32": tol = 1e-3
                case "fp16": tol = 1e-2
                case "int8": tol = 1e-3
                case "fp8", "fp8e5m2", "fp4": tol = 1e-2
                default: tol = 1e-1
                }

                if paths.contains("mps") {
                    runMPS(section: &section, queue: queue, device: device, n: n,
                           a: aV, b: bV, ref: ref, dtype: dtype, tol: tol, iters: iters)
                }
                if let lib = gemmLib {
                    runMSL(section: &section, lib: lib, queue: queue, device: device, n: n,
                           a: aV, b: bV, ref: ref, dtype: dtype, tol: tol, iters: iters)
                }
                if let lib = toLib, let m4 {
                    runTensorOps(section: &section, lib: lib, m4: m4, device: device, n: n,
                                 a: aV, b: bV, ref: ref, dtype: dtype, tol: tol, iters: iters)
                }
            }
        }
        return section
    }

    static func runPrimer(queue: MTLCommandQueue, device: MTLDevice) {
        let n = 2048
        let buf = device.makeBuffer(length: n * n * 4, options: .storageModeShared)!
        let d = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * 4, dataType: .float32)
        let m = MPSMatrix(buffer: buf, descriptor: d)
        let mps = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                          resultRows: n, resultColumns: n, interiorColumns: n,
                                          alpha: 1.0, beta: 1.0)
        Timed.runFor(warmupMs: 2500, iters: 1) {
            let cmd = queue.makeCommandBuffer()!
            mps.encode(commandBuffer: cmd, leftMatrix: m, rightMatrix: m, resultMatrix: m)
            cmd.commit()
            cmd.waitUntilCompleted()
        }
    }

    // MARK: - MPS

    static func runMPS(section: inout SuiteSection, queue: MTLCommandQueue,
                       device: MTLDevice, n: Int, a: [Float], b: [Float], ref: [Float],
                       dtype: String, tol: Double, iters: Int) {
        let (aDT, cDT) = mpsDtypes(dtype)
        guard let aDT, let cDT else {
            section.notes.append("mps/\(dtype): unsupported data type (skipped)")
            return
        }
        func elem(_ dt: MPSDataType) -> Int {
            switch dt {
            case .float32, .int32: return 4
            case .float16, .bFloat16: return 2
            case .int8, .float8e4m3, .float8e5m2, .float4e2m1: return 1
            default: return 4
            }
        }
        let inBytes = n * n * elem(aDT)
        let outBytes = n * n * elem(cDT)
        let aBuf = device.makeBuffer(length: inBytes, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: inBytes, options: .storageModeShared)!
        let cBuf = device.makeBuffer(length: outBytes, options: .storageModeShared)!
        fillDtypeBuffer(aBuf, a, n, dtype)
        fillDtypeBuffer(bBuf, b, n, dtype)
        let dA = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * elem(aDT), dataType: aDT)
        let dB = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * elem(aDT), dataType: aDT)
        let dC = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * elem(cDT), dataType: cDT)
        let mA = MPSMatrix(buffer: aBuf, descriptor: dA)
        let mB = MPSMatrix(buffer: bBuf, descriptor: dB)
        let mC = MPSMatrix(buffer: cBuf, descriptor: dC)
        let mps = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                          resultRows: n, resultColumns: n, interiorColumns: n,
                                          alpha: 1.0, beta: 0.0)

        var dispatchErr: (any Error)?
        func dispatch() {
            let cmd = queue.makeCommandBuffer()!
            mps.encode(commandBuffer: cmd, leftMatrix: mA, rightMatrix: mB, resultMatrix: mC)
            cmd.commit()
            cmd.waitUntilCompleted()
            dispatchErr = cmd.error
        }
        dispatch()
        if let e = dispatchErr {
            section.notes.append("mps/\(dtype)/n\(n): encode failed \(e) (skipped)")
            section.runs.append(.failed(name: "gemm_\(n)_\(dtype)", variant: "mps",
                                        config: ["n": "\(n)"], unit: "ms", status: "compile_failed",
                                        error: "encode: \(e)"))
            return
        }
        guard let cOut = readC(cBuf, count: n * n, dtype: cDT == .float16 ? "fp16" : "fp32") else {
            section.notes.append("mps/\(dtype)/n\(n): result read failed (skipped)")
            return
        }
        let (maxAbsErr, rel) = relErrMatrix(cOut, ref)
        guard rel < tol else {
            section.notes.append("mps/\(dtype)/n\(n): VALIDATION FAILED max_abs_err=\(maxAbsErr) rel_err=\(rel) (excluded)")
            section.runs.append(.failed(name: "gemm_\(n)_\(dtype)", variant: "mps",
                                        config: ["n": "\(n)"], unit: "ms", status: "validation_failed",
                                        error: "max_abs_err=\(maxAbsErr) rel_err=\(rel) tol=\(tol)",
                                        derived: ["max_abs_err": maxAbsErr, "rel_err": rel]))
            return
        }
        let times = Timed.runFor(warmupMs: 1000, iters: iters, work: dispatch)
        guard let st = RunStats.from(timesMs: times) else { return }
        section.runs.append(.make(name: "gemm_\(n)_\(dtype)", variant: "mps",
                                  config: ["n": "\(n)"], unit: "ms", stats: st,
                                  derived: ["GFLOPS": gflops(n: n, medianMs: st.median),
                                            "max_abs_err": maxAbsErr, "rel_err": rel]))
    }

    static func mpsDtypes(_ dtype: String) -> (MPSDataType?, MPSDataType?) {
        switch dtype {
        case "fp32": return (.float32, .float32)
        case "fp16": return (.float16, .float16)
        case "int8": return (.int8, .float32)
        default: return (nil, nil)
        }
    }

    // MARK: - custom MSL (simdgroup_matrix)

    static func runMSL(section: inout SuiteSection, lib: MTLLibrary, queue: MTLCommandQueue,
                       device: MTLDevice, n: Int, a: [Float], b: [Float], ref: [Float],
                       dtype: String, tol: Double, iters: Int) {
        let fnName: String
        switch dtype {
        case "fp32": fnName = "gemm_f32"
        case "fp16": fnName = n > 4096 ? "gemm_f16_f32acc" : "gemm_f16"
        case "bf16": fnName = "gemm_bf16"
        case "int8": fnName = "gemm_i8"
        case "fp8": fnName = "gemm_fp8"
        case "fp8e5m2": fnName = "gemm_fp8e5m2"
        case "fp4": fnName = "gemm_fp4"
        default:
            section.notes.append("msl/\(dtype): not implemented (skipped)")
            return
        }
        guard let fn = lib.makeFunction(name: fnName) else {
            section.notes.append("msl/\(dtype): kernel missing (skipped)")
            return
        }
        let cCount = n * n
        let elem: Int
        switch dtype {
        case "fp32": elem = 4
        case "fp16", "bf16": elem = 2
        case "int8", "fp8", "fp8e5m2", "fp4": elem = 1
        default:
            section.notes.append("msl/\(dtype): not implemented (skipped)")
            return
        }
        let aBuf = device.makeBuffer(length: cCount * elem, options: .storageModeShared)!
        let bBuf = device.makeBuffer(length: cCount * elem, options: .storageModeShared)!
        let cBytes = (dtype == "int8" || dtype == "fp8" || dtype == "fp8e5m2" || dtype == "fp4" || (dtype == "fp16" && n > 4096)) ? cCount * 4 : cCount * elem
        let cBuf = device.makeBuffer(length: cBytes, options: .storageModeShared)!

        fillKernelBuffers(a: a, b: b, aBuf: aBuf, bBuf: bBuf, n: n, dtype: dtype)

        let consts = device.makeBuffer(length: 12, options: .storageModeShared)!
        let mk: [UInt32] = [UInt32(n), UInt32(n), UInt32(n)]
        mk.withUnsafeBytes { consts.contents().copyMemory(from: $0.baseAddress!, byteCount: 12) }

        guard let pipeline = try? device.makeComputePipelineState(function: fn) else {
            section.notes.append("msl/\(dtype): pipeline failed (skipped)")
            return
        }
        var groups: MTLSize
        var tgSize: MTLSize
        if dtype == "int8" || dtype == "fp8" || dtype == "fp8e5m2" || dtype == "fp4" {
            tgSize = MTLSize(width: 16, height: 16, depth: 1)
            groups = MTLSize(width: n / 16, height: n / 128, depth: 1)
        } else {
            tgSize = MTLSize(width: 32, height: 4, depth: 1)
            groups = MTLSize(width: n / 32, height: n / 32, depth: 1)
        }
        if dtype == "fp16" && n > 4096 {
            tgSize = MTLSize(width: 16, height: 16, depth: 1)
            groups = MTLSize(width: n / 16, height: n / 128, depth: 1)
        }

        func dispatch() {
            let cmd = queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(aBuf, offset: 0, index: 0)
            enc.setBuffer(bBuf, offset: 0, index: 1)
            enc.setBuffer(cBuf, offset: 0, index: 2)
            enc.setBuffer(consts, offset: 0, index: 3)
            enc.setBuffer(consts, offset: 4, index: 4)
            enc.setBuffer(consts, offset: 8, index: 5)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tgSize)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
        }
        dispatch()
        guard let cOut = readC(cBuf, count: cCount, dtype: (dtype == "fp16" && n > 4096) ? "fp32" : dtype) else {
            section.notes.append("msl/\(dtype)/n\(n): result read failed (skipped)")
            return
        }
        let (maxAbsErr, rel) = relErrMatrix(cOut, ref)
        guard rel < tol else {
            section.notes.append("msl/\(dtype)/n\(n): VALIDATION FAILED max_abs_err=\(maxAbsErr) rel_err=\(rel) (excluded)")
            section.runs.append(.failed(name: "gemm_\(n)_\(dtype)", variant: "msl",
                                        config: ["n": "\(n)"], unit: "ms", status: "validation_failed",
                                        error: "max_abs_err=\(maxAbsErr) rel_err=\(rel) tol=\(tol)",
                                        derived: ["max_abs_err": maxAbsErr, "rel_err": rel]))
            return
        }
        let times = Timed.runFor(warmupMs: 1000, iters: iters, work: dispatch)
        guard let st = RunStats.from(timesMs: times) else { return }
        section.runs.append(.make(name: "gemm_\(n)_\(dtype)", variant: "msl",
                                  config: ["n": "\(n)", "tile": "32x32"], unit: "ms", stats: st,
                                  derived: ["GFLOPS": gflops(n: n, medianMs: st.median),
                                            "max_abs_err": maxAbsErr, "rel_err": rel]))
    }

    static func fillKernelBuffers(a: [Float], b: [Float], aBuf: MTLBuffer, bBuf: MTLBuffer,
                                  n: Int, dtype: String) {
        if dtype == "int8" {
            let qb = b.map { Int8($0.rounded()) }
            var ibt = [Int8](repeating: 0, count: n * n)
            for k in 0..<n {
                for nn in 0..<n {
                    ibt[nn * n + k] = qb[k * n + nn]
                }
            }
            fillDtypeBuffer(aBuf, a, n, dtype)
            ibt.withUnsafeBytes { bBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        } else {
            fillDtypeBuffer(aBuf, a, n, dtype)
            fillDtypeBuffer(bBuf, b, n, dtype)
        }
    }

    // MARK: - TensorOps (Metal 4 tensor ops, GPU neural accelerators)

    static func runTensorOpsProbe(section: inout SuiteSection, lib: MTLLibrary, m4: M4Context,
                                  device: MTLDevice) {
        guard let fn = lib.makeFunction(name: "to_gemm_f16"),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            section.notes.append("tensorops/matmul2d probe: kernel unavailable (skipped)")
            return
        }
        let K = 64, mD = 64, nD = 32
        func t(_ dims: [Int], _ dt: MTLTensorDataType) -> MTLTensor? {
            let d = MTLTensorDescriptor()
            d.dimensions = ShaderLib.mtlExt(dims)
            d.dataType = dt
            d.storageMode = .shared
            return try? device.makeTensor(descriptor: d)
        }
        guard let tA = t([K, mD], .float16), let tB = t([nD, K], .float16), let tC = t([nD, mD], .float32) else {
            section.notes.append("tensorops/matmul2d probe: tensor creation failed (skipped)")
            return
        }
        let aH = [UInt16](repeating: 0x3C00, count: K * mD)
        let bH = [UInt16](repeating: 0x3C00, count: nD * K)
        aH.withUnsafeBytes { tA.replace(sliceOrigin: ShaderLib.mtlExt([0, 0]), sliceDimensions: ShaderLib.mtlExt([K, mD]), withBytes: $0.baseAddress!, strides: ShaderLib.mtlExt([1, K])) }
        bH.withUnsafeBytes { tB.replace(sliceOrigin: ShaderLib.mtlExt([0, 0]), sliceDimensions: ShaderLib.mtlExt([nD, K]), withBytes: $0.baseAddress!, strides: ShaderLib.mtlExt([1, nD])) }
        let at = MTL4ArgumentTableDescriptor(); at.maxBufferBindCount = 4
        guard let tab = try? device.makeArgumentTable(descriptor: at) else {
            section.notes.append("tensorops/matmul2d probe: argument table unavailable (skipped)")
            return
        }
        tab.setResource(tA.gpuResourceID, bufferIndex: 0)
        tab.setResource(tB.gpuResourceID, bufferIndex: 1)
        tab.setResource(tC.gpuResourceID, bufferIndex: 2)
        let consts = device.makeBuffer(length: 12, options: .storageModeShared)!
        [UInt32(mD), UInt32(nD), UInt32(K)].withUnsafeBytes { consts.contents().copyMemory(from: $0.baseAddress!, byteCount: 12) }
        tab.setAddress(consts.gpuAddress, index: 3)
        let cmd = device.makeCommandBuffer()!
        cmd.beginCommandBuffer(allocator: m4.alloc)
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setArgumentTable(tab)
        enc.dispatchThreadgroups(threadgroupsPerGrid: MTLSize(width: nD / 32, height: mD / 64, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth * 4, height: 1, depth: 1))
        enc.endEncoding()
        cmd.endCommandBuffer()
        m4.run(cmd)
        var raw = [Float](repeating: 0, count: nD * mD)
        tC.getBytes(&raw, strides: ShaderLib.mtlExt([1, nD]), sliceOrigin: ShaderLib.mtlExt([0, 0]), sliceDimensions: ShaderLib.mtlExt([nD, mD]))
        var cOut = [Float](repeating: 0, count: mD * nD)
        for m in 0..<mD { for nx in 0..<nD { cOut[m * nD + nx] = raw[nx + m * nD] } }
        let (maxAbsErr, rel) = relErrMatrix(cOut, [Float](repeating: Float(K), count: mD * nD))
        if rel < 1e-3 {
            section.notes.append("tensorops/matmul2d probe: PASSED (single 64x32 tile, K=64, all elements correct); "
                                + "multi-tile/multi-K-chunk matmul2d dispatch silently no-ops on macOS 27.0 beta GPUCompiler (32023): full-size rows below report validation_failed, not skipped, because the op runs and returns zeros")
        } else {
            section.notes.append("tensorops/matmul2d probe: FAILED even single-tile (max_abs_err=\(maxAbsErr) rel_err=\(rel)); hardware path unusable on this build")
        }
    }

    static func runTensorOps(section: inout SuiteSection, lib: MTLLibrary, m4: M4Context,
                             device: MTLDevice, n: Int, a: [Float], b: [Float], ref: [Float],
                             dtype: String, tol: Double, iters: Int) {
        let fnMap = ["fp16": "to_gemm_f16", "bf16": "to_gemm_bf16", "int8": "to_gemm_i8",
                     "fp8": "to_gemm_fp8", "fp4": "to_gemm_fp4"]
        guard let fnName = fnMap[dtype] else {
            section.notes.append("tensorops/\(dtype): not supported (skipped)")
            return
        }
        guard let fn = lib.makeFunction(name: fnName) else {
            section.notes.append("tensorops/\(dtype): kernel missing (skipped)")
            return
        }
        guard let pipeline = try? device.makeComputePipelineState(function: fn) else {
            section.notes.append("tensorops/\(dtype): pipeline failed (skipped)")
            return
        }

        let aDT: MTLTensorDataType
        let bDT: MTLTensorDataType
        switch dtype {
        case "fp16": (aDT, bDT) = (.float16, .float16)
        case "bf16": (aDT, bDT) = (.bfloat16, .bfloat16)
        case "int8": (aDT, bDT) = (.int8, .int8)
        case "fp8": (aDT, bDT) = (.float16, .metalFloat8e4m3)
        case "fp4": (aDT, bDT) = (.metalFloat4e2m1, .metalFloat4e2m1)
        default:
            section.notes.append("tensorops/\(dtype): not supported (skipped)")
            return
        }
        let cDT: MTLTensorDataType = dtype == "int8" ? .int32 : .float32

        func makeTensor(_ dt: MTLTensorDataType) -> MTLTensor? {
            let d = MTLTensorDescriptor()
            d.dimensions = ShaderLib.mtlExt([n, n])
            d.dataType = dt
            d.storageMode = .shared
            return (try? device.makeTensor(descriptor: d))
        }
        guard let tA = makeTensor(aDT), let tB = makeTensor(bDT), let tC = makeTensor(cDT) else {
            section.notes.append("tensorops/\(dtype): tensor creation failed (skipped)")
            return
        }

        // Memory order: dim0 (innermost) fastest.
        // A extents (K, M): Adev[k + m*n] = a[m*n + k]
        // B extents (N, K): Bdev[nx + k*n] = b[k*n + nx]
        // C extents (N, M): Cdev[nx + m*n] = c[m*n + nx]
        var aMem = [Float](repeating: 0, count: n * n)
        var bMem = [Float](repeating: 0, count: n * n)
        for m in 0..<n {
            for k in 0..<n { aMem[k + m * n] = a[m * n + k] }
        }
        for nx in 0..<n {
            for k in 0..<n { bMem[nx + k * n] = b[k * n + nx] }
        }
        fillTensor(tA, mem: aMem, aDT, n: n)
        fillTensor(tB, mem: bMem, bDT, n: n)

        let atDesc = MTL4ArgumentTableDescriptor()
        atDesc.maxBufferBindCount = 4
        guard let argTable = try? device.makeArgumentTable(descriptor: atDesc) else {
            section.notes.append("tensorops/\(dtype): argument table unavailable (skipped)")
            return
        }
        argTable.setResource(tA.gpuResourceID, bufferIndex: 0)
        argTable.setResource(tB.gpuResourceID, bufferIndex: 1)
        argTable.setResource(tC.gpuResourceID, bufferIndex: 2)

        let consts = device.makeBuffer(length: 12, options: .storageModeShared)!
        let mk: [UInt32] = [UInt32(n), UInt32(n), UInt32(n)]
        mk.withUnsafeBytes { consts.contents().copyMemory(from: $0.baseAddress!, byteCount: 12) }
        argTable.setAddress(consts.gpuAddress, index: 3)

        let tgs = pipeline.threadExecutionWidth
        let groups = MTLSize(width: n / 32, height: n / 64, depth: 1)

        func dispatch() {
            let cmd = device.makeCommandBuffer()!
            cmd.beginCommandBuffer(allocator: m4.alloc)
            let enc = cmd.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setArgumentTable(argTable)
            enc.dispatchThreadgroups(threadgroupsPerGrid: groups,
                                     threadsPerThreadgroup: MTLSize(width: tgs * 4, height: 1, depth: 1))
            enc.endEncoding()
            cmd.endCommandBuffer()
            m4.run(cmd)
        }
        dispatch()

        let origin = ShaderLib.mtlExt([0, 0])
        let full = ShaderLib.mtlExt([n, n])
        let strides = ShaderLib.mtlExt([1, n])
        var cOut = [Float](repeating: 0, count: n * n)
        if cDT == .int32 {
            var raw = [Int32](repeating: 0, count: n * n)
            tC.getBytes(&raw, strides: strides, sliceOrigin: origin, sliceDimensions: full)
            for m in 0..<n {
                for nx in 0..<n { cOut[m * n + nx] = Float(raw[nx + m * n]) }
            }
        } else {
            var raw = [Float](repeating: 0, count: n * n)
            tC.getBytes(&raw, strides: strides, sliceOrigin: origin, sliceDimensions: full)
            for m in 0..<n {
                for nx in 0..<n { cOut[m * n + nx] = raw[nx + m * n] }
            }
        }
        let (maxAbsErr, rel) = relErrMatrix(cOut, ref)
        guard rel < tol else {
            section.notes.append("tensorops/\(dtype)/n\(n): VALIDATION FAILED max_abs_err=\(maxAbsErr) rel_err=\(rel) (excluded; multi-tile matmul2d silently no-ops on this beta toolchain, see docs/research-notes.md R8)")
            section.runs.append(.failed(name: "gemm_\(n)_\(dtype)", variant: "tensorops",
                                        config: ["n": "\(n)"], unit: "ms", status: "validation_failed",
                                        error: "max_abs_err=\(maxAbsErr) rel_err=\(rel) tol=\(tol); multi-tile matmul2d no-op (R8)",
                                        derived: ["max_abs_err": maxAbsErr, "rel_err": rel]))
            return
        }
        let times = Timed.runFor(warmupMs: 1000, iters: iters, work: dispatch)
        guard let st = RunStats.from(timesMs: times) else { return }
        section.runs.append(.make(name: "gemm_\(n)_\(dtype)", variant: "tensorops",
                                  config: ["n": "\(n)", "tile": "64x32", "simdgroups": "4"], unit: "ms", stats: st,
                                  derived: ["GFLOPS": gflops(n: n, medianMs: st.median),
                                            "max_abs_err": maxAbsErr, "rel_err": rel]))
    }

    static func fillTensor(_ t: MTLTensor, mem: [Float], _ dt: MTLTensorDataType, n: Int) {
        let origin = ShaderLib.mtlExt([0, 0])
        let full = ShaderLib.mtlExt([n, n])
        let strides = ShaderLib.mtlExt([1, n])
        switch dt {
        case .float32:
            mem.withUnsafeBytes { t.replace(sliceOrigin: origin, sliceDimensions: full, withBytes: $0.baseAddress!, strides: strides) }
        case .float16:
            let h = mem.map { f16Bits($0) }
            h.withUnsafeBytes { t.replace(sliceOrigin: origin, sliceDimensions: full, withBytes: $0.baseAddress!, strides: strides) }
        case .bfloat16:
            let h = mem.map { bf16Bits($0) }
            h.withUnsafeBytes { t.replace(sliceOrigin: origin, sliceDimensions: full, withBytes: $0.baseAddress!, strides: strides) }
        case .int8:
            let i8 = mem.map { Int8($0.rounded()) }
            i8.withUnsafeBytes { t.replace(sliceOrigin: origin, sliceDimensions: full, withBytes: $0.baseAddress!, strides: strides) }
        case .metalFloat8e4m3:
            let bytes = mem.map { fp8e4m3Bits($0) }
            bytes.withUnsafeBytes { t.replace(sliceOrigin: origin, sliceDimensions: full, withBytes: $0.baseAddress!, strides: strides) }
        case .metalFloat4e2m1:
            var packed = [UInt8](repeating: 0, count: (mem.count + 1) / 2)
            for i in 0..<(mem.count / 2) {
                packed[i] = fp4e2m1Bits(mem[2 * i]) | (fp4e2m1Bits(mem[2 * i + 1]) << 4)
            }
            packed.withUnsafeBytes { t.replace(sliceOrigin: origin, sliceDimensions: full, withBytes: $0.baseAddress!, strides: strides) }
        default:
            break
        }
    }

}

func fp8e4m3Bits(_ x: Float) -> UInt8 {
    let absx = abs(x)
    if absx < 0.015625 { return x < 0 ? 0x80 : 0 }
    var exp: Int = 0
    var m = absx
    while m >= 1.0 { m /= 2; exp += 1 }
    while m < 0.5 && exp > -6 { m *= 2; exp -= 1 }
    if exp < -6 { return x < 0 ? 0x80 : 0 }
    var mant = Int((m * 16.0 - 8.0).rounded())
    if mant == 8 { mant = 0; exp += 1 }
    var bits = UInt8((exp + 6) << 3) | UInt8(mant & 0x7)
    if x < 0 { bits |= 0x80 }
    return bits
}

func fp8e5m2Float(_ b: UInt8) -> Float {
    let s = (b >> 7) & 1
    let e = Int((b >> 2) & 0x1F)
    let m = Float(b & 0x3)
    let r: Float = e == 0 ? (m / 4.0) * powf(2, -14) : (1 + m / 4.0) * powf(2, Float(e - 15))
    return s == 1 ? -r : r
}

func fp8e5m2Bits(_ x: Float) -> UInt8 {
    let absx = abs(x)
    if absx == 0 { return x < 0 ? 0x80 : 0 }
    var e = Int(floor(log2(absx)))
    var m: Int
    var code: UInt8
    if e < -14 {
        m = Int((absx / powf(2, -16)).rounded())
        if m >= 4 { e = -14; m = 0 }
        code = UInt8(m)
    } else {
        m = Int(((absx / powf(2, Float(e)) - 1) * 4).rounded())
        if m == 4 { e += 1; m = 0 }
        code = UInt8((e + 15) << 2) | UInt8(m & 3)
    }
    if x < 0 { code |= 0x80 }
    return code
}

func fp4e2m1Bits(_ x: Float) -> UInt8 {
    let absx = abs(x)
    let grid: [(Float, UInt8)] = [(0, 0), (0.5, 1), (1, 2), (1.5, 3), (2, 4), (3, 5), (4, 6), (6, 7)]
    var best = grid[0]
    for g in grid where abs(g.0 - absx) < abs(best.0 - absx) { best = g }
    var code = best.1
    if x < 0 { code |= 0x8 }
    return code
}

/// M4 command queue + allocator + shared event for CPU-side completion waits.
final class M4Context {
    let queue: MTL4CommandQueue
    let alloc: MTL4CommandAllocator
    let event: MTLSharedEvent
    private var counter: UInt64 = 0

    init(device: MTLDevice) throws {
        guard let q = device.makeMTL4CommandQueue(),
              let a = device.makeCommandAllocator(),
              let e = device.makeSharedEvent() else {
            throw NSError(domain: "M4Context", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "M4 queue/allocator/event unavailable"])
        }
        queue = q
        alloc = a
        event = e
    }

    func run(_ cmd: MTL4CommandBuffer) {
        counter += 1
        queue.commit([cmd], options: MTL4CommitOptions())
        queue.signalEvent(event, value: counter)
        _ = event.wait(untilSignaledValue: counter, timeoutMS: 10_000)
    }
}

func f16Bits(_ v: Float) -> UInt16 {
    Float16(v).bitPattern
}

func bf16Bits(_ v: Float) -> UInt16 {
    UInt16(truncatingIfNeeded: v.bitPattern >> 16)
}

func relErr(_ got: Float, _ exp: Float) -> Double {
    let d = abs(Double(got) - Double(exp))
    return d / max(abs(Double(exp)), 1e-3)
}

func relErrMatrix(_ got: [Float], _ exp: [Float]) -> (Double, Double) {
    var maxAbs = 0.0
    var maxErr = 0.0
    for i in 0..<got.count {
        maxAbs = max(maxAbs, abs(Double(exp[i])))
        maxErr = max(maxErr, abs(Double(got[i]) - Double(exp[i])))
    }
    return (maxErr, maxErr / max(maxAbs, 1e-6))
}

func gflops(n: Int, medianMs: Double) -> Double {
    2.0 * Double(n) * Double(n) * Double(n) / 1e9 / (medianMs / 1000.0)
}

func fillDtypeBuffer(_ buf: MTLBuffer, _ values: [Float], _ n: Int, _ dtype: String) {
    switch dtype {
    case "fp32":
        values.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    case "fp16":
        let h = values.map { f16Bits($0) }
        h.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    case "bf16":
        let h = values.map { bf16Bits($0) }
        h.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    case "int8":
        let i8 = values.map { Int8($0.rounded()) }
        i8.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    case "fp8":
        let bytes = values.map { fp8e4m3Bits($0) }
        bytes.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    case "fp8e5m2":
        let bytes = values.map { fp8e5m2Bits($0) }
        bytes.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    case "fp4":
        var packed = [UInt8](repeating: 0, count: values.count / 2)
        for i in 0..<(values.count / 2) {
            packed[i] = fp4e2m1Bits(values[2 * i]) | (fp4e2m1Bits(values[2 * i + 1]) << 4)
        }
        packed.withUnsafeBytes { buf.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    default:
        break
    }
}

func readC(_ buf: MTLBuffer, count: Int, dtype: String) -> [Float]? {
    var out = [Float](repeating: 0, count: count)
    let base = buf.contents()
    switch dtype {
    case "int8":
        base.withMemoryRebound(to: Int32.self, capacity: count) { p in
            for i in 0..<count { out[i] = Float(p[i]) }
        }
    case "fp16":
        base.withMemoryRebound(to: UInt16.self, capacity: count) { p in
            for i in 0..<count { out[i] = Float(Float16(bitPattern: p[i])) }
        }
    case "bf16":
        base.withMemoryRebound(to: UInt16.self, capacity: count) { p in
            for i in 0..<count { out[i] = Float(bitPattern: UInt32(p[i]) << 16) }
        }
    default:
        base.withMemoryRebound(to: Float.self, capacity: count) { p in
            for i in 0..<count { out[i] = p[i] }
        }
    }
    return out
}
