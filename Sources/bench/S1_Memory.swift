import Foundation
import Metal
import Accelerate

struct MemorySuite {
    static func run(device: MTLDevice, iters: Int) throws -> SuiteSection {
        var section = SuiteSection(id: "s1_memory", title: "Memory bandwidth (CPU + GPU)")

        // Pre-flight: refuse to measure while other work contends for the SoC —
        // background daemons (mds/backupd) produce wide, unreproducible CPU
        // bandwidth windows on this platform.
        var load: [Double] = []
        var prevBusy = 0.0, prevTotal = 0.0
        var havePrev = false
        for _ in 0..<4 {
            var hostLoad = host_cpu_load_info()
            var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.size / MemoryLayout<integer_t>.size)
            withUnsafeMutablePointer(to: &hostLoad) {
                _ = $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
                }
            }
            let busy = Double(hostLoad.cpu_ticks.0 + hostLoad.cpu_ticks.1 + hostLoad.cpu_ticks.3)
            let total = busy + Double(hostLoad.cpu_ticks.2)
            if havePrev, total > prevTotal {
                load.append((busy - prevBusy) / (total - prevTotal))
            }
            prevBusy = busy; prevTotal = total; havePrev = true
            usleep(500_000)
        }
        let avgLoad = load.isEmpty ? 0 : load.reduce(0, +) / Double(load.count)
        if avgLoad > 0.15 {
            throw NSError(domain: "bench", code: 1,
                          userInfo: [NSLocalizedDescriptionKey:
                            "SoC busy (avg CPU load \(String(format: "%.0f%%", avgLoad * 100)) over pre-flight) — retry when idle"])
        }
        section.notes.append("CPU: vDSP vectorized, one chunk per core. GPU: shared storage buffers, full DRAM traffic. Tensor: MTL4 encoder native tensor copy.")
        section.notes.append("CPU timed in three 3s windows (median per window reported); pre-flight aborts if the SoC is busy so background daemons cannot contaminate a measurement.")
        section.notes.append("GPU timed via Metal GPU timestamps (gpuEndTime-gpuStartTime, 8 dispatches/command buffer) to exclude CPU commit/wait overhead; a self-calibrating mixed CPU+GPU primer runs until measured GPU-copy bandwidth recovers to >=340 GB/s sustained over two consecutive windows (20-120s) so deep-idle processes converge to the same power state.")

        // ---- CPU ----
        let cpuBytes = 512 * 1024 * 1024
        let n = cpuBytes / MemoryLayout<Float>.size
        let cores = max(1, ProcessInfo.processInfo.activeProcessorCount)
        let chunk = (n / cores) & ~4095
        let nchunks = n / chunk
        let aBuf = CPUBuffer([Float](repeating: 1.0, count: n))
        let cBuf = CPUBuffer([Float](repeating: 3.0, count: n))
        let twoBuf = CPUBuffer([Float](repeating: 2.0, count: chunk))

        // Windowed CPU timing: warm 1.5 s, then take the median of 3-second
        // windows x3 and report per-window medians. A background-daemon
        // interference window can contaminate at most one window, and the
        // cross-window median lands in the stable band.
        func cpuWindowed(_ work: () -> Void) -> [Double] {
            let warm = Stopwatch()
            warm.start()
            repeat { work() } while warm.elapsedMs() < 1500
            var medians: [Double] = []
            for _ in 0..<3 {
                var times: [Double] = []
                let wStart = Stopwatch()
                wStart.start()
                while wStart.elapsedMs() < 3000 {
                    let it = Stopwatch()
                    it.start()
                    work()
                    times.append(it.elapsedMs())
                }
                if let s = RunStats.from(timesMs: times) { medians.append(s.median) }
            }
            return medians
        }

        var times = cpuWindowed {
            DispatchQueue.concurrentPerform(iterations: nchunks) { i in
                var s: Float = 0
                vDSP_sve(aBuf.base + i * chunk, 1, &s, vDSP_Length(chunk))
                if s == -1.2345 { print("x") }
            }
        }
        if let st = RunStats.from(timesMs: times) {
            section.runs.append(.make(name: "cpu_read", variant: "512MB", config: ["cores": "\(cores)"], unit: "GB/s",
                                      stats: st, derived: ["GBs": gbPerSec(bytes: cpuBytes, medianMs: st.median), "p95_GBs": gbPerSec(bytes: cpuBytes, medianMs: st.p95)]))
        }
        times = cpuWindowed {
            DispatchQueue.concurrentPerform(iterations: nchunks) { i in
                var fillVal: Float = 1
                vDSP_vfill(&fillVal, aBuf.base + i * chunk, 1, vDSP_Length(chunk))
            }
        }
        if let st = RunStats.from(timesMs: times) {
            section.runs.append(.make(name: "cpu_write", variant: "512MB", config: ["cores": "\(cores)"], unit: "GB/s",
                                      stats: st, derived: ["GBs": gbPerSec(bytes: cpuBytes, medianMs: st.median), "p95_GBs": gbPerSec(bytes: cpuBytes, medianMs: st.p95)]))
        }
        times = cpuWindowed {
            DispatchQueue.concurrentPerform(iterations: nchunks) { i in
                vDSP_vsma(aBuf.base + i * chunk, 1, twoBuf.base,
                          cBuf.base + i * chunk, 1, aBuf.base + i * chunk, 1,
                          vDSP_Length(chunk))
            }
        }
        if let st = RunStats.from(timesMs: times) {
            section.runs.append(.make(name: "cpu_triad", variant: "512MB", config: ["cores": "\(cores)"], unit: "GB/s",
                                      stats: st, derived: ["GBs": gbPerSec(bytes: cpuBytes * 3, medianMs: st.median), "p95_GBs": gbPerSec(bytes: cpuBytes * 3, medianMs: st.p95)]))
        }

        // ---- mixed CPU+GPU primer (converges the SoC power allocator before
        // any subtest: idle-launched processes otherwise start with a cold
        // split — boosted CPU, idle GPU clocks — and CPU/GPU subtests then
        // anti-correlate by up to 2x) ----
        do {
            let primeLib = try ShaderLib.makeLibrary(device: device, file: "memory")
            let primeFn = primeLib.makeFunction(name: "mem_copy")!
            let primePipe = try device.makeComputePipelineState(function: primeFn)
            let gpuBytes = 256 * 1024 * 1024
            let gn = Int(gpuBytes) / MemoryLayout<Float>.size
            let queue = device.makeCommandQueue()!
            let b1 = device.makeBuffer(length: gpuBytes, options: .storageModeShared)!
            let b2 = device.makeBuffer(length: gpuBytes, options: .storageModeShared)!
            let sw = Stopwatch()
            sw.start()
            var lastGb = 0.0
            var plateauWindows = 0
            while sw.elapsedMs() < 120_000 {
                let win = Stopwatch()
                win.start()
                var gpuMs = 0.0
                var copies = 0
                while win.elapsedMs() < 2000 || sw.elapsedMs() < 20_000 {
                    DispatchQueue.concurrentPerform(iterations: nchunks) { i in
                        var fillVal: Float = 1
                        vDSP_vfill(&fillVal, aBuf.base + i * chunk, 1, vDSP_Length(chunk))
                    }
                    let cmd = queue.makeCommandBuffer()!
                    let enc = cmd.makeComputeCommandEncoder()!
                    enc.setComputePipelineState(primePipe)
                    enc.setBuffer(b1, offset: 0, index: 0)
                    enc.setBuffer(b2, offset: 0, index: 1)
                    enc.dispatchThreadgroups(MTLSize(width: max(1, gn / 32), height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    enc.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()
                    gpuMs += (cmd.gpuEndTime - cmd.gpuStartTime) * 1000
                    copies += 1
                }
                guard copies > 0, gpuMs > 0 else { continue }
                let gb = Double(copies) * Double(gpuBytes) * 2 / gpuMs / 1e6
                lastGb = gb
                if gb >= 340 && plateauWindows >= 1 { break }
                if gb >= 340 { plateauWindows = 1 } else { plateauWindows = 0 }
            }
            if lastGb < 340 {
                section.notes.append("WARNING: GPU power-state primer hit 120s cap below sustained bandwidth (\(String(format: "%.0f GB/s", lastGb))) — results may be low; rerun when idle")
            }
        }

        // ---- GPU ----
        let lib = try ShaderLib.makeLibrary(device: device, file: "memory")
        let gpuBytes = 256 * 1024 * 1024
        let gn = Int(gpuBytes) / MemoryLayout<Float>.size
        let queue = device.makeCommandQueue()!

        let bufA = device.makeBuffer(length: gpuBytes, options: .storageModeShared)!
        let bufB = device.makeBuffer(length: gpuBytes, options: .storageModeShared)!
        let countBuf = device.makeBuffer(length: 64, options: .storageModeShared)!
        var n4Init: UInt32 = UInt32(gn / 4)
        let n4Buffer = device.makeBuffer(bytes: &n4Init, length: 4, options: .storageModeShared)!

        func gpuBench(_ fnName: String, iters: Int, bytes: Int,
                      setup: (MTLComputeCommandEncoder) -> Void) throws -> [Double] {
            let fn = lib.makeFunction(name: fnName)!
            let pipeline = try device.makeComputePipelineState(function: fn)
            let tgs = pipeline.threadExecutionWidth
            let groups: MTLSize
            if fnName == "mem_read" {
                groups = MTLSize(width: 4096, height: 1, depth: 1)
            } else {
                groups = MTLSize(width: max(1, gn / tgs), height: 1, depth: 1)
            }
            let batch = 8
            func oneBatch() -> Double {
                let cmd = queue.makeCommandBuffer()!
                let enc = cmd.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipeline)
                for _ in 0..<batch {
                    setup(enc)
                    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: tgs, height: 1, depth: 1))
                }
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                return (cmd.gpuEndTime - cmd.gpuStartTime) * 1000.0 / Double(batch)
            }
            let sw = Stopwatch()
            sw.start()
            repeat { _ = oneBatch() } while sw.elapsedMs() < 1500
            var gpu: [Double] = []
            gpu.reserveCapacity(iters)
            for _ in 0..<iters { gpu.append(oneBatch()) }
            return gpu
        }

        do {
            let initVals = [Float](repeating: 1.0, count: gn)
            initVals.withUnsafeBytes { bufA.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            var times = try gpuBench("mem_read", iters: iters, bytes: gpuBytes) { enc in
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBuffer(n4Buffer, offset: 0, index: 1)
                enc.setBuffer(countBuf, offset: 0, index: 2)
            }
            if let st = RunStats.from(timesMs: times) {
                section.runs.append(.make(name: "gpu_read", variant: "256MB", config: [:], unit: "GB/s",
                                          stats: st, derived: ["GBs": gbPerSec(bytes: gpuBytes, medianMs: st.median), "p95_GBs": gbPerSec(bytes: gpuBytes, medianMs: st.p95)]))
            }
            times = try gpuBench("mem_write", iters: iters, bytes: gpuBytes) { enc in
                enc.setBuffer(bufA, offset: 0, index: 0)
            }
            if let st = RunStats.from(timesMs: times) {
                section.runs.append(.make(name: "gpu_write", variant: "256MB", config: [:], unit: "GB/s",
                                          stats: st, derived: ["GBs": gbPerSec(bytes: gpuBytes, medianMs: st.median), "p95_GBs": gbPerSec(bytes: gpuBytes, medianMs: st.p95)]))
            }
            times = try gpuBench("mem_copy", iters: iters, bytes: gpuBytes * 2) { enc in
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBuffer(bufB, offset: 0, index: 1)
            }
            if let st = RunStats.from(timesMs: times) {
                section.runs.append(.make(name: "gpu_copy", variant: "256MB", config: [:], unit: "GB/s",
                                          stats: st, derived: ["GBs": gbPerSec(bytes: gpuBytes * 2, medianMs: st.median), "p95_GBs": gbPerSec(bytes: gpuBytes * 2, medianMs: st.p95)]))
            }
            times = try gpuBench("mem_triad", iters: iters, bytes: gpuBytes * 3) { enc in
                enc.setBuffer(bufB, offset: 0, index: 0)
                enc.setBuffer(bufA, offset: 0, index: 1)
                enc.setBuffer(bufB, offset: 0, index: 2)
            }
            if let st = RunStats.from(timesMs: times) {
                section.runs.append(.make(name: "gpu_triad", variant: "256MB", config: [:], unit: "GB/s",
                                          stats: st, derived: ["GBs": gbPerSec(bytes: gpuBytes * 3, medianMs: st.median), "p95_GBs": gbPerSec(bytes: gpuBytes * 3, medianMs: st.p95)]))
            }
        } catch {
            section.notes.append("GPU kernels failed: \(error)")
        }

        // ---- MTLTensor copy via MTL4 encoder (best effort) ----
        if let q4 = device.makeMTL4CommandQueue() {
            let side = 8192
            let tdesc = MTLTensorDescriptor()
            tdesc.dimensions = ShaderLib.mtlExt([side, side])
            tdesc.dataType = .float32
            tdesc.storageMode = .shared
            guard let tA = try? device.makeTensor(descriptor: tdesc),
                  let tB = try? device.makeTensor(descriptor: tdesc) else {
                section.notes.append("MTLTensor copy: tensor creation unavailable")
                return section
            }
            let tbytes = side * side * 4
            let origin = ShaderLib.mtlExt([0, 0])
            let full = ShaderLib.mtlExt([side, side])
            let alloc = device.makeCommandAllocator()!
            let event = device.makeSharedEvent()!
            var evVal: UInt64 = 0
            let times = Timed.runFor(warmupMs: 1000, iters: iters) {
                let cmd = device.makeCommandBuffer()!
                cmd.beginCommandBuffer(allocator: alloc)
                let enc = cmd.makeComputeCommandEncoder()!
                enc.copy(sourceTensor: tA, sourceOrigin: origin, sourceDimensions: full,
                         destinationTensor: tB, destinationOrigin: origin, destinationDimensions: full)
                enc.endEncoding()
                cmd.endCommandBuffer()
                evVal += 1
                q4.commit([cmd], options: MTL4CommitOptions())
                q4.signalEvent(event, value: evVal)
                _ = event.wait(untilSignaledValue: evVal, timeoutMS: 10_000)
            }
            if let st = RunStats.from(timesMs: times) {
                section.runs.append(.make(name: "tensor_copy", variant: "256MB fp32", config: ["api": "MTL4 encoder"], unit: "GB/s",
                                          stats: st, derived: ["GBs": gbPerSec(bytes: tbytes * 2, medianMs: st.median), "p95_GBs": gbPerSec(bytes: tbytes * 2, medianMs: st.p95)]))
            }
        } else {
            section.notes.append("MTL4 command queue unavailable")
        }

        return section
    }
}

final class CPUBuffer: @unchecked Sendable {
    let base: UnsafeMutablePointer<Float>
    let count: Int
    init(_ values: [Float]) {
        count = values.count
        var raw: UnsafeMutableRawPointer?
        let bytes = Swift.max(1, values.count) * MemoryLayout<Float>.size
        guard posix_memalign(&raw, 2 * 1024 * 1024, bytes) == 0, let raw else {
            fatalError("cpu buffer allocation failed")
        }
        base = raw.bindMemory(to: Float.self, capacity: values.count)
        mlock(raw, bytes)
        let n = values.count
        let per = Swift.max(1, n / 16)
        values.withUnsafeBufferPointer { sb in
            DispatchQueue.concurrentPerform(iterations: 16) { i in
                let start = i * per
                let len = Swift.min(per, n - start)
                if len > 0 {
                    base.advanced(by: start).update(from: sb.baseAddress!.advanced(by: start), count: len)
                }
            }
        }
    }
    deinit {
        UnsafeMutableRawPointer(base).deallocate()
    }
}

func gbPerSec(bytes: Int, medianMs: Double) -> Double {
    guard medianMs > 0 else { return 0 }
    return Double(bytes) / medianMs / 1e6
}
