import Foundation
import ANEBridge

struct SpikeResult: Codable {
    var go: Bool
    var compileLoadMs: Double?
    var evalMedianMs: Double?
    var maxRelErr: Double?
    var notes: [String]
    var os: String
    var timestamp: String
}

enum SpikeA {
    static func run(iters: Int) throws -> SpikeResult {
        var notes: [String] = []
        let v = ProcessInfo.processInfo.operatingSystemVersion
        let os = "macOS \(v.majorVersion).\(v.minorVersion)"
        let stamp = { ISO8601DateFormatter().string(from: Date()) }

        guard ane_bridge_init() == 0 else {
            return SpikeResult(go: false, compileLoadMs: nil, evalMedianMs: nil,
                               maxRelErr: nil, notes: ["ane_bridge_init failed (framework dlopen or class resolution)"],
                               os: os, timestamp: stamp())
        }
        notes.append("framework dlopen + class resolution OK")

        var go = false
        let ch = 512, sp = 64
        let chain = MILConvChain(ch: ch, sp: sp, depth: 1, int8: false, fp16IO: false)
        let (checkOK, compileMs) = verifyModel(chain: chain, notes: &notes, iters: iters, tag: "fp16 512x64 d1")
        go = checkOK
        let chain8 = MILConvChain(ch: 256, sp: 64, depth: 2, int8: true, fp16IO: false)
        let (ok8, _) = verifyModel(chain: chain8, notes: &notes, iters: iters, tag: "int8 256x64 d2")
        go = go && ok8
        let r = SpikeResult(go: go, compileLoadMs: compileMs, evalMedianMs: nil,
                            maxRelErr: nil, notes: notes, os: os, timestamp: stamp())
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted]
        let url = JSONReport.resultsDir.appendingPathComponent("ane_spike.json")
        try enc.encode(r).write(to: url)
        return r
    }

    static func verifyModel(chain: MILConvChain, notes: inout [String], iters: Int, tag: String) -> (Bool, Double) {
        let mil = chain.milText()
        let (blob, blobLen) = chain.buildWeightBlob()
        notes.append("[\(tag)] blob \(blobLen) B, MIL \(mil.count) chars")
        let input = chain.deterministicInput()
        let expected: [Float]
        if chain.int8 {
            expected = cpuRefConvChain1x1(w: chain.int8QuantizedWeights().eff, x: input, ch: chain.ch, sp: chain.sp, depth: chain.depth)
        } else {
            expected = cpuRefConvChain1x1(w: chain.deterministicWeights(), x: input, ch: chain.ch, sp: chain.sp, depth: chain.depth)
        }
        let inBytes = chain.ch * chain.sp * 4
        let sw = Stopwatch()
        var kernel: UnsafeMutableRawPointer?
        var compileMs = -1.0
        let milData = mil.data(using: .utf8)!
        let sizes = [inBytes]
        blob.withUnsafeBytes { blobPtr in
            sizes.withUnsafeBufferPointer { sp in
                milData.withUnsafeBytes { milPtr in
                    sw.start()
                    kernel = ane_bridge_compile(
                        milPtr.baseAddress!, milData.count,
                        blobPtr.baseAddress, blob.count,
                        1, sp.baseAddress!, 1, sp.baseAddress!)
                    compileMs = sw.elapsedMs()
                }
            }
        }
        guard let k = kernel else {
            notes.append("[\(tag)] compile/load failed (research track R2)")
            return (false, compileMs)
        }
        notes.append("[\(tag)] compile+load OK in \(String(format: "%.1f", compileMs)) ms")
        input.withUnsafeBytes { p in ane_bridge_write_input(k, 0, p.baseAddress, p.count) }
        guard ane_bridge_eval(k) else {
            ane_bridge_free(k)
            notes.append("[\(tag)] first eval failed")
            return (false, compileMs)
        }
        var out = [Float](repeating: 0, count: chain.ch * chain.sp)
        out.withUnsafeMutableBytes { p in ane_bridge_read_output(k, 0, p.baseAddress, p.count) }
        var maxAbs: Double = 0, maxAbsOut: Double = 0, maxRel: Double = 0
        for i in 0..<chain.ch * chain.sp {
            let d = abs(Double(out[i]) - Double(expected[i]))
            maxAbs = max(maxAbs, d)
            maxAbsOut = max(maxAbsOut, abs(Double(expected[i])))
            let denom = max(abs(Double(expected[i])), 1e-3)
            maxRel = max(maxRel, d / denom)
        }
        let tol = max(1e-4, 0.01 * maxAbsOut)
        notes.append("[\(tag)] max_abs_err=\(String(format: "%.3e", maxAbs)) (tol \(String(format: "%.3e", tol))), max_rel_err=\(String(format: "%.4f", maxRel))")
        for _ in 0..<5 { _ = ane_bridge_eval(k) }
        var times: [Double] = []
        for _ in 0..<iters {
            sw.start()
            _ = ane_bridge_eval(k)
            times.append(sw.elapsedMs())
        }
        let st = RunStats.from(timesMs: times)
        ane_bridge_free(k)
        if let st { notes.append("[\(tag)] eval median \(String(format: "%.3f", st.median)) ms") }
        return (maxAbs <= tol, compileMs)
    }
}
