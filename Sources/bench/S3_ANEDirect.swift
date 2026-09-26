import Foundation
import ANEBridge

struct ANESuite {
    struct Cfg: Hashable {
        var name: String
        var ch: Int
        var sp: Int
        var depth: Int
        var int8: Bool
        var fp16IO: Bool
        var subset: String
    }

    static let allConfigs: [Cfg] = [
        // dispatch-limited floor
        Cfg(name: "dispatch_floor", ch: 256, sp: 64, depth: 1, int8: false, fp16IO: false, subset: "dispatch"),
        // single-op scaling (SRAM cliff via working set growth)
        Cfg(name: "scale_256", ch: 256, sp: 256, depth: 1, int8: false, fp16IO: false, subset: "scaling"),
        Cfg(name: "scale_512", ch: 512, sp: 512, depth: 1, int8: false, fp16IO: false, subset: "scaling"),
        Cfg(name: "scale_1024", ch: 1024, sp: 1024, depth: 1, int8: false, fp16IO: false, subset: "scaling"),
        Cfg(name: "scale_2048", ch: 2048, sp: 2048, depth: 1, int8: false, fp16IO: false, subset: "scaling"),
        Cfg(name: "scale_4096", ch: 4096, sp: 4096, depth: 1, int8: false, fp16IO: false, subset: "scaling"),
        // deep-graph peak FP16
        Cfg(name: "peak_512x64_d32", ch: 512, sp: 64, depth: 32, int8: false, fp16IO: false, subset: "peak_fp16"),
        Cfg(name: "peak_512x64_d48", ch: 512, sp: 64, depth: 48, int8: false, fp16IO: false, subset: "peak_fp16"),
        Cfg(name: "peak_512x64_d64", ch: 512, sp: 64, depth: 64, int8: false, fp16IO: false, subset: "peak_fp16"),
        Cfg(name: "peak_256x64_d128", ch: 256, sp: 64, depth: 128, int8: false, fp16IO: false, subset: "peak_fp16"),
        Cfg(name: "peak_256x64_d256", ch: 256, sp: 64, depth: 256, int8: false, fp16IO: false, subset: "peak_fp16"),
        Cfg(name: "peak_512x64_d64_fp16io", ch: 512, sp: 64, depth: 64, int8: false, fp16IO: true, subset: "peak_fp16"),
        // deep-graph peak INT8 W8A16
        Cfg(name: "peak_int8_512x64_d64", ch: 512, sp: 64, depth: 64, int8: true, fp16IO: false, subset: "peak_int8"),
        Cfg(name: "peak_int8_512x64_d128", ch: 512, sp: 64, depth: 128, int8: true, fp16IO: false, subset: "peak_int8"),
        // SRAM working-set sweep (depth=16)
        Cfg(name: "sram_2.6MB", ch: 256, sp: 32, depth: 16, int8: false, fp16IO: false, subset: "sram"),
        Cfg(name: "sram_10MB", ch: 512, sp: 64, depth: 16, int8: false, fp16IO: false, subset: "sram"),
        Cfg(name: "sram_24MB", ch: 768, sp: 96, depth: 16, int8: false, fp16IO: false, subset: "sram"),
        Cfg(name: "sram_42MB", ch: 1024, sp: 128, depth: 16, int8: false, fp16IO: false, subset: "sram"),
        Cfg(name: "sram_94MB", ch: 1536, sp: 192, depth: 16, int8: false, fp16IO: false, subset: "sram"),
        Cfg(name: "sram_168MB", ch: 2048, sp: 256, depth: 16, int8: false, fp16IO: false, subset: "sram"),
        // shared-blob vs per-file A/B (M4 measured 1.58x shared-faster; verify H17)
        Cfg(name: "blob_probe_d1", ch: 512, sp: 64, depth: 1, int8: false, fp16IO: false, subset: "blob_probe"),
        Cfg(name: "blob_probe_d2", ch: 512, sp: 64, depth: 2, int8: false, fp16IO: false, subset: "blob_probe"),
        Cfg(name: "blob_512x64_d16", ch: 512, sp: 64, depth: 16, int8: false, fp16IO: false, subset: "blob_ab"),
        Cfg(name: "blob_512x64_d128", ch: 512, sp: 64, depth: 128, int8: false, fp16IO: false, subset: "blob_ab")
    ]

    static func workingSetMB(_ c: Cfg) -> Double {
        let perLayer = (c.ch * c.ch * (c.int8 ? 1 : 2) + 2 * c.ch * c.sp * 2)
        return Double(perLayer * c.depth) / 1024 / 1024
    }

    static func flops(_ c: Cfg) -> Double {
        2.0 * Double(c.ch) * Double(c.ch) * Double(c.sp) * Double(c.depth)
    }

    static func run(subsets: [String], iters: Int) throws -> SuiteSection {
        var section = SuiteSection(id: "s3_ane_raw", title: "ANE raw throughput (direct _ANEInMemoryModel private API)")
        section.notes.append("Methodology: MIL 1x1-conv chains via _ANEInMemoryModelDescriptor, IOSurface fp32 I/O (or fp16 in), mach_absolute_time, median of N. int8 = W8A16 (per-channel symmetric, constexpr_affine_dequantize, same scheme as CoreML-compiled quantized models). The conv chain is the ANE's native dense datapath (matmul/linear layers fold into the convolution datapath; a linear y=x@W.T is a 1x1 conv) — see docs/research-notes.md R7 and arXiv:2606.22283. Adapted from maderix/ANE (MIT).")

        guard ane_bridge_init() == 0 else {
            section.notes.append("NOT RUN: ANE private API unavailable (spike NO-GO or research track pending)")
            return section
        }

        let configs = allConfigs.filter { subsets.contains($0.subset) || subsets.isEmpty }
        for cfg in configs {
            if cfg.name.hasPrefix("blob_") { try blobArm(section: &section, cfg: cfg, iters: iters); continue }
            let chain = MILConvChain(ch: cfg.ch, sp: cfg.sp, depth: cfg.depth, int8: cfg.int8, fp16IO: cfg.fp16IO)
            let mil = chain.milText()
            let (blob, _) = chain.buildWeightBlob()
            let inData = inputBytes(for: cfg)
            let outSize = outputBytes(for: cfg)
            let inSize = inData.count
            let sizesIn = [inSize]
            let sizesOut = [outSize]

            let sw = Stopwatch()
            var kernel: UnsafeMutableRawPointer?
            let milData = mil.data(using: .utf8)!
            blob.withUnsafeBytes { bPtr in
                sizesIn.withUnsafeBufferPointer { spIn in
                    sizesOut.withUnsafeBufferPointer { spOut in
                        milData.withUnsafeBytes { mPtr in
                            sw.start()
                            kernel = ane_bridge_compile(mPtr.baseAddress!, milData.count, bPtr.baseAddress, blob.count,
                                                        1, spIn.baseAddress!, 1, spOut.baseAddress!)
                        }
                    }
                }
            }
            guard let k = kernel else {
                section.notes.append("\(cfg.name): compile failed (skipped)")
                continue
            }
            inData.withUnsafeBytes { p in ane_bridge_write_input(k, 0, p.baseAddress, p.count) }
            var ok = ane_bridge_eval(k)
            for _ in 0..<5 where ok { ok = ane_bridge_eval(k) }
            guard ok else {
                ane_bridge_free(k)
                section.notes.append("\(cfg.name): eval failed (skipped)")
                continue
            }
            var times: [Double] = []
            for _ in 0..<iters {
                sw.start()
                guard ane_bridge_eval(k) else { break }
                times.append(sw.elapsedMs())
            }
            ane_bridge_free(k)
            guard let st = RunStats.from(timesMs: times) else {
                section.notes.append("\(cfg.name): no successful iters (skipped)")
                continue
            }
            let tops = flops(cfg) / 1e12 / (st.median / 1000.0)
            section.runs.append(.make(
                name: cfg.name,
                variant: cfg.int8 ? "int8 W8A16" : (cfg.fp16IO ? "fp16 IO" : "fp32 IO"),
                config: ["ch": "\(cfg.ch)", "sp": "\(cfg.sp)", "depth": "\(cfg.depth)",
                         "subset": cfg.subset, "working_set_mb": String(format: "%.1f", workingSetMB(cfg))],
                unit: "ms/eval",
                stats: st,
                derived: ["TOPS": tops, "GBs_weights": Double(chain.weightBytes) / st.median / 1e6]
            ))
        }
        section.notes.append("compiles this run: \(ane_bridge_get_compile_count()) (per-process limit ~119)")
        return section
    }

    static func blobArm(section: inout SuiteSection, cfg: Cfg, iters: Int) throws {
        let inData = inputBytes(for: cfg)
        let outSize = outputBytes(for: cfg)
        let sizesIn = [inData.count]
        let sizesOut = [outSize]
        var sharedOut = [UInt8](repeating: 0, count: outSize)
        for mode in ["shared", "per_file"] {
            let perFile = mode == "per_file"
            let chain = MILConvChain(ch: cfg.ch, sp: cfg.sp, depth: cfg.depth, int8: false, fp16IO: false, perFile: perFile)
            let mil = chain.milText()
            let milData = mil.data(using: .utf8)!
            let sw = Stopwatch()
            var kernel: UnsafeMutableRawPointer?
            if perFile {
                let files = chain.buildWeightFiles()
                let bufs = files.map { f -> UnsafeMutablePointer<UInt8> in
                    let p = UnsafeMutablePointer<UInt8>.allocate(capacity: f.data.count)
                    f.data.withUnsafeBytes { _ = memcpy(p, $0.baseAddress!, f.data.count) }
                    return p
                }
                defer { for p in bufs { p.deallocate() } }
                var ptrs: [UnsafeRawPointer?] = bufs.map { UnsafeRawPointer($0) }
                var lens: [Int] = files.map { $0.data.count }
                var cNames: [UnsafeMutablePointer<CChar>?] = files.map { strdup($0.name) }
                defer { for p in cNames { free(p) } }
                sizesIn.withUnsafeBufferPointer { spIn in
                    sizesOut.withUnsafeBufferPointer { spOut in
                        milData.withUnsafeBytes { mPtr in
                            sw.start()
                            kernel = ptrs.withUnsafeMutableBufferPointer { bp in
                                lens.withUnsafeMutableBufferPointer { lp in
                                    cNames.withUnsafeMutableBufferPointer { np in
                                        ane_bridge_compile_files(mPtr.baseAddress!, milData.count, 1,
                                                                 UnsafeRawPointer(bp.baseAddress!).assumingMemoryBound(to: Optional<UnsafePointer<UInt8>>.self),
                                                                 lp.baseAddress!,
                                                                 UnsafeRawPointer(np.baseAddress!).assumingMemoryBound(to: Optional<UnsafePointer<CChar>>.self),
                                                                 1, spIn.baseAddress!, 1, spOut.baseAddress!)
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                let (blob, _) = chain.buildWeightBlob()
                blob.withUnsafeBytes { bPtr in
                    sizesIn.withUnsafeBufferPointer { spIn in
                        sizesOut.withUnsafeBufferPointer { spOut in
                            milData.withUnsafeBytes { mPtr in
                                sw.start()
                                kernel = ane_bridge_compile(mPtr.baseAddress!, milData.count, bPtr.baseAddress, blob.count,
                                                            1, spIn.baseAddress!, 1, spOut.baseAddress!)
                            }
                        }
                    }
                }
            }
            guard let k = kernel else {
                section.runs.append(.failed(name: cfg.name, variant: mode, config: ["weights_mode": mode],
                                            unit: "ms", status: "compile_failed",
                                            error: "n_files=\(perFile ? cfg.depth : 1)"))
                continue
            }
            inData.withUnsafeBytes { p in ane_bridge_write_input(k, 0, p.baseAddress, p.count) }
            var ok = ane_bridge_eval(k)
            for _ in 0..<5 where ok { ok = ane_bridge_eval(k) }
            guard ok else {
                ane_bridge_free(k)
                section.runs.append(.failed(name: cfg.name, variant: mode, config: ["weights_mode": mode],
                                            unit: "ms", status: "compile_failed", error: "eval failed"))
                continue
            }
            let compileMs = sw.elapsedMs()
            var times: [Double] = []
            for _ in 0..<iters {
                sw.start()
                guard ane_bridge_eval(k) else { break }
                times.append(sw.elapsedMs())
            }
            var out = [UInt8](repeating: 0, count: outSize)
            ane_bridge_read_output(k, 0, &out, outSize)
            ane_bridge_free(k)
            guard let st = RunStats.from(timesMs: times) else {
                section.runs.append(.failed(name: cfg.name, variant: mode, config: ["weights_mode": mode],
                                            unit: "ms", status: "validation_failed", error: "no successful iters"))
                continue
            }
            if mode == "shared" {
                sharedOut = out
            } else {
                let mismatch = zip(out, sharedOut).reduce(0) { $0 + ($1.0 != $1.1 ? 1 : 0) }
                if mismatch > 0 {
                    section.runs.append(.failed(name: cfg.name, variant: mode, config: ["weights_mode": mode],
                                                unit: "ms", status: "validation_failed",
                                                error: "per_file output differs from shared at \(mismatch) bytes"))
                    continue
                }
            }
            section.runs.append(.make(
                name: cfg.name, variant: mode,
                config: ["ch": "\(cfg.ch)", "sp": "\(cfg.sp)", "depth": "\(cfg.depth)",
                         "weights_mode": mode, "subset": cfg.subset,
                         "weight_files": "\(perFile ? cfg.depth : 1)"],
                unit: "ms", stats: st,
                derived: ["compile_ms": compileMs,
                          "TOPS": flops(cfg) / 1e12 / (st.median / 1000.0)]))
        }
    }

    static func inputBytes(for cfg: Cfg) -> [UInt8] {
        if cfg.fp16IO {
            var h = [UInt16]()
            h.reserveCapacity(cfg.ch * cfg.sp)
            for c in 0..<cfg.ch {
                for s in 0..<cfg.sp {
                    let v = Float(((c * 13 + s * 7) % 11) - 5) * 0.01
                    h.append(Float16(v).bitPattern)
                }
            }
            return h.withUnsafeBytes { Array($0) }
        }
        var x = [Float]()
        x.reserveCapacity(cfg.ch * cfg.sp)
        for c in 0..<cfg.ch {
            for s in 0..<cfg.sp {
                x.append(Float(((c * 13 + s * 7) % 11) - 5) * 0.01)
            }
        }
        return x.withUnsafeBytes { Array($0) }
    }

    static func outputBytes(for cfg: Cfg) -> Int {
        cfg.ch * cfg.sp * 4
    }
}
