import Foundation
import Metal

struct CLI {
    nonisolated(unsafe) static var args: [String] = []

    static func parse() -> (cmd: String, flags: [String: String], positional: [String]) {
        var flags: [String: String] = [:]
        var positional: [String] = []
        var i = 1
        while i < args.count {
            let a = args[i]
            if a.hasPrefix("--") {
                let key = String(a.dropFirst(2))
                if i + 1 < args.count && !args[i + 1].hasPrefix("--") {
                    flags[key] = args[i + 1]
                    i += 2
                } else {
                    flags[key] = "1"
                    i += 1
                }
            } else {
                positional.append(a)
                i += 1
            }
        }
        return (positional.first ?? "help", flags, positional)
    }

    static func ints(_ s: String?, default d: [Int]) -> [Int] {
        guard let s, !s.isEmpty else { return d }
        return s.split(separator: ",").compactMap { Int($0) }
    }

    static func strs(_ s: String?, default d: [String]) -> [String] {
        guard let s, !s.isEmpty else { return d }
        return s.split(separator: ",").map(String.init)
    }

    static func intFlag(_ flags: [String: String], _ key: String, _ d: Int) -> Int {
        Int(flags[key] ?? "") ?? d
    }

    static func device() -> MTLDevice {
        guard let d = MTLCreateSystemDefaultDevice() else {
            FileHandle.standardError.write("no Metal device\n".data(using: .utf8)!)
            exit(1)
        }
        return d
    }

    static func makeReport(machine: MachineInfo, sections: [SuiteSection], status: [String: String]) -> ReportV1 {
        ReportV1(schema: ReportV1.schemaID,
                 generated: ISO8601DateFormatter().string(from: Date()),
                 machine: machine, sections: sections, status: status)
    }

    static func padL(_ s: String, _ w: Int) -> String {
        s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
    }

    static func padR(_ s: String, _ w: Int) -> String {
        s.count >= w ? s : String(repeating: " ", count: w - s.count) + s
    }

    static func printTable(_ section: SuiteSection) {
        print("\n== \(section.title) (\(section.id)) ==")
        if section.runs.isEmpty {
            print("  (no runs)")
        } else {
            let nameW = Swift.max(section.runs.map { $0.name.count }.max() ?? 0, 12)
            let varW = Swift.max(section.runs.map { $0.variant.count }.max() ?? 0, 6)
            print("  " + padL("name", nameW) + "  " + padL("variant", varW) +
                  "  " + padR("iters", 8) + "  " + padR("median", 10) + "  " + padR("p95", 10) + "  " + padR("unit", 8))
            for r in section.runs {
                let med: String
                if r.unit == "GB/s" { med = String(format: "%.1f", r.derived["GBs"] ?? 0) }
                else if r.unit == "ms" || r.unit == "ms/run" || r.unit == "ms/eval" { med = String(format: "%.3f", r.median) }
                else { med = String(format: "%.4f", r.median) }
                let p95s: String
                if r.unit == "GB/s" { p95s = String(format: "%.1f", r.derived["p95_GBs"] ?? 0) }
                else { p95s = String(format: "%.3f", r.p95) }
                print("  " + padL(r.name, nameW) + "  " + padL(r.variant, varW) +
                      "  " + padR("\(r.iters)", 8) + "  " + padR(med, 10) +
                      "  " + padR(p95s, 10) + "  " + padR(r.unit, 8) +
                      (r.runStatus == "ok" ? "" : "  " + r.runStatus))
            }
            let extra = section.runs.compactMap { r -> String? in
                if let g = r.derived["GFLOPS"] { return "\(r.name)/\(r.variant): \(String(format: "%.0f", g)) GFLOPS" }
                if let t = r.derived["TOPS"] { return "\(r.name)/\(r.variant): \(String(format: "%.2f", t)) TOPS" }
                return nil
            }
            for e in extra { print("    \(e)") }
        }
        for n in section.notes { print("  note: \(n)") }
    }

    static func usage() {
        print("""
        silicon-ledger-bench — M5 Max local AI hardware benchmark

        Usage:
          bench probe                     S0 machine fingerprint (JSON + table)
          bench memory [--iters N]        S1 memory bandwidth (CPU vDSP / GPU Metal / MTL4 tensor copy)
          bench gemm [--sizes 512,1024,2048,4096,8192] [--dtypes fp32,fp16,bf16,int8,fp8,fp8e5m2,fp4]
                     [--paths mps,msl,tensorops] [--iters N]
                                          S2 GEMM throughput (MPS / custom MSL simdgroup / TensorOps)
          bench spike [--iters N]         S-A ANE direct-path spike (GO/NO-GO -> results/ane_spike.json)
          bench ane [--subset dispatch,scaling,peak_fp16,peak_int8,sram] [--iters N]
                                          S3 ANE raw throughput (direct _ANEInMemoryModel)
          bench coreai [--models deep_fp16,deep_int8,matmul] [--variants default,ane,gpu] [--iters N]
                                          S4 Core AI .aimodel inference (official stack)
          bench coreml [--iters N]        S5 legacy Core ML ANE-preferred baseline
          bench diff A.json B.json        compare two report JSONs (per-run median delta)
          bench all [--include-coreml] [--iters N]
                                          full run: probe+spike+memory+gemm+ane(if GO)+coreai(+coreml)

        Results: results/<timestamp>-<fingerprint>.json (schema silicon-ledger/1)
        """)
    }

    static func writeReport(_ report: ReportV1) throws {
        let url = try JSONReport.write(report)
        print("\nreport: \(url.path)")
    }

    @MainActor
    static func runAll() async -> Int32 {
        let (cmd, flags, positional) = parse()
        switch cmd {
        case "help", "--help", "-h":
            usage()
            return 0
        case "probe":
            return runProbe()
        case "memory":
            do {
                let iters = intFlag(flags, "iters", 15)
                let machine = Fingerprint.probe()
                let section = try MemorySuite.run(device: device(), iters: iters)
                printTable(section)
                try writeReport(makeReport(machine: machine, sections: [section], status: ["memory": "done"]))
                return 0
            } catch {
                print("memory failed: \(error)")
                return 1
            }
        case "gemm":
            do {
                let iters = intFlag(flags, "iters", 30)
                let sizes = ints(flags["sizes"], default: [512, 1024, 2048, 4096, 8192])
                let dtypes = strs(flags["dtypes"], default: ["fp32", "fp16", "bf16", "int8", "fp8", "fp8e5m2", "fp4"])
                let paths = strs(flags["paths"], default: ["mps", "msl", "tensorops"])
                let machine = Fingerprint.probe()
                let section = try GEMMSuite.run(device: device(), sizes: sizes, dtypes: dtypes, paths: paths, iters: iters)
                printTable(section)
                try writeReport(makeReport(machine: machine, sections: [section], status: ["gemm": "done"]))
                return 0
            } catch {
                print("gemm failed: \(error)")
                return 1
            }
        case "spike":
            do {
                let iters = intFlag(flags, "iters", 200)
                let r = try SpikeA.run(iters: iters)
                print(r.go ? "SPIKE: GO" : "SPIKE: NO-GO")
                if let ms = r.compileLoadMs { print("  compile+load: \(String(format: "%.1f", ms)) ms") }
                if let ms = r.evalMedianMs { print("  eval median: \(String(format: "%.3f", ms)) ms") }
                if let e = r.maxRelErr { print("  max_rel_err: \(String(format: "%.4f", e))") }
                for n in r.notes { print("  note: \(n)") }
                print("  -> results/ane_spike.json")
                return r.go ? 0 : 1
            } catch {
                print("spike failed: \(error)")
                return 1
            }
        case "ane":
            do {
                let iters = intFlag(flags, "iters", 100)
                let subsets = strs(flags["subset"], default: [])
                let machine = Fingerprint.probe()
                let section = try ANESuite.run(subsets: subsets, iters: iters)
                printTable(section)
                try writeReport(makeReport(machine: machine, sections: [section],
                                           status: ["ane": section.runs.isEmpty ? "no-runs" : "done"]))
                return 0
            } catch {
                print("ane failed: \(error)")
                return 1
            }
        case "coreai":
            do {
                let iters = intFlag(flags, "iters", 50)
                let models = strs(flags["models"], default: [])
                let variants = strs(flags["variants"], default: ["default", "ane", "gpu"])
                let machine = Fingerprint.probe()
                let section = try await CoreAISuite.run(models: models, variants: variants, iters: iters)
                printTable(section)
                try writeReport(makeReport(machine: machine, sections: [section], status: ["coreai": "done"]))
                return 0
            } catch {
                print("coreai failed: \(error)")
                return 1
            }
        case "coreml":
            do {
                let iters = intFlag(flags, "iters", 100)
                let machine = Fingerprint.probe()
                let section = try await CoreMLSuite.run(device: device(), iters: iters)
                printTable(section)
                try writeReport(makeReport(machine: machine, sections: [section], status: ["coreml": "done"]))
                return 0
            } catch {
                print("coreml failed: \(error)")
                return 1
            }
        case "diff":
            guard positional.count >= 3 else {
                print("usage: bench diff A.json B.json")
                return 2
            }
            return runDiff(positional[1], positional[2])
        case "all":
            await runAllSuites(flags: flags)
            return 0
        default:
            usage()
            return 2
        }
    }

    @MainActor
    static func runAllSuites(flags: [String: String]) async {
        let iters = intFlag(flags, "iters", 30)
        let t0 = Date()
        let includeCoreML = flags["include-coreml"] != nil
        let machine = Fingerprint.probe()
        var sections: [SuiteSection] = []
        var status: [String: String] = [:]

        print("== silicon-ledger-bench full run ==")
        print("machine: \(machine.chip) | \(machine.model) | macOS \(machine.osVersion) \(machine.osBuild)")
        print("gpu: \(machine.gpuName) (\(machine.gpuCores) cores) | \(machine.memoryGB) GB | families: \(machine.metalFamilies.joined(separator: ","))")
        print("tensor dtypes: \(machine.metalTensor.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value ? 1 : 0)" }.joined(separator: " "))")

        // spike first (gates S3)
        do {
            let spike = try SpikeA.run(iters: 200)
            status["spike"] = spike.go ? "GO" : "NO-GO"
            print("\nspike: \(spike.go ? "GO" : "NO-GO") \(spike.evalMedianMs.map { String(format: "(eval %.3f ms)", $0) } ?? "")")
        } catch {
            status["spike"] = "ERROR \(error)"
            print("\nspike: ERROR \(error)")
        }

        do {
            let s = try MemorySuite.run(device: device(), iters: 15)
            sections.append(s)
            status["s1_memory"] = "done"
            printTable(s)
        } catch {
            status["s1_memory"] = "ERROR \(error)"
            print("\nmemory: ERROR \(error)")
        }

        do {
            let s = try GEMMSuite.run(device: device(),
                                      sizes: [512, 1024, 2048, 4096, 8192],
                                      dtypes: ["fp32", "fp16", "bf16", "int8", "fp8", "fp8e5m2", "fp4"],
                                      paths: ["mps", "msl", "tensorops"],
                                      iters: iters)
            sections.append(s)
            status["s2_gemm"] = "done"
            printTable(s)
        } catch {
            status["s2_gemm"] = "ERROR \(error)"
            print("\ngemm: ERROR \(error)")
        }

        if status["spike"] == "GO" {
            do {
                let s = try ANESuite.run(subsets: [], iters: 100)
                sections.append(s)
                status["s3_ane"] = "done"
                printTable(s)
            } catch {
                status["s3_ane"] = "ERROR \(error)"
                print("\nane: ERROR \(error)")
            }
        } else {
            var s = SuiteSection(id: "s3_ane", title: "ANE raw throughput (direct _ANEInMemoryModel private API)")
            s.notes.append("NOT RUN: spike NO-GO (see results/ane_spike.json)")
            sections.append(s)
            status["s3_ane"] = "skipped (spike NO-GO)"
            print("\ns3: skipped (spike NO-GO)")
        }

        do {
            let s = try await CoreAISuite.run(models: [], variants: ["default", "ane", "gpu"], iters: 50)
            sections.append(s)
            status["s4_coreai"] = "done"
            printTable(s)
        } catch {
            status["s4_coreai"] = "ERROR \(error)"
            print("\ncoreai: ERROR \(error)")
        }

        if includeCoreML {
            do {
                let s = try await CoreMLSuite.run(device: device(), iters: 100)
                sections.append(s)
                status["s5_coreml"] = "done"
                printTable(s)
            } catch {
                status["s5_coreml"] = "ERROR \(error)"
                print("\ncoreml: ERROR \(error)")
            }
        } else {
            status["s5_coreml"] = "off (use --include-coreml)"
        }

        do {
            let report = makeReport(machine: machine, sections: sections, status: status)
            let url = try JSONReport.write(report)
            let secs = Date().timeIntervalSince(t0)
            print("\n== summary ==")
            for (k, v) in status.sorted(by: { $0.key < $1.key }) { print("  \(k): \(v)") }
            print("  total wall time: \(String(format: "%.0f", secs)) s")
            print("report: \(url.path)")
        } catch {
            print("report write failed: \(error)")
        }
    }

    static func runDiff(_ aPath: String, _ bPath: String) -> Int32 {
        do {
            let a = try JSONReport.load(aPath)
            let b = try JSONReport.load(bPath)
            print("A: \(aPath)  (\(a.machine.chip), \(a.generated))")
            print("B: \(bPath)  (\(b.machine.chip), \(b.generated))")
            if a.machine.fingerprint != b.machine.fingerprint {
                print("warning: machine fingerprints differ (A=\(a.machine.fingerprint) B=\(b.machine.fingerprint))")
            }
            for sa in a.sections {
                guard let sb = b.sections.first(where: { $0.id == sa.id }) else { continue }
                print("\n== \(sa.id) ==")
                print("  " + padL("name", 32) + " " + padL("variant", 18) + " " + padL("unit", 8) +
                      " " + padR("A median", 12) + " " + padR("B median", 12) + " " + padR("delta%", 9))
                func disp(_ r: RunResult) -> Double { r.unit == "GB/s" ? (r.derived["GBs"] ?? 0) : r.median }
                for ra in sa.runs {
                    guard let rb = sb.runs.first(where: { $0.name == ra.name && $0.variant == ra.variant }) else {
                        print("  " + padL(ra.name, 32) + " " + padL(ra.variant, 18) + " " + padL(ra.unit, 8) +
                              " " + padR(String(format: "%.4f", disp(ra)), 12) + " " + padR("-", 12) + " " + padR("-", 9))
                        continue
                    }
                    if ra.runStatus != "ok" || rb.runStatus != "ok" {
                        print("  " + padL(ra.name, 32) + " " + padL(ra.variant, 18) + " " + padL(ra.unit, 8) +
                              " " + padR(ra.runStatus, 12) + " " + padR(rb.runStatus, 12) + " " + padR("n/a", 9))
                        continue
                    }
                    let va = disp(ra), vb = disp(rb)
                    let delta = va > 0 ? (vb - va) / va * 100 : 0
                    print("  " + padL(ra.name, 32) + " " + padL(ra.variant, 18) + " " + padL(ra.unit, 8) +
                          " " + padR(String(format: "%.4f", va), 12) + " " + padR(String(format: "%.4f", vb), 12) +
                          " " + padR(String(format: "%+.1f%%", delta), 9))
                }
                let onlyA = sa.runs.filter { ra in sb.runs.allSatisfy { $0.name != ra.name || $0.variant != ra.variant } }.map { "\($0.name)/\($0.variant)" }
                let onlyB = sb.runs.filter { rb in sa.runs.allSatisfy { $0.name != rb.name || $0.variant != rb.variant } }.map { "\($0.name)/\($0.variant)" }
                if !onlyA.isEmpty { print("  only in A: \(onlyA.joined(separator: ", "))") }
                if !onlyB.isEmpty { print("  only in B: \(onlyB.joined(separator: ", "))") }
            }
            return 0
        } catch {
            print("diff failed: \(error)")
            return 1
        }
    }

    @discardableResult
    static func runProbe() -> Int32 {
        let m = Fingerprint.probe()
        print("== machine fingerprint ==")
        print("  fingerprint : \(m.fingerprint)")
        print("  host        : \(m.host) (\(m.model))")
        print("  chip        : \(m.chip)")
        print("  os          : macOS \(m.osVersion) (\(m.osBuild))")
        print("  memory      : \(m.memoryGB) GB")
        print("  cpu cores   : \(m.cpuCores.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: " "))")
        print("  gpu         : \(m.gpuName) (\(m.gpuCores) cores)")
        print("  metal fams  : \(m.metalFamilies.joined(separator: ", "))")
        print("  tensor dt   : \(m.metalTensor.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value ? "y" : "n")" }.joined(separator: " "))")
        print("  ane         : \(m.ane.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value ? "y" : "n")" }.joined(separator: " "))")
        print("  tools       : \(m.tools.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " "))")
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let out = (try? enc.encode(m)) ?? Data()
        print("\n" + String(data: out, encoding: .utf8)!)
        return 0
    }
}
