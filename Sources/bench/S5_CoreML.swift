import Foundation
import CoreML

/// Legacy Core ML baseline (off by default, --include-coreml).
/// Loads the same deterministic conv chain built by tools/gen_coreml.py
/// (coremltools -> mlprogram fp16) and runs MLModel with the neural-engine
/// preferred configuration, as a reference point vs the Core AI stack.
struct CoreMLSuite {
    static let modelsDir = URL(fileURLWithPath: "models/coreml")
    static let modelName = "conv_512x64_d8"
    static let ch = 512
    static let sp = 64

    static func run(device _: MTLDevice?, iters: Int) async throws -> SuiteSection {
        var section = SuiteSection(id: "s5_coreml", title: "Core ML (legacy) ANE-preferred baseline")
        section.notes.append("Legacy Core ML path (MLModel, neuralEngine preferred). Same deterministic conv chain as S3/S4 (512ch, 1x1, depth 8), built by tools/gen_coreml.py via coremltools (mlprogram, fp16 weights).")

        let pkg = modelsDir.appendingPathComponent("\(modelName).mlpackage")
        if !FileManager.default.fileExists(atPath: pkg.path) {
            let gen = try await runGenerator()
            section.notes.append("mlpackage missing — generator ran: \(gen)")
        }
        guard FileManager.default.fileExists(atPath: pkg.path) else {
            section.notes.append("NOT RUN: generator did not produce \(pkg.path)")
            return section
        }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        let model: MLModel
        do {
            let compiled = try await MLModel.compileModel(at: pkg)
            model = try await MLModel.load(contentsOf: compiled, configuration: cfg)
        } catch {
            section.notes.append("NOT RUN: MLModel load failed: \(error)")
            return section
        }

        let input = deterministicInput()

        func makeProvider() throws -> MLDictionaryFeatureProvider {
            let arr = try MLMultiArray(shape: [1, NSNumber(value: ch), 1, NSNumber(value: sp)], dataType: .float16)
            let f16 = input.map { Float16($0) }
            f16.withUnsafeBufferPointer { src in
                let dst = arr.dataPointer.bindMemory(to: Float16.self, capacity: ch * sp)
                dst.update(from: src.baseAddress!, count: ch * sp)
            }
            return try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: arr)])
        }

        guard let ref = loadRef(section: &section) else { return section }

        var maxRel: Double = 0, normRel: Double = 0, cos = 0.0, refMax = 0.0
        var dot = 0.0, na = 0.0, nb = 0.0
        do {
            let provider = try makeProvider()
            let result = try await model.prediction(from: provider)
            let outName = Array(model.modelDescription.outputDescriptionsByName.keys).first ?? "out"
            guard let outArr = result.featureValue(for: outName)?.multiArrayValue,
                  outArr.count == ch * sp else {
                section.notes.append("unexpected output (outputs: \(Array(model.modelDescription.outputDescriptionsByName.keys)))")
                return section
            }
            var raw = [Float](repeating: 0, count: ch * sp)
            try readMultiArray(outArr, into: &raw)
            for i in 0..<ch * sp {
                let got = Double(raw[i]), exp = Double(ref[i])
                let d = abs(got - exp)
                maxRel = max(maxRel, d / max(abs(exp), 1e-30))
                refMax = max(refMax, abs(exp))
                dot += got * exp; na += got * got; nb += exp * exp
            }
            normRel = refMax > 0 ? (zip(raw, ref).map { abs(Double($0) - Double($1)) }.max() ?? 0) / refMax : 0
            cos = (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 0
        } catch {
            section.notes.append("NOT RUN: prediction failed: \(error)")
            return section
        }
        let validated = normRel <= 0.15 && cos >= 0.999
        section.notes.append("validated vs fp64 torch ref: normalized_rel<=\(String(format: "%.3e", normRel)) (tol 0.15) cosine=\(String(format: "%.6f", cos)) (tol 0.999) pointwise_rel=\(String(format: "%.3e", maxRel)) ok=\(validated)")
        section.notes.append("fp16 mlprogram end-to-end (I/O cast to fp16); pointwise error large at near-zero outputs, same acceptance rule as S4 deep_fp16.")

        let st: RunStats
        do {
            let times = try await Timed.runAsync(warmup: 3, iters: iters) {
                _ = try await model.prediction(from: makeProvider())
            }
            guard let s = RunStats.from(timesMs: times) else {
                section.notes.append("no timed iters (skipped)")
                return section
            }
            st = s
        } catch {
            section.notes.append("timed run failed: \(error)")
            return section
        }
        let tops = 2.0 * Double(ch) * Double(ch) * Double(sp) * Double(8) / 1e12 / (st.median / 1000.0)
        section.runs.append(.make(
            name: "coreml_\(modelName)", variant: "cpuAndNeuralEngine",
            config: ["ch": "\(ch)", "sp": "\(sp)", "depth": "8", "validated": validated ? "yes" : "no"],
            unit: "ms/run", stats: st,
            derived: ["TOPS": tops, "pointwise_rel_err": maxRel, "cosine_sim": cos]))
        return section
    }

    static func loadRef(section: inout SuiteSection) -> [Float]? {
        let p = modelsDir.appendingPathComponent("\(modelName).ref.bin")
        guard let rd = try? Data(contentsOf: p) else {
            section.notes.append("missing ref.bin")
            return nil
        }
        var vals = [Float](repeating: 0, count: rd.count / 4)
        vals.withUnsafeMutableBytes { vp in
            rd.withUnsafeBytes { raw in vp.baseAddress!.copyMemory(from: raw.baseAddress!, byteCount: rd.count) }
        }
        return vals
    }

    static func readMultiArray(_ arr: MLMultiArray, into out: inout [Float]) throws {
        let n = min(arr.count, out.count)
        if arr.dataType == .float16 {
            arr.withUnsafeBufferPointer(ofType: Float16.self) { (buf: UnsafeBufferPointer<Float16>) in
                for i in 0..<n { out[i] = Float(buf[i]) }
            }
        } else {
            arr.withUnsafeBufferPointer(ofType: Float.self) { (buf: UnsafeBufferPointer<Float>) in
                for i in 0..<n { out[i] = buf[i] }
            }
        }
    }

    static func deterministicInput() -> [Float] {
        var x = [Float](repeating: 0, count: ch * sp)
        for c in 0..<ch {
            for s in 0..<sp {
                x[c * sp + s] = Float(Double((c * 13 + s * 7) % 11 - 5) * 0.01)
            }
        }
        return x
    }

    static func runGenerator() async throws -> String {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("tools/.venv/bin/python")
        p.arguments = ["tools/gen_coreml.py"]
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        try p.run()
        p.waitUntilExit()
        let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return "exit=\(p.terminationStatus) tail=\(out.split(separator: "\n").suffix(2).joined(separator: " | "))"
    }
}
