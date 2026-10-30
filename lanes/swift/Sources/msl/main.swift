import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

#if canImport(FoundationModels)
    import FoundationModels
    import MLXFoundationModels
#endif

// Minimal arg parsing (no swift-argument-parser dependency).
struct Args {
    var model = ""
    var lane = ""
    var maxTokens = 128
    var runs = 3
    var wiredBytes = 0
    var prefillSizes = [32, 512, 2048]
    var l2Temp = 0.0
    var out = ""
}

func parseArgs() -> Args {
    var a = Args()
    let argv = CommandLine.arguments
    var i = 1
    while i < argv.count {
        switch argv[i] {
        case "--model": a.model = argv[i + 1]; i += 1
        case "--lane": a.lane = argv[i + 1]; i += 1
        case "--max-tokens": a.maxTokens = Int(argv[i + 1]) ?? 128; i += 1
        case "--runs": a.runs = Int(argv[i + 1]) ?? 3; i += 1
        case "--wired-bytes": a.wiredBytes = Int(argv[i + 1]) ?? 0; i += 1
        case "--l2-temp": a.l2Temp = Double(argv[i + 1]) ?? 0.0; i += 1
        case "--out": a.out = argv[i + 1]; i += 1
        default: break
        }
        i += 1
    }
    return a
}

func fjson(_ v: Any) -> String {
    if let d = v as? Double { return String(format: "%.3f", d) }
    if let s = v as? String { return "\"\(s)\"" }
    return "\(v)"
}

func promptOfLength(_ n: Int) -> String {
    let unit = "The quick brown fox jumps over the lazy dog. "
    var s = ""
    while s.count < n * 8 { s += unit }
    return s
}

// Chat templates add tokens around the user text, so tune the body until the
// fully-templated prompt encodes to exactly n tokens.
func promptForTokens(container: ModelContainer, target n: Int) async throws -> String {
    let s = promptOfLength(n)
    var overhead = 11
    do {
        let cal = try await container.prepare(input: UserInput(prompt: "Hello."))
        let params = GenerateParameters(maxTokens: 1, temperature: 0.0)
        let st = try await container.generate(input: cal, parameters: params)
        for await g in st {
            if case .info(let i) = g {
                let enc = try await container.tokenizer.encode(text: "Hello.")
                overhead = max(0, i.promptTokenCount - enc.count)
            }
        }
    }
    var prompt = String(s.prefix(max(0, (n - overhead) * 5)))
    for _ in 0..<16 {
        let enc = try await container.tokenizer.encode(text: prompt)
        let delta = (n - overhead) - enc.count
        if delta == 0 { break }
        if delta > 0 {
            prompt += String(s.prefix(delta * 5))
        } else {
            prompt = String(prompt.prefix(prompt.count + delta * 5))
        }
    }
    return prompt
}

func wallDecode(container: ModelContainer, maxTokens: Int, runs: Int, wiredBytes: Int) async throws
    -> [String: Any]
{
    var out: [String: Any] = [:]
    var runWall: [[String: Any]] = []
    var allStepMs: [Double] = []
    let prompt = "What is the weather like today?"

    for run in 0...runs {  // 1 warmup + N measured
        let lmInput = try await container.prepare(input: UserInput(prompt: prompt))
        let params = GenerateParameters(maxTokens: maxTokens, temperature: 0.0)
        var ticket: WiredMemoryTicket?
        if wiredBytes > 0 {
            ticket = WiredMaxPolicy().ticket(size: wiredBytes)
        }
        let start = Date()
        var prev = start
        var chunkMs: [Double] = []
        var nTok = 0
        var info: GenerateCompletionInfo?
        let stream = try await container.generate(
            input: lmInput, parameters: params, wiredMemoryTicket: ticket)
        for await gen in stream {
            switch gen {
            case .chunk(let s):
                let now = Date()
                if nTok > 0 { chunkMs.append(now.timeIntervalSince(prev) * 1000) }
                prev = now
                nTok += 1
                _ = s
            case .info(let i):
                info = i
            default:
                break
            }
        }
        let wallSec = Date().timeIntervalSince(start)
        if run > 0 {
            // count tokens precisely via the completion info when available
            let tokCount = info?.generationTokenCount ?? nTok
            runWall.append([
                "wall_tps": Double(tokCount) / wallSec,
                "internal_tps": info?.tokensPerSecond ?? 0,
                "tokens": tokCount,
            ])
            allStepMs.append(contentsOf: chunkMs)
        }
    }

    let sorted = allStepMs.sorted()
    let p50 = sorted.isEmpty ? 0.0 : sorted[sorted.count / 2]
    let p95 = sorted.isEmpty ? 0.0 : sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]
    let mean = allStepMs.isEmpty ? 0.0 : allStepMs.reduce(0, +) / Double(allStepMs.count)
    out["runs"] = runWall
    out["mean_step_ms"] = mean
    out["p50_step_ms"] = p50
    out["p95_step_ms"] = p95
    out["mean_decode_tps_wall"] = mean > 0 ? 1000.0 / mean : 0
    return out
}

func measuredTokens(_ container: ModelContainer, _ prompt: String) async throws -> Int {
    let lmInput = try await container.prepare(input: UserInput(prompt: prompt))
    let params = GenerateParameters(maxTokens: 1, temperature: 0.0)
    let stream = try await container.generate(input: lmInput, parameters: params)
    for await gen in stream {
        if case .info(let i) = gen { return i.promptTokenCount }
    }
    return 0
}

func wallPrefill(container: ModelContainer, sizes: [Int]) async throws -> [String: Any] {
    var out: [String: Any] = [:]
    for n in sizes {
        var prompt = try await promptForTokens(container: container, target: n)
        let pool = "wxyz 0123"
        var promptTok = n
        for it in 0..<48 {  // char-level fixup against MEASURED token count
            promptTok = try await measuredTokens(container, prompt)
            if promptTok == n { break }
            if promptTok < n {
                prompt.append(pool[pool.index(pool.startIndex, offsetBy: it % pool.count)])
            } else {
                prompt = String(prompt.dropLast())
            }
        }
        var best = Double.infinity
        for _ in 0..<4 {  // 1 warmup + 3 measured
            let lmInput = try await container.prepare(input: UserInput(prompt: prompt))
            let params = GenerateParameters(maxTokens: 1, temperature: 0.0)
            let start = Date()
            let stream = try await container.generate(input: lmInput, parameters: params)
            for await gen in stream {
                if case .info(let i) = gen { promptTok = i.promptTokenCount }
            }
            let sec = Date().timeIntervalSince(start)
            if sec < best { best = sec }
        }
        out["\(n)"] = [
            "ttfb_s": best,
            "prompt_tokens": promptTok,
            "ttfb_tps": best > 0 ? Double(promptTok) / best : 0,
        ]
    }
    return out
}

func emitRow(_ row: [String: Any], out: String) throws {
    let data = try JSONSerialization.data(
        withJSONObject: row, options: [.prettyPrinted, .sortedKeys])
    let s = String(data: data, encoding: .utf8) ?? "{}"
    if !out.isEmpty {
        try data.write(to: URL(fileURLWithPath: out))
    }
    print(s)
}


#if canImport(FoundationModels)
@available(macOS 27.0, *)
func fmDecode(modelID: String, container: ModelContainer, maxTokens: Int, runs: Int,
              temperature: Double) async throws
    -> [String: Any]
{
    let mlm = MLXLanguageModel(
        configuration: ModelConfiguration(id: modelID),
        capabilities: [.reasoning],
        weightsLocation: { _ in
            URL(fileURLWithPath: NSString(string: "~/.cache/huggingface").expandingTildeInPath)
        },
        load: { configuration, _ in
            try await loadModelContainer(
                from: #hubDownloader(), using: #huggingFaceTokenizerLoader(),
                configuration: configuration)
        })
    let session = LanguageModelSession(model: mlm)
    let prompt = "What is the weather like today? Give a short answer."
    var runsOut: [[String: Any]] = []
    for run in 0...runs {
        let t0 = Date()
        var resp = try await session.respond(
            to: prompt,
            options: GenerationOptions(temperature: temperature, maximumResponseTokens: maxTokens))
        var wall = Date().timeIntervalSince(t0)
        var retries = 0
        while resp.content.isEmpty && retries < 5 && run > 0 {
            // FM sessions intermittently emit an immediate-EOS empty
            // completion (LF-13); retry rather than pollute the cell.
            let t1 = Date()
            resp = try await session.respond(
                to: prompt,
                options: GenerationOptions(temperature: temperature,
                                           maximumResponseTokens: maxTokens))
            wall = Date().timeIntervalSince(t1)
            retries += 1
        }
        let ids = await container.tokenizer.encode(text: resp.content)
        FileHandle.standardError.write("l2 run=\(run) wall=\(wall) chars=\(resp.content.count) retries=\(retries) content=\(resp.content.prefix(40))\n".data(using: .utf8)!)
        if run > 0 {
            runsOut.append([
                "wall_s": wall,
                "tokens_retokenized": ids.count,
                "wall_tps": Double(ids.count) / max(wall, 1e-9),
                "chars": resp.content.count,
                "retries": retries,
            ])
        }
    }
    return [
        "runs": runsOut,
        "note":
            "non-streaming respond(); tokens via re-tokenizing output; session prompt overhead included",
    ]
}
#endif

let args = parseArgs()
let version = "swift-mlm 3.32.3 / mlx-swift 0.32.3"

Task {
    do {
        let container = try await loadModelContainer(
            from: #hubDownloader(), using: #huggingFaceTokenizerLoader(), id: args.model)

        var row: [String: Any] = [
            "lane": args.lane,
            "model": args.model,
            "build_mode": "release",
            "version": version,
            "wired_bytes": args.wiredBytes,
            "max_tokens": args.maxTokens,
        ]

        switch args.lane {
        case "l1":
            row["decode"] = try await wallDecode(
                container: container, maxTokens: args.maxTokens, runs: args.runs,
                wiredBytes: args.wiredBytes)
            row["prefill"] = try await wallPrefill(container: container, sizes: args.prefillSizes)
#if canImport(FoundationModels)
        case "l2":
            if #available(macOS 27.0, *) {
                row["decode"] = try await fmDecode(
                    modelID: args.model, container: container, maxTokens: args.maxTokens,
                    runs: args.runs, temperature: args.l2Temp)
                row["l2_temperature"] = args.l2Temp
            } else {
                row["error"] = "macOS 27 required"
            }
#endif
        default:
            break
        }

        row["peak_memory_bytes"] = Memory.peakMemory
        row["peak_memory_GB"] = Double(Memory.peakMemory) / 1e9
        try emitRow(row, out: args.out)
        exit(0)
    } catch {
        FileHandle.standardError.write("msl error: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}

dispatchMain()
