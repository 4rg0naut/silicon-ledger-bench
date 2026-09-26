import Foundation
import CryptoKit

struct RunResult: Codable {
    var name: String
    var variant: String
    var config: [String: String]
    var unit: String
    var iters: Int
    var median: Double
    var mean: Double
    var p95: Double
    var min: Double
    var max: Double
    var derived: [String: Double]
    var status: String?
    var error: String?

    var runStatus: String { status ?? "ok" }

    static func make(name: String, variant: String, config: [String: String], unit: String,
                     stats: RunStats, derived: [String: Double] = [:],
                     status: String = "ok", error: String? = nil) -> RunResult {
        RunResult(name: name, variant: variant, config: config, unit: unit,
                  iters: stats.iters, median: stats.median, mean: stats.mean,
                  p95: stats.p95, min: stats.min, max: stats.max, derived: derived,
                  status: status, error: error)
    }

    static func failed(name: String, variant: String, config: [String: String], unit: String,
                       status: String, error: String, derived: [String: Double] = [:]) -> RunResult {
        RunResult(name: name, variant: variant, config: config, unit: unit,
                  iters: 0, median: 0, mean: 0, p95: 0, min: 0, max: 0, derived: derived,
                  status: status, error: error)
    }
}

struct SuiteSection: Codable {
    var id: String
    var title: String
    var runs: [RunResult]
    var notes: [String]

    init(id: String, title: String) {
        self.id = id
        self.title = title
        self.runs = []
        self.notes = []
    }
}

struct MachineInfo: Codable {
    var fingerprint: String
    var host: String
    var model: String
    var chip: String
    var osVersion: String
    var osBuild: String
    var memoryGB: Int
    var cpuCores: [String: Int]
    var gpuName: String
    var gpuCores: Int
    var metalFamilies: [String]
    var metalTensor: [String: Bool]
    var ane: [String: Bool]
    var tools: [String: String]
}

struct ReportV1: Codable {
    var schema: String
    var generated: String
    var machine: MachineInfo
    var sections: [SuiteSection]
    var status: [String: String]

    static let schemaID = "silicon-ledger/1"
}

enum JSONReport {
    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    static func write(_ report: ReportV1, dir: URL = resultsDir) throws -> URL {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let path = dir.appendingPathComponent("\(stamp)-\(report.machine.fingerprint).json")
        try encoder.encode(report).write(to: path, options: .atomic)
        return path
    }

    static func load(_ path: String) throws -> ReportV1 {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let report = try decoder.decode(ReportV1.self, from: data)
        if report.schema == "m5max-bench/1" {
            try? FileHandle.standardError.write("note: legacy schema id '\(report.schema)' in \(path); fields identical to \(ReportV1.schemaID)\n".data(using: .utf8)!)
        }
        return report
    }

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    static var resultsDir: URL {
        let fm = FileManager.default
        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        let dir = cwd.appendingPathComponent("results")
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

func sha256Hex(_ data: Data) -> String {
    let digest = SHA256.hash(data: data)
    return digest.map { String(format: "%02x", $0) }.joined()
}

func sha256Hex(_ s: String) -> String {
    sha256Hex(Data(s.utf8))
}
