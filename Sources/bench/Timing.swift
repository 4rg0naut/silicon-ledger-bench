import Darwin
import Foundation

final class Stopwatch {
    private var startTicks: UInt64 = 0
    private static let timebase: mach_timebase_info_data_t = {
        var tb = mach_timebase_info_data_t()
        mach_timebase_info(&tb)
        return tb
    }()

    func start() { startTicks = mach_absolute_time() }

    func elapsedMs() -> Double {
        let t = mach_absolute_time()
        return Double(t - startTicks) * Double(Self.timebase.numer) / Double(Self.timebase.denom) / 1_000_000.0
    }
}

struct RunStats: Codable {
    var iters: Int
    var median: Double
    var mean: Double
    var p95: Double
    var min: Double
    var max: Double

    static func from(timesMs: [Double]) -> RunStats? {
        guard !timesMs.isEmpty else { return nil }
        let sorted = timesMs.sorted()
        let n = sorted.count
        let median = n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        let mean = sorted.reduce(0, +) / Double(n)
        let p95idx = Swift.min(n - 1, Int(Double(n) * 0.95))
        return RunStats(
            iters: n,
            median: median,
            mean: mean,
            p95: sorted[p95idx],
            min: sorted[0],
            max: sorted[n - 1]
        )
    }

    func spreadPct() -> Double {
        guard mean > 0 else { return 0 }
        return (max - min) / mean * 100.0
    }
}

enum Timed {
    /// Runs `work` `warmup` + `iters` times, timing each measured iteration.
    @discardableResult
    static func run(warmup: Int, iters: Int, work: () throws -> Void) rethrows -> [Double] {
        let sw = Stopwatch()
        var times: [Double] = []
        times.reserveCapacity(iters)
        for _ in 0..<warmup { try work() }
        for _ in 0..<iters {
            sw.start()
            try work()
            times.append(sw.elapsedMs())
        }
        return times
    }

    /// Warmup until `work` has run for at least `warmupMs` of wall time (DVFS
    /// steady state), then measure `iters` iterations.
    @discardableResult
    static func runFor(warmupMs: Double, iters: Int, work: () throws -> Void) rethrows -> [Double] {
        let sw = Stopwatch()
        var times: [Double] = []
        times.reserveCapacity(iters)
        sw.start()
        repeat { try work() } while sw.elapsedMs() < warmupMs
        for _ in 0..<iters {
            sw.start()
            try work()
            times.append(sw.elapsedMs())
        }
        return times
    }

    @discardableResult
    static func runAsync(warmup: Int, iters: Int, work: () async throws -> Void) async rethrows -> [Double] {
        let sw = Stopwatch()
        var times: [Double] = []
        times.reserveCapacity(iters)
        for _ in 0..<warmup { try await work() }
        for _ in 0..<iters {
            sw.start()
            try await work()
            times.append(sw.elapsedMs())
        }
        return times
    }
}
