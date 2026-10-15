import Foundation

/// Per-run environment snapshot — *"the window"*.
///
/// Two machines never have identical running processes (different daemons, GUI
/// state, background jobs), so a cross-machine comparison must declare what was
/// running on each side and let the reader judge. This block is that declaration:
/// it is embedded in every run record and printed by `bench state`.
///
/// Read-only, non-root, and cheap (one or two seconds — a handful of short-lived
/// CLI reads, no benchmark work). It deliberately lives beside `Fingerprint.swift`
/// and uses the same plain-Codable style.
///
/// The full rule is in REPRODUCE.md ("The run window").
struct WindowSnapshot: Codable {
    /// One logged-in session, as reported by `who`. A GUI `console` session
    /// changes GPU/CPU behaviour, so its presence must be visible in the record.
    struct Session: Codable {
        var user: String
        var tty: String
        var login: String
    }

    /// One process in the top-CPU list.
    struct Process: Codable {
        var pid: Int
        var cpu: Double
        var command: String
    }

    /// One power assertion (`pmset -g assertions`), i.e. something holding the
    /// machine awake during the run.
    struct Assertion: Codable {
        var pid: Int
        var process: String
        var type: String
        var name: String
    }

    /// GUI-presence flags. `headless == true` means no WindowServer at all;
    /// a Mac mini with a live desktop is NOT headless and shows up here.
    struct GUI: Codable {
        var windowServer: Bool
        var dock: Bool
        var finder: Bool
        var wallpaper: Bool
        var loginwindow: Bool
        var controlCenter: Bool
        var headless: Bool
    }

    var captured: String          // ISO-8601 UTC timestamp of this snapshot
    var bootTime: String          // ISO-8601 UTC boot instant
    var uptimeSeconds: Double     // seconds since boot
    var loadAverage: [Double]     // [1m, 5m, 15m] at capture
    var sessions: [Session]       // `who` output-equivalent
    var consoleSession: Bool      // a GUI/console login exists
    var gui: GUI                  // WindowServer / dock / wallpaper presence
    var topCPU: [Process]         // top 10 by %CPU (cumulative since process start)
    var watchlist: [String]       // known contention suspects currently running
    var sleepAssertions: [Assertion]  // who is holding the machine awake
    var preventSleepCount: Int    // system-wide Prevent* assertion count
    var powerSource: String       // "AC Power" / "Battery Power"
    var thermalState: String      // nominal / fair / serious / critical
    var fingerprint: String       // owning machine fingerprint — never orphaned from its host
    var host: String              // owning machine host name
}

enum WindowCapture {
    /// Processes whose mere presence is worth flagging: Apple background ML /
    /// media / telemetry / sync work and remote-session daemons that content for
    /// the same silicon. Presence is not guilt — activity is (see `topCPU`).
    static let watchNames: Set<String> = [
        "oMLX", "omlx-server", "distilld", "mediaanalysisd", "photolibraryd",
        "photoanalysisd", "cloudd", "bird", "backupd", "mds", "mds_stores",
        "mdworker", "mdworker_shared", "spotlightknowledged", "corespeechd",
        "assistantd", "siri", "intelligenceflowd", "modelcatalogd",
        "knowledgeconstructiond", "aned", "ANECompilerService", "powerlogd",
        "thermald", "screensharingd", "ScreensharingAgent", "VTEncoderXPCService",
        "avconferenced", "coreaudiod", "backupd-helper", "cloudd"
    ]

    static func capture(machine: MachineInfo) -> WindowSnapshot {
        let now = Date()
        let boot = bootInstant()
        let procs = parsePS(tool("/bin/ps", ["-Ao", "pid=,pcpu=,comm=", "-r"]) ?? "")
        let names = Set(procs.map { ($0.command as NSString).lastPathComponent })

        var loads = [Double](repeating: 0, count: 3)
        _ = getloadavg(&loads, 3)

        let sessions = parseWho(tool("/usr/bin/who", []) ?? "")

        var gui = WindowSnapshot.GUI(
            windowServer: names.contains("WindowServer"),
            dock: names.contains("Dock"),
            finder: names.contains("Finder"),
            wallpaper: names.contains("WallpaperAgent") || names.contains("Wallpaper"),
            loginwindow: names.contains("loginwindow"),
            controlCenter: names.contains("ControlCenter"),
            headless: false)
        gui.headless = !gui.windowServer

        let assertionsRaw = tool("/usr/bin/pmset", ["-g", "assertions"]) ?? ""
        let (assertions, preventCount) = parseAssertions(assertionsRaw)

        let detector = ISO8601DateFormatter()
        return WindowSnapshot(
            captured: detector.string(from: now),
            bootTime: boot.map { detector.string(from: $0) } ?? "",
            uptimeSeconds: ProcessInfo.processInfo.systemUptime,
            loadAverage: loads,
            sessions: sessions,
            consoleSession: sessions.contains { $0.tty == "console" },
            gui: gui,
            topCPU: Array(procs.prefix(10)),
            watchlist: watchNames.intersection(names).sorted(),
            sleepAssertions: assertions,
            preventSleepCount: preventCount,
            powerSource: parsePowerSource(tool("/usr/bin/pmset", ["-g", "batt"]) ?? ""),
            thermalState: thermalString(ProcessInfo.processInfo.thermalState),
            fingerprint: machine.fingerprint,
            host: machine.host
        )
    }

    // MARK: - capture helpers

    /// Run a read-only CLI with a C locale so decimal separators are dots
    /// (e.g. `ps` on a fr_FR box otherwise prints "9,1").
    static func tool(_ path: String, _ args: [String]) -> String? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        var env = ProcessInfo.processInfo.environment
        env["LC_ALL"] = "C"
        env["LANG"] = "C"
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = Pipe()
        do {
            try p.run()
            p.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let s = String(data: data, encoding: .utf8) else { return nil }
            return s.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch { return nil }
    }

    static func bootInstant() -> Date? {
        var tv = timeval()
        var size = MemoryLayout<timeval>.size
        guard sysctlbyname("kern.boottime", &tv, &size, nil, 0) == 0 else { return nil }
        return Date(timeIntervalSince1970: Double(tv.tv_sec) + Double(tv.tv_usec) / 1_000_000)
    }

    /// Parse `ps -Ao pid=,pcpu=,comm=` (already CPU-sorted by `-r`).
    static func parsePS(_ out: String) -> [WindowSnapshot.Process] {
        var result: [WindowSnapshot.Process] = []
        for line in out.split(separator: "\n") {
            let fields = line.split(separator: " ", maxSplits: 2, omittingEmptySubsequences: true)
            guard fields.count >= 3,
                  let pid = Int(fields[0]),
                  let cpu = Double(fields[1].replacingOccurrences(of: ",", with: ".")) else { continue }
            result.append(WindowSnapshot.Process(pid: pid, cpu: cpu,
                                                 command: fields[2].trimmingCharacters(in: .whitespaces)))
        }
        return result
    }

    /// Parse `who`: user, tty/console, login time (remainder).
    static func parseWho(_ out: String) -> [WindowSnapshot.Session] {
        var result: [WindowSnapshot.Session] = []
        for line in out.split(separator: "\n") {
            let fields = line.split(separator: " ", omittingEmptySubsequences: true)
            guard fields.count >= 3 else { continue }
            result.append(WindowSnapshot.Session(user: String(fields[0]),
                                                 tty: String(fields[1]),
                                                 login: fields[2...].joined(separator: " ")))
        }
        return result
    }

    /// Parse `pmset -g assertions`: sum the system-wide Prevent* counters and
    /// list each holding assertion by pid/process/type/name.
    static func parseAssertions(_ out: String) -> ([WindowSnapshot.Assertion], Int) {
        var assertions: [WindowSnapshot.Assertion] = []
        var preventCount = 0
        var inSystemWide = false
        for raw in out.split(separator: "\n") {
            let t = raw.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("Assertion status system-wide:") { inSystemWide = true; continue }
            if t.hasPrefix("Listed by owning process") { inSystemWide = false; continue }
            if inSystemWide {
                let parts = t.split(separator: " ", omittingEmptySubsequences: true)
                if parts.count == 2, parts[0].contains("Prevent"), let n = Int(parts[1]) {
                    preventCount += n
                }
                continue
            }
            guard t.hasPrefix("pid "), let namedR = t.range(of: " named: \"") else { continue }
            let afterPid = t.dropFirst(4)
            guard let open = afterPid.firstIndex(of: "("),
                  let pid = Int(afterPid[..<open]) else { continue }
            let rest = afterPid[afterPid.index(after: open)...]
            guard let close = rest.firstIndex(of: ")") else { continue }
            let process = String(rest[..<close])
            let type = t[..<namedR.lowerBound].split(separator: " ").last.map(String.init) ?? ""
            let nameStart = namedR.upperBound
            let nameEnd = t.range(of: "\"", options: .backwards)?.lowerBound ?? t.endIndex
            let name = nameStart <= nameEnd ? String(t[nameStart..<nameEnd]) : ""
            assertions.append(WindowSnapshot.Assertion(pid: pid, process: process, type: type, name: name))
        }
        return (assertions, preventCount)
    }

    /// `pmset -g batt` first line: `Now drawing from 'AC Power'`.
    static func parsePowerSource(_ out: String) -> String {
        guard let r = out.range(of: "'") else {
            return out.split(separator: "\n").first.map(String.init) ?? "unknown"
        }
        let after = out[r.upperBound...]
        guard let close = after.firstIndex(of: "'") else { return String(after) }
        return String(after[..<close])
    }

    static func thermalString(_ s: ProcessInfo.ThermalState) -> String {
        switch s {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}
