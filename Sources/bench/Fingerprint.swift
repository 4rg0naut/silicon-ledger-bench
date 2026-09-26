import Foundation
import Metal
import ANEBridge

func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
    let end = buf.firstIndex(of: 0) ?? buf.count
    return String(decoding: buf[0..<end].map { UInt8($0) }, as: UTF8.self)
}

func sysctlInt64(_ name: String) -> Int64? {
    var value: Int64 = 0
    var size = MemoryLayout<Int64>.size
    guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
    return value
}

func runTool(_ path: String, _ args: [String]) -> String? {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: path)
    p.arguments = args
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

func parseSystemProfiler() -> [String: String] {
    let out = runTool("/usr/sbin/system_profiler", ["SPHardwareDataType", "SPDisplaysDataType"]) ?? ""
    var result: [String: String] = [:]
    var inDisplays = false
    for line in out.split(separator: "\n") {
        if line.trimmingCharacters(in: .whitespaces).hasPrefix("Graphics/Displays:") {
            inDisplays = true
            continue
        }
        let parts = line.split(separator: ":", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count == 2 else { continue }
        let key = parts[0], val = parts[1]
        switch key {
        case "Model Name" where result["hostModel"] == nil: result["hostModel"] = val
        case "Chip" where result["chip"] == nil: result["chip"] = val
        case "Memory" where result["memory"] == nil: result["memory"] = val
        case "Total Number of Cores" where inDisplays && result["gpuCores"] == nil: result["gpuCores"] = val
        default: break
        }
    }
    return result
}

enum Fingerprint {
    static let tbBin = "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin"
    static let xdevBin = "/Applications/Xcode.app/Contents/Developer/usr/bin"

    static func probe() -> MachineInfo {
        let prof = parseSystemProfiler()
        let device = MTLCreateSystemDefaultDevice()
        let info = ProcessInfo.processInfo

        var gpuFamilies: [String] = []
        if let device {
            let fams: [(String, MTLGPUFamily)] = [
                ("apple1", .apple1), ("apple2", .apple2), ("apple3", .apple3),
                ("apple4", .apple4), ("apple5", .apple5), ("apple6", .apple6),
                ("apple7", .apple7), ("apple8", .apple8), ("apple9", .apple9),
                ("apple10", .apple10), ("apple11", .apple11)
            ]
            for (name, f) in fams where device.supportsFamily(f) {
                gpuFamilies.append(name)
            }
        }

        var tensorSupport: [String: Bool] = [:]
        if let device {
            let dtypes: [(String, MTLTensorDataType)] = [
                ("fp32", .float32), ("fp16", .float16), ("bf16", .bfloat16),
                ("int8", .int8), ("int4", .int4),
                ("fp8e4m3", .metalFloat8e4m3), ("fp8e5m2", .metalFloat8e5m2), ("fp4e2m1", .metalFloat4e2m1)
            ]
            for (name, dt) in dtypes {
                let desc = MTLTensorDescriptor()
                desc.dimensions = ShaderLib.mtlExt([32])
                desc.dataType = dt
                tensorSupport[name] = (try? device.makeTensor(descriptor: desc)) != nil
            }
        }

        let aneProbe = ane_bridge_probe()
        var ane: [String: Bool] = [
            "framework_loadable": (aneProbe & 1) != 0,
            "inmemory_classes": (aneProbe & 2) != 0,
            "request_class": (aneProbe & 4) != 0,
            "aned_daemon": FileManager.default.fileExists(atPath: "/usr/libexec/aned"),
            "ane_compiler_xpc": FileManager.default.fileExists(
                atPath: "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/XPCServices/ANECompilerService.xpc"),
            "coreml_mlmodel": NSClassFromString("MLModel") != nil
        ]
        var coreaiLoaded = false
        if let h = dlopen("/System/Library/Frameworks/CoreAI.framework/CoreAI", RTLD_LAZY) {
            coreaiLoaded = true
            dlclose(h)
        }
        ane["coreai_framework"] = coreaiLoaded

        var tools: [String: String] = [:]
        if let v = runTool("\(tbBin)/swiftc", ["--version"]) { tools["swift_xcode"] = v.components(separatedBy: "\n").last ?? v }
        if let v = runTool("/usr/bin/swiftc", ["--version"]) { tools["swift_clt"] = v.components(separatedBy: "\n").last ?? v }
        tools["aimodelc"] = FileManager.default.fileExists(atPath: "\(xdevBin)/aimodelc") ? "present" : "missing"
        tools["coremlc"] = FileManager.default.fileExists(atPath: "\(xdevBin)/coremlc") ? "present" : "missing"
        if let v = runTool("/opt/homebrew/bin/python3.11", ["--version"]) { tools["python"] = v }
        if FileManager.default.fileExists(atPath: "tools/.venv/bin/pip") {
            if let v = runTool("tools/.venv/bin/pip", ["show", "coreai-torch"]),
               let line = v.split(separator: "\n").first(where: { $0.hasPrefix("Version:") }) {
                tools["coreai-torch"] = line.replacingOccurrences(of: "Version:", with: "").trimmingCharacters(in: .whitespaces)
            }
        }

        let osV = info.operatingSystemVersion
        let memoryBytes = sysctlInt64("hw.memsize") ?? 0
        let chip = prof["chip"] ?? sysctlString("machdep.cpu.brand_string") ?? "unknown"
        let model = prof["hostModel"] ?? sysctlString("hw.model") ?? "unknown"
        let gpuCores = Int(prof["gpuCores"] ?? "") ?? 0
        let osVersion = "\(osV.majorVersion).\(osV.minorVersion)"
        let osBuild = sysctlString("kern.osversion") ?? ""

        let fpInput = "\(chip)|\(memoryBytes)|\(device?.name ?? "")|\(gpuCores)|\(osVersion)|\(model)|\(gpuFamilies.joined(separator: ","))"
        let fp = String(sha256Hex(fpInput).prefix(16))

        return MachineInfo(
            fingerprint: fp,
            host: info.hostName,
            model: model,
            chip: chip,
            osVersion: osVersion,
            osBuild: osBuild,
            memoryGB: Int(memoryBytes / (1024 * 1024 * 1024)),
            cpuCores: [
                "total": Int(sysctlInt64("hw.physicalcpu") ?? 0),
                "perflevel0": Int(sysctlInt64("hw.perflevel0.physicalcpu") ?? 0),
                "perflevel1": Int(sysctlInt64("hw.perflevel1.physicalcpu") ?? 0)
            ],
            gpuName: device?.name ?? "unknown",
            gpuCores: gpuCores,
            metalFamilies: gpuFamilies,
            metalTensor: tensorSupport,
            ane: ane,
            tools: tools
        )
    }
}
