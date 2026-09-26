import Foundation
import Metal

enum ShaderLib {
    static func mtlExt(_ vals: [Int]) -> MTLTensorExtents {
        vals.withUnsafeBufferPointer { p in MTLTensorExtents(__rank: p.count, values: p.baseAddress!)! }
    }

    static func source(_ name: String) -> String {
        guard let url = Bundle.module.url(forResource: name, withExtension: "metal", subdirectory: "Shaders"),
              let s = try? String(contentsOf: url, encoding: .utf8) else {
            fatalError("shader source missing: \(name).metal")
        }
        return s
    }

    static var mpsHeadersRoot: String {
        "/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers"
    }

    /// Textually expand `#include "..."` lines (relative to the including file),
    /// leaving angle-bracket includes (metal_numeric, __exec/units.h, ...) for
    /// the runtime compiler. Include guards make repeated inlining a no-op.
    static func expandIncludes(_ path: String, seen: inout Set<String>) -> String {
        if seen.contains(path) { return "" }
        seen.insert(path)
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return "" }
        var out = ""
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("#include"),
               let range = line.range(of: "\""),
               let end = line[line.index(after: range.lowerBound)...].range(of: "\"") {
                let inc = String(line[line.index(after: range.lowerBound)..<end.lowerBound])
                let dir = (path as NSString).deletingLastPathComponent
                out += expandIncludes((dir as NSString).appendingPathComponent(inc), seen: &seen)
                out += "\n"
                continue
            }
            out += line
            out += "\n"
        }
        return out
    }

    /// The Metal 4 runtime compiler exposes no header search path, so the MPS
    /// TensorOps header tree (Apple headers installed on this Mac) is inlined
    /// into one translation unit. The bench kernels are inserted just before
    /// the final `#pragma METAL internals : disable` so they sit inside the
    /// internals-enabled region.
    static func tensorOpsSource() -> String? {
        let root = mpsHeadersRoot
        guard FileManager.default.fileExists(atPath: root + "/MPPTensorOpsMatMul2d.h"),
              let header = try? String(contentsOfFile: root + "/MPPTensorOpsMatMul2d.h", encoding: .utf8),
              let kurl = Bundle.module.url(forResource: "tensorops", withExtension: "metal", subdirectory: "Shaders"),
              let kernels = try? String(contentsOf: kurl, encoding: .utf8) else {
            return nil
        }
        var seen = Set<String>()
        let inlined = expandIncludes(root + "/MPPTensorOpsMatMul2d.h", seen: &seen)
        _ = header
        // Append bench kernels at global scope (outside any `#pragma METAL
        // internals` region) so they are exported as library entry points.
        return "#include <metal_stdlib>\n#include <metal_packed_numeric>\nusing namespace metal;\n"
            + inlined + "\n" + kernels
    }

    @discardableResult
    static func makeLibrary(device: MTLDevice, file: String) throws -> MTLLibrary {
        let opts = MTLCompileOptions()
        if file == "tensorops" {
            // metal_fp8_*/metal_fp4_* tensor element types are only defined for
            // __METAL_VERSION__ >= 410 (GPUCompiler metal_config feature gates).
            opts.languageVersion = .version4_1
            guard let src = tensorOpsSource() else {
                throw NSError(domain: "ShaderLib", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "MPS TensorOps headers not found at \(mpsHeadersRoot)"])
            }
            return try device.makeLibrary(source: src, options: opts)
        }
        opts.languageVersion = .version3_1
        return try device.makeLibrary(source: source(file), options: opts)
    }
}
