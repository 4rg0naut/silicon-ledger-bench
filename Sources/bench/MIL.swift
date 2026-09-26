import Foundation
import ANEBridge

struct MILConvChain {
    let ch: Int
    let sp: Int
    let depth: Int
    let int8: Bool
    let fp16IO: Bool
    var perFile: Bool = false

    var weightBytes: Int { ch * ch * (int8 ? 1 : 2) * depth }

    func milText() -> String {
        if int8 { return milTextInt8() }
        var m = "program(1.3)\n"
        m += "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"
        let ioType = fp16IO ? "fp16" : "fp32"
        m += "    func main<ios18>(tensor<\(ioType), [1, \(ch), 1, \(sp)]> x) {\n"
        m += """
        string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
        tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
        tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
        int32 c_groups = const()[name = string("c_groups"), val = int32(1)];

        """
        if !fp16IO {
            m += "        string x_to_fp16 = const()[name = string(\"x_to_fp16\"), val = string(\"fp16\")];\n"
            m += "        tensor<fp16, [1, \(ch), 1, \(sp)]> h0 = cast(dtype = x_to_fp16, x = x)[name = string(\"cast_in\")];\n"
        }
        var prev = fp16IO ? "x" : "h0"
        let cs = 64 + ch * ch * 2
        for i in 0..<depth {
            let wpath = perFile ? "@model_path/weights/w\(i).bin" : "@model_path/weights/weight.bin"
            let woff = perFile ? 64 : 64 + i * cs
            m += "        tensor<fp16, [\(ch), \(ch), 1, 1]> W\(i) = const()[name = string(\"W\(i)\"), val = tensor<fp16, [\(ch), \(ch), 1, 1]>(BLOBFILE(path = string(\"\(wpath)\"), offset = uint64(\(woff))))];\n"
            let out = "c\(i)"
            m += "        tensor<fp16, [1, \(ch), 1, \(sp)]> \(out) = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W\(i), x = \(prev))[name = string(\"\(out)\")];\n"
            prev = out
        }
        m += "        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"
        m += "        tensor<fp32, [1, \(ch), 1, \(sp)]> o = cast(dtype = to_fp32, x = \(prev))[name = string(\"cast_out\")];\n"
        m += "    } -> (o);\n}\n"
        return m
    }

    func milTextInt8() -> String {
        let offs = int8ChunkOffsets()
        var m = "program(1.0)\n"
        m += "[buildInfo = dict<tensor<string, []>, tensor<string, []>>({{\"coremlc-component-MIL\", \"3600.16.1\"}, {\"coremlc-version\", \"3600.25.1\"}})]\n{\n"
        m += "    func main<ios16>(tensor<fp32, [1, \(ch), 1, \(sp)]> x) {\n"
        for i in 0..<depth {
            let p = "c\(i)"
            m += "        tensor<string, []> \(p)_pt = const()[name = tensor<string, []>(\"\(p)_pt\"), val = tensor<string, []>(\"valid\")];\n"
            m += "        tensor<int32, [2]> \(p)_st = const()[name = tensor<string, []>(\"\(p)_st\"), val = tensor<int32, [2]>([1, 1])];\n"
            m += "        tensor<int32, [4]> \(p)_pd = const()[name = tensor<string, []>(\"\(p)_pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
            m += "        tensor<int32, [2]> \(p)_di = const()[name = tensor<string, []>(\"\(p)_di\"), val = tensor<int32, [2]>([1, 1])];\n"
            m += "        tensor<int32, []> \(p)_gr = const()[name = tensor<string, []>(\"\(p)_gr\"), val = tensor<int32, []>(1)];\n"
            let o = offs[i]
            m += "        tensor<fp16, [\(ch), \(ch), 1, 1]> W\(i) = constexpr_affine_dequantize()[axis = tensor<int32, []>(0), name = tensor<string, []>(\"W\(i)\"), quantized_data = tensor<int8, [\(ch), \(ch), 1, 1]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>(\(o.w)))), scale = tensor<fp16, [\(ch)]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>(\(o.scale)))), zero_point = tensor<int8, [\(ch)]>(BLOBFILE(path = tensor<string, []>(\"@model_path/weights/weight.bin\"), offset = tensor<uint64, []>(\(o.zp))))];\n"
            if i == 0 {
                m += "        tensor<string, []> x_to_fp16_dtype_0 = const()[name = tensor<string, []>(\"x_to_fp16_dtype_0\"), val = tensor<string, []>(\"fp16\")];\n"
                m += "        tensor<fp16, [1, \(ch), 1, \(sp)]> x_to_fp16 = cast(dtype = x_to_fp16_dtype_0, x = x)[name = tensor<string, []>(\"cast_1\")];\n"
            }
            let prev = i == 0 ? "x_to_fp16" : "c\(i-1)"
            m += "        tensor<fp16, [1, \(ch), 1, \(sp)]> c\(i) = conv(dilations = \(p)_di, groups = \(p)_gr, pad = \(p)_pd, pad_type = \(p)_pt, strides = \(p)_st, weight = W\(i), x = \(prev))[name = tensor<string, []>(\"c\(i)\")];\n"
        }
        m += "        tensor<string, []> to_fp32 = const()[name = tensor<string, []>(\"to_fp32\"), val = tensor<string, []>(\"fp32\")];\n"
        m += "        tensor<fp32, [1, \(ch), 1, \(sp)]> o = cast(dtype = to_fp32, x = c\(depth-1))[name = tensor<string, []>(\"cast_0\")];\n"
        m += "    } -> (o);\n}\n"
        return m
    }

    struct Int8Offsets { var w: Int; var zp: Int; var scale: Int }

    func int8ChunkOffsets() -> [Int8Offsets] {
        let wB = ch * ch
        let zB = ch
        let sB = ch * 2
        func pad(_ b: Int) -> Int { b % 64 == 0 ? b : ((b + 63) / 64) * 64 }
        func region(_ b: Int, last: Bool) -> Int { 64 + (last || b % 64 == 0 ? b : pad(b)) }
        var off = 64
        var out = [Int8Offsets]()
        out.reserveCapacity(depth)
        for i in 0..<depth {
            out.append(Int8Offsets(w: off, zp: off + region(wB, last: false), scale: off + region(wB, last: false) + region(zB, last: false)))
            off += region(wB, last: false) + region(zB, last: false) + region(sB, last: i == depth - 1)
        }
        return out
    }

    /// Per-layer symmetric int8 quantization: scale[n] = max_c |w[n,c]| / 127.
    /// Returns (quantized int8 weights, fp32 scales, dequantized fp32 weights for CPU reference).
    func int8QuantizedWeights() -> (q: [Int8], scales: [Float], eff: [Float]) {
        let w = deterministicWeights()
        var q = [Int8](repeating: 0, count: w.count)
        var scales = [Float](repeating: 1, count: ch * depth)
        var eff = [Float](repeating: 0, count: w.count)
        for i in 0..<depth {
            for n in 0..<ch {
                var m: Float = 0
                for c in 0..<ch {
                    let a = abs(w[(i * ch + n) * ch + c])
                    if a > m { m = a }
                }
                let s = m / 127.0
                if s == 0 { scales[i * ch + n] = 1; continue }
                scales[i * ch + n] = s
                for c in 0..<ch {
                    let v = w[(i * ch + n) * ch + c] / s
                    var qi = Int(v.rounded())
                    if qi > 127 { qi = 127 }
                    if qi < -128 { qi = -128 }
                    q[(i * ch + n) * ch + c] = Int8(qi)
                    eff[(i * ch + n) * ch + c] = Float(qi) * s
                }
            }
        }
        return (q, scales, eff)
    }

    /// Deterministic weights (per layer, row-major [n][c]) with small values.
    func deterministicWeights() -> [Float] {
        var w = [Float]()
        w.reserveCapacity(ch * ch * depth)
        for i in 0..<depth {
            for n in 0..<ch {
                for c in 0..<ch {
                    let v = Float(((n * 31 + c * 17 + i * 7) % 13) - 6) * 0.01
                    w.append(v)
                }
            }
        }
        return w
    }

    /// Deterministic input [c][s].
    func deterministicInput() -> [Float] {
        var x = [Float]()
        x.reserveCapacity(ch * sp)
        for c in 0..<ch {
            for s in 0..<sp {
                x.append(Float(((c * 13 + s * 7) % 11) - 5) * 0.01)
            }
        }
        return x
    }

    func buildWeightFiles() -> [(name: String, data: Data)] {
        let w = deterministicWeights()
        var out: [(String, Data)] = []
        for i in 0..<depth {
            let slice = Array(w[i * ch * ch..<(i + 1) * ch * ch])
            var len: Int = 0
            let buf = slice.withUnsafeBufferPointer { p in
                ane_bridge_build_fp16_weight_file(p.baseAddress!, Int32(ch), Int32(ch), &len)!
            }
            defer { ane_bridge_free_blob(buf) }
            out.append(("w\(i).bin", Data(bytes: buf, count: len)))
        }
        return out
    }

    func buildWeightBlob() -> (Data, Int) {
        let w = deterministicWeights()
        var len: Int = 0
        if int8 {
            let (q, scales, _) = int8QuantizedWeights()
            let blob = q.withUnsafeBufferPointer { qp -> UnsafeMutablePointer<UInt8> in
                scales.withUnsafeBufferPointer { sp in
                    ane_bridge_build_int8_weight_blob(qp.baseAddress!, sp.baseAddress!, Int32(ch), Int32(ch), Int32(depth), &len)!
                }
            }
            defer { ane_bridge_free_blob(blob) }
            return (Data(bytes: blob, count: len), len)
        }
        let blob = w.withUnsafeBufferPointer { p -> UnsafeMutablePointer<UInt8> in
            ane_bridge_build_fp16_weight_blob(p.baseAddress!, Int32(ch), Int32(ch), Int32(depth), &len)!
        }
        defer { ane_bridge_free_blob(blob) }
        return (Data(bytes: blob, count: len), len)
    }
}

/// CPU reference for a single 1x1 conv layer: out[n][s] = sum_c W[n][c] * x[c][s]
func cpuRefConv1x1(w: [Float], x: [Float], ch: Int, sp: Int) -> [Float] {
    var out = [Double](repeating: 0, count: ch * sp)
    for n in 0..<ch {
        for s in 0..<sp {
            var acc: Double = 0
            for c in 0..<ch {
                acc += Double(w[n * ch + c]) * Double(x[c * sp + s])
            }
            out[n * sp + s] = acc
        }
    }
    return out.map { Float($0) }
}

/// CPU reference for a chain of 1x1 conv layers (w: depth x ch x ch, row-major).
func cpuRefConvChain1x1(w: [Float], x: [Float], ch: Int, sp: Int, depth: Int) -> [Float] {
    var cur = x
    for i in 0..<depth {
        let li = i * ch * ch
        cur = cpuRefConv1x1(w: Array(w[li..<(li + ch * ch)]), x: cur, ch: ch, sp: sp)
    }
    return cur
}
