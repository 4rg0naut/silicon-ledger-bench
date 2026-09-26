// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "silicon-ledger-bench",
    platforms: [.macOS(.v27)],
    products: [
        .executable(name: "bench", targets: ["bench"])
    ],
    targets: [
        .target(
            name: "ANEBridge",
            publicHeadersPath: ".",
            cSettings: [
                .define("ANE_BRIDGE_VERSION", to: "1.0")
            ],
            linkerSettings: [
                .linkedFramework("Foundation"),
                .linkedFramework("CoreFoundation"),
                .linkedFramework("IOSurface"),
                .linkedLibrary("dl")
            ]
        ),
        .executableTarget(
            name: "bench",
            dependencies: ["ANEBridge"],
            resources: [.copy("Shaders")]
        )
    ]
)
