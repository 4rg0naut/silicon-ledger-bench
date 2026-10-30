// swift-tools-version: 6.4
import PackageDescription

let package = Package(
    name: "msl",
    platforms: [.macOS(.v27)],
    products: [
        .executable(name: "msl", targets: ["msl"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", exact: "3.32.3"),
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.4.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.2.0"),
    ],
    targets: [
        .executableTarget(
            name: "msl",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXFoundationModels", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        )
    ]
)
