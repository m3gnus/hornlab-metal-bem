// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "HornlabSolverMetalNative",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "HornlabSolverMetalNative",
            targets: ["HornlabSolverMetalNative"]
        )
    ],
    targets: [
        .executableTarget(
            name: "HornlabSolverMetalNative",
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        )
    ]
)
