// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "HornlabMetalBemNative",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "HornlabMetalBemNative",
            targets: ["HornlabMetalBemNative"]
        )
    ],
    targets: [
        .executableTarget(
            name: "HornlabMetalBemNative",
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        )
    ]
)
