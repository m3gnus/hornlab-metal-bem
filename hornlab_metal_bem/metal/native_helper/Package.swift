// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "HornlabMetalBemNative",
    platforms: [
        .macOS("13.3") // Accelerate new LAPACK requires macOS 13.3
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
            swiftSettings: [
                // Select Accelerate's current LAPACK ($NEWLAPACK symbols). The
                // legacy CLAPACK entry points return NaN under concurrent calls.
                .unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK"])
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        )
    ]
)
