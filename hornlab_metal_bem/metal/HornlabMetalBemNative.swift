#!/usr/bin/env swift

import Foundation

let backendDir = URL(fileURLWithPath: CommandLine.arguments[0])
    .deletingLastPathComponent()
let packageDir = backendDir.appendingPathComponent("native_helper")
let swiftExecutable = ProcessInfo.processInfo.environment["HORNLAB_METAL_BEM_SWIFT"] ?? "swift"

let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
process.arguments = [
    swiftExecutable,
    "run",
    "--package-path",
    packageDir.path,
    "HornlabMetalBemNative",
] + Array(CommandLine.arguments.dropFirst())

// Let the child inherit stdout/stderr directly. Buffering through Pipe()
// and draining only after waitUntilExit() deadlocks once the child writes
// more than the pipe buffer (e.g. swift build progress or a long error).

do {
    try process.run()
    process.waitUntilExit()
    exit(process.terminationStatus)
} catch {
    FileHandle.standardError.write(Data("failed to run packaged native helper: \(error)\n".utf8))
    exit(1)
}
