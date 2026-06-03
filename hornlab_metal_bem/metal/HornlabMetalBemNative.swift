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

let stdout = Pipe()
let stderr = Pipe()
process.standardOutput = stdout
process.standardError = stderr

do {
    try process.run()
    process.waitUntilExit()
    FileHandle.standardOutput.write(stdout.fileHandleForReading.readDataToEndOfFile())
    FileHandle.standardError.write(stderr.fileHandleForReading.readDataToEndOfFile())
    exit(process.terminationStatus)
} catch {
    FileHandle.standardError.write(Data("failed to run packaged native helper: \(error)\n".utf8))
    exit(1)
}
