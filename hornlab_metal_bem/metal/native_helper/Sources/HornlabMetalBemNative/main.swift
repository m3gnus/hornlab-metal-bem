#!/usr/bin/env swift

import Foundation
import Metal
import Accelerate

let schema = "hornlab.metal.standard.v1"

struct ContractError: Error, CustomStringConvertible {
    let description: String
}

func fail(_ message: String) throws -> Never {
    throw ContractError(description: message)
}

func loadJSON(_ path: String) throws -> [String: Any] {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        try fail("JSON root must be an object")
    }
    return object
}

func writeJSON(_ path: String, _ object: [String: Any]) throws {
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    let data = try JSONSerialization.data(
        withJSONObject: object,
        options: [.prettyPrinted, .sortedKeys]
    )
    // Atomic write (temp file + rename) so a reader polling for streamed
    // per-case results never observes a partially written manifest.
    try data.write(to: url, options: .atomic)
}

func descriptorPath(root: String, descriptor: [String: Any]) throws -> String {
    return URL(fileURLWithPath: root)
        .appendingPathComponent(try requireString(descriptor, "path"))
        .path
}

func readF32(_ path: String, expectedCount: Int) throws -> [Float] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    if data.count != expectedCount * MemoryLayout<Float>.stride {
        try fail("float32 byte count mismatch for \(path)")
    }
    return data.withUnsafeBytes { rawBuffer in
        Array(rawBuffer.bindMemory(to: Float.self))
    }
}

func readI32(_ path: String, expectedCount: Int) throws -> [Int32] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    if data.count != expectedCount * MemoryLayout<Int32>.stride {
        try fail("int32 byte count mismatch for \(path)")
    }
    return data.withUnsafeBytes { rawBuffer in
        Array(rawBuffer.bindMemory(to: Int32.self))
    }
}

func writeF32(_ path: String, _ values: [Float]) throws {
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    var output = values
    let data = Data(bytes: &output, count: output.count * MemoryLayout<Float>.stride)
    try data.write(to: url)
}

func requireInt(_ object: [String: Any], _ key: String) throws -> Int {
    if let value = object[key] as? Int {
        return value
    }
    if let value = object[key] as? NSNumber {
        return value.intValue
    }
    try fail("\(key) must be an integer")
}

func requireDouble(_ object: [String: Any], _ key: String) throws -> Double {
    if let value = object[key] as? NSNumber {
        return value.doubleValue
    }
    try fail("\(key) must be a number")
}

func optionalDouble(_ object: [String: Any], _ key: String, default defaultValue: Double) throws -> Double {
    guard object[key] != nil else {
        return defaultValue
    }
    return try requireDouble(object, key)
}

func requireString(_ object: [String: Any], _ key: String) throws -> String {
    guard let value = object[key] as? String else {
        try fail("\(key) must be a string")
    }
    return value
}

func requireObject(_ object: [String: Any], _ key: String) throws -> [String: Any] {
    guard let value = object[key] as? [String: Any] else {
        try fail("\(key) must be an object")
    }
    return value
}

func optionalIntArray(_ object: [String: Any], _ key: String) throws -> [Int]? {
    guard let raw = object[key] else {
        return nil
    }
    guard let values = raw as? [Any] else {
        try fail("\(key) must be an integer array")
    }
    return try values.map { value in
        if let intValue = value as? Int {
            return intValue
        }
        if let numberValue = value as? NSNumber {
            return numberValue.intValue
        }
        try fail("\(key) must contain only integers")
    }
}

func requireShape(_ object: [String: Any], _ key: String) throws -> [Int] {
    guard let raw = object[key] as? [Any], !raw.isEmpty else {
        try fail("\(key) must be a non-empty shape array")
    }
    let shape = raw.map { value -> Int in
        if let intValue = value as? Int {
            return intValue
        }
        if let numberValue = value as? NSNumber {
            return numberValue.intValue
        }
        return -1
    }
    if shape.contains(where: { $0 <= 0 }) {
        try fail("\(key) must contain only positive dimensions")
    }
    return shape
}

@discardableResult
func validateDescriptor(
    _ descriptor: [String: Any],
    name: String,
    dtype: String,
    shape expectedShape: [Int]? = nil,
    rank expectedRank: Int? = nil
) throws -> [Int] {
    let path = try requireString(descriptor, "path")
    if path.hasPrefix("/") || path.split(separator: "/").contains("..") {
        try fail("\(name).path must be relative and must not contain '..'")
    }
    let shape = try requireShape(descriptor, "shape")
    if let expectedRank, shape.count != expectedRank {
        try fail("\(name).shape must have rank \(expectedRank)")
    }
    if let expectedShape, shape != expectedShape {
        try fail("\(name).shape must be \(expectedShape), got \(shape)")
    }
    if try requireString(descriptor, "dtype") != dtype {
        try fail("\(name).dtype must be \(dtype)")
    }
    if try requireString(descriptor, "byte_order") != "little" {
        try fail("\(name).byte_order must be little")
    }
    if try requireString(descriptor, "order") != "C" {
        try fail("\(name).order must be C")
    }
    return shape
}

func validateSession(_ manifest: [String: Any]) throws -> [String: Any] {
    if try requireString(manifest, "schema") != schema {
        try fail("unsupported schema")
    }
    if try requireString(manifest, "op") != "create_session" {
        try fail("expected create_session op")
    }
    if try requireInt(manifest, "index_base") != 0 {
        try fail("expected index_base=0")
    }
    if try requireString(manifest, "matrix_layout") != "row_major_c" {
        try fail("expected row_major_c matrix layout")
    }

    let mesh = try requireObject(manifest, "mesh")
    let verticesShape = try validateDescriptor(
        try requireObject(mesh, "vertices_f32"),
        name: "mesh.vertices_f32",
        dtype: "float32",
        rank: 2
    )
    let trianglesShape = try validateDescriptor(
        try requireObject(mesh, "triangles_i32"),
        name: "mesh.triangles_i32",
        dtype: "int32",
        rank: 2
    )
    if verticesShape[0] != 3 {
        try fail("mesh.vertices_f32 must have shape [3, n_vertices]")
    }
    if trianglesShape[0] != 3 {
        try fail("mesh.triangles_i32 must have shape [3, n_triangles]")
    }

    let nTriangles = trianglesShape[1]
    try validateDescriptor(
        try requireObject(mesh, "physical_tags_i32"),
        name: "mesh.physical_tags_i32",
        dtype: "int32",
        shape: [nTriangles]
    )
    try validateDescriptor(
        try requireObject(mesh, "p1_local2global_i32"),
        name: "mesh.p1_local2global_i32",
        dtype: "int32",
        shape: [nTriangles, 3]
    )
    try validateDescriptor(
        try requireObject(mesh, "triangle_areas_f32"),
        name: "mesh.triangle_areas_f32",
        dtype: "float32",
        shape: [nTriangles]
    )
    try validateDescriptor(
        try requireObject(mesh, "triangle_normals_3xm_f32"),
        name: "mesh.triangle_normals_3xm_f32",
        dtype: "float32",
        shape: [3, nTriangles]
    )

    let space = try requireObject(manifest, "space")
    let p1DofCount = try requireInt(space, "p1_dof_count")
    let dp0DofCount = try requireInt(space, "dp0_dof_count")
    if p1DofCount <= 0 {
        try fail("space.p1_dof_count must be positive")
    }
    if dp0DofCount != nTriangles {
        try fail("space.dp0_dof_count must equal n_triangles")
    }
    let symmetryPlane = try parseSymmetryPlane(manifest)

    return [
        "schema": schema,
        "op": "validate_session_result",
        "implementation": "swift_native_contract_probe",
        "session_id": try requireString(manifest, "session_id"),
        "n_vertices": verticesShape[1],
        "n_triangles": nTriangles,
        "p1_dof_count": p1DofCount,
        "dp0_dof_count": dp0DofCount,
        "symmetry_plane": symmetryPlane.map { $0 as Any } ?? NSNull(),
        "status": "ok",
    ]
}

func parseSymmetryPlane(_ manifest: [String: Any]) throws -> String? {
    let scope = manifest["assembly_scope"] as? [String: Any]
    guard let rawPlane = scope?["symmetry_plane"], !(rawPlane is NSNull) else {
        return nil
    }
    guard let plane = rawPlane as? String else {
        try fail("assembly_scope.symmetry_plane must be null or a string")
    }
    if plane != "yz" && plane != "xz" && plane != "xy" && plane != "yz+xz" {
        try fail("native symmetry currently supports yz, xz, xy, and yz+xz")
    }
    return plane
}

struct Complex32 {
    var re: Float
    var im: Float

    static let zero = Complex32(re: 0, im: 0)

    static func + (lhs: Complex32, rhs: Complex32) -> Complex32 {
        Complex32(re: lhs.re + rhs.re, im: lhs.im + rhs.im)
    }

    static func - (lhs: Complex32, rhs: Complex32) -> Complex32 {
        Complex32(re: lhs.re - rhs.re, im: lhs.im - rhs.im)
    }

    static func * (lhs: Complex32, rhs: Complex32) -> Complex32 {
        Complex32(
            re: lhs.re * rhs.re - lhs.im * rhs.im,
            im: lhs.re * rhs.im + lhs.im * rhs.re
        )
    }

    static func * (lhs: Complex32, rhs: Float) -> Complex32 {
        Complex32(re: lhs.re * rhs, im: lhs.im * rhs)
    }
}

struct AssemblyArrays {
    let aRe: [Float]
    let aIm: [Float]
    let rhsRe: [Float]
    let rhsIm: [Float]
}

struct AssemblyRun {
    let arrays: AssemblyArrays
    let implementation: String
    let mode: String
    let seconds: Double
    let parity: [String: Any]?
    let duffyStats: DuffyCorrectionStats?
    let nearStats: NearQuadratureStats?
    let metalDispatch: [String: Any]?
}

struct FieldRun {
    let values: [Complex32]
    let implementation: String
    let mode: String
    let seconds: Double
    let parity: [String: Any]?
    let metalDispatch: [String: Any]?
}

struct MetalAssemblyOutput {
    let arrays: AssemblyArrays
    let dispatch: [String: Any]
}

struct MetalFieldOutput {
    let values: [Complex32]
    let dispatch: [String: Any]
    // GPU execution time plus readback, excluding command-queue wait. Set by
    // the resident path so pipelined batches (where the field command buffer
    // queues behind the next case's assembly) still report field cost, not
    // queue latency.
    var gpuSeconds: Double? = nil
}

struct MetalDuffyBlockOutput {
    let slpRe: [Float]
    let slpIm: [Float]
    let dlpRe: [Float]
    let dlpIm: [Float]
    let dispatch: [String: Any]
}

struct DenseSolveRun {
    let pressure: [Complex32]
    let implementation: String
    let seconds: Double
    let lapackInfo: Int32
    // Reciprocal 1-norm condition estimate from cgecon on the LU factors;
    // nil when the factorization or the estimator failed. Lets interior-
    // resonance spikes in sweeps be attributed to ill conditioning.
    let rcond: Double?
    // Mixed-precision iterative refinement bookkeeping; nil when disabled.
    // Refinement corrects LU/rounding error against the float32 operator
    // only — float32 assembly and quadrature error survive it.
    var refineIterations: Int? = nil
    var refineResidualRel: Double? = nil
    // Precision of the dense factor/solve. "float32" (default) is the historical
    // Complex32 LU; "float64" factors/solves the float32-assembled system in
    // complex128 (zgesv) and narrows the result back to f32. The default keeps
    // every existing cgesv/cgetrf constructor reporting "float32" automatically.
    var dtype: String = "float32"
    // Relative residual of the CHIEF interior null-field constraint rows from the
    // overdetermined least-squares (zgels) solve: ||scale*(C*p - d)||_2 / ||b||_2.
    // nil for the plain square LU/zgesv paths (no CHIEF rows). Near the f64 floor
    // means the boundary solution already satisfies the interior constraint; an
    // O(1) value means CHIEF is actively correcting a fictitious resonance there
    // (or the points are badly placed).
    var chiefResidualRel: Double? = nil
}

func matrixOneNorm(_ matrix: inout [__CLPK_complex], n: Int) -> __CLPK_real {
    var normChar = Int8(49) // "1"
    var mClpk = __CLPK_integer(n)
    var nClpk = __CLPK_integer(n)
    var lda = __CLPK_integer(n)
    var work = [__CLPK_real(0)] // unused for the 1-norm
    return __CLPK_real(clange_(&normChar, &mClpk, &nClpk, &matrix, &lda, &work))
}

/// 1-norm condition estimate via cgecon on an LU-factorized matrix.
/// `anorm` must be the 1-norm of the original matrix, computed before the
/// factorization overwrote it.
func estimateReciprocalCondition(
    factored: inout [__CLPK_complex],
    n: Int,
    anorm: __CLPK_real
) -> Double? {
    var normChar = Int8(49) // "1"
    var nClpk = __CLPK_integer(n)
    var lda = __CLPK_integer(n)
    var anormValue = anorm
    var rcond = __CLPK_real(0)
    var info = __CLPK_integer(0)
    var work = Array(repeating: __CLPK_complex(r: 0.0, i: 0.0), count: 2 * n)
    var rwork = Array(repeating: __CLPK_real(0), count: 2 * n)
    cgecon_(
        &normChar,
        &nClpk,
        &factored,
        &lda,
        &anormValue,
        &rcond,
        &work,
        &rwork,
        &info
    )
    if info != 0 {
        return nil
    }
    return Double(rcond)
}

/// complex128 twin of `matrixOneNorm` (zlange). The work array is
/// `[__CLPK_doublereal]` (Double), not doublecomplex — zlange's `work` is real,
/// matching the float32 path's `[__CLPK_real]` work.
func matrixOneNormZ(_ matrix: inout [__CLPK_doublecomplex], n: Int) -> Double {
    var normChar = Int8(49) // "1"
    var mClpk = __CLPK_integer(n)
    var nClpk = __CLPK_integer(n)
    var lda = __CLPK_integer(n)
    var work = [Double(0)] // unused for the 1-norm
    return Double(zlange_(&normChar, &mClpk, &nClpk, &matrix, &lda, &work))
}

/// complex128 twin of `estimateReciprocalCondition` (zgecon on LU factors).
/// `anorm` must be the 1-norm of the original matrix, computed before the
/// factorization overwrote it.
func estimateReciprocalConditionZ(
    factored: inout [__CLPK_doublecomplex],
    n: Int,
    anorm: Double
) -> Double? {
    var normChar = Int8(49) // "1"
    var nClpk = __CLPK_integer(n)
    var lda = __CLPK_integer(n)
    var anormValue = anorm
    var rcond = Double(0)
    var info = __CLPK_integer(0)
    var work = Array(repeating: __CLPK_doublecomplex(r: 0.0, i: 0.0), count: 2 * n)
    var rwork = Array(repeating: Double(0), count: 2 * n)
    zgecon_(
        &normChar,
        &nClpk,
        &factored,
        &lda,
        &anormValue,
        &rcond,
        &work,
        &rwork,
        &info
    )
    if info != 0 {
        return nil
    }
    return Double(rcond)
}

struct DuffyPairPlan {
    let coincident: Int
    let edge: Int
    let vertex: Int

    var total: Int {
        coincident + edge + vertex
    }

    func toJSON() -> [String: Any] {
        [
            "coincident": coincident,
            "edge": edge,
            "vertex": vertex,
            "total": total,
        ]
    }
}

struct DuffyPair {
    let test: Int
    let trial: Int
    let testImageMask: Int
    let trialImageMask: Int
    let kind: Int
    let testLocal1: Int
    let testLocal2: Int
    let trialLocal1: Int
    let trialLocal2: Int
}

struct DuffyPairList {
    let pairs: [DuffyPair]
    let plan: DuffyPairPlan
}

struct NearPair {
    let test: Int
    let trial: Int
    let testImageMask: Int
    let trialImageMask: Int
}

struct NearPairList {
    let pairs: [NearPair]
}

struct DuffyReductionPlan {
    let pairTrialTriangles: [Int]
    let rhsRows: [Int]
    let rowWeights: [Float]
    let dlpSlots: [Int]
    let matrixIndices: [Int]
    let imagePairs: Int
}

struct DuffyCorrectionStats {
    let plan: DuffyPairPlan
    let rawTriplets: Int
    let uniqueTriplets: Int
    let seconds: Double
    let implementation: String
    let blockSeconds: Double?
    let reductionSeconds: Double?
    let dispatch: [String: Any]?
    let imagePairs: Int
    let reductionPrecomputed: Bool
    let reductionPlanBuildSeconds: Double?

    func toJSON() -> [String: Any] {
        var payload: [String: Any] = [
            "implemented": true,
            "scope": "matrix_and_rhs_duffy_delta",
            "implementation": implementation,
            "planned_pairs": plan.toJSON(),
            "raw_triplets_if_expanded": rawTriplets,
            "unique_triplets": uniqueTriplets,
            "correction_seconds": seconds,
        ]
        if let blockSeconds {
            payload["block_seconds"] = blockSeconds
        }
        if let reductionSeconds {
            payload["reduction_seconds"] = reductionSeconds
        }
        if reductionPrecomputed {
            payload["reduction_precomputed"] = true
        }
        if let reductionPlanBuildSeconds {
            payload["reduction_plan_build_seconds"] = reductionPlanBuildSeconds
        }
        if let dispatch {
            payload["metal_dispatch"] = dispatch
        }
        if imagePairs > 0 {
            payload["image_adjacent_pairs"] = imagePairs
            payload["image_singular_correction"] = true
        }
        return payload
    }
}

struct NearQuadratureConfig {
    let level: Int
    let threshold: Double
}

struct NearQuadratureStats {
    let level: Int
    let threshold: Double
    let pairCount: Int
    let seconds: Double

    func toJSON() -> [String: Any] {
        [
            "level": level,
            "threshold": threshold,
            "pair_count": pairCount,
            "seconds": seconds,
        ]
    }
}

struct Geometry {
    let root: String
    let manifest: [String: Any]
    let px: [Float]
    let py: [Float]
    let pz: [Float]
    let triangles: [Int32]
    let physicalTags: [Int32]
    let p1Local2Global: [Int32]
    let areas: [Float]
    let normals: [Float]
    let nVertices: Int
    let nTriangles: Int
    let p1DofCount: Int
    let dp0DofCount: Int
    let symmetryPlane: String?

    func triangleVertex(_ triangle: Int, _ local: Int) -> Int {
        Int(triangles[local * nTriangles + triangle])
    }

    func p1Dof(_ triangle: Int, _ local: Int) -> Int {
        Int(p1Local2Global[triangle * 3 + local])
    }

    func normal(_ triangle: Int, _ component: Int) -> Float {
        normals[component * nTriangles + triangle]
    }

    var symmetryPlaneCode: Int32 {
        if symmetryPlane == "yz" {
            return 1
        }
        if symmetryPlane == "xz" {
            return 2
        }
        if symmetryPlane == "xy" {
            return 4
        }
        if symmetryPlane == "yz+xz" {
            return 3
        }
        return 0
    }

    func symmetryRowWeight(_ row: Int) -> Float {
        if symmetryPlaneCode == 0 {
            return 1.0
        }
        var weight: Float = 1.0
        if symmetryPlaneCode & 1 != 0 {
            weight *= 2.0
        }
        if symmetryPlaneCode & 2 != 0 {
            weight *= 2.0
        }
        if symmetryPlaneCode & 4 != 0 {
            weight *= 2.0
        }
        return weight
    }

}

func readGeometry(_ sessionManifestPath: String) throws -> Geometry {
    let manifest = try loadJSON(sessionManifestPath)
    _ = try validateSession(manifest)
    let root = URL(fileURLWithPath: sessionManifestPath).deletingLastPathComponent().path
    let mesh = try requireObject(manifest, "mesh")
    let space = try requireObject(manifest, "space")

    let verticesDesc = try requireObject(mesh, "vertices_f32")
    let verticesShape = try requireShape(verticesDesc, "shape")
    let nVertices = verticesShape[1]
    let vertices = try readF32(
        try descriptorPath(root: root, descriptor: verticesDesc),
        expectedCount: 3 * nVertices
    )

    let trianglesDesc = try requireObject(mesh, "triangles_i32")
    let trianglesShape = try requireShape(trianglesDesc, "shape")
    let nTriangles = trianglesShape[1]
    let triangles = try readI32(
        try descriptorPath(root: root, descriptor: trianglesDesc),
        expectedCount: 3 * nTriangles
    )

    let physicalTagsDesc = try requireObject(mesh, "physical_tags_i32")
    let physicalTags = try readI32(
        try descriptorPath(root: root, descriptor: physicalTagsDesc),
        expectedCount: nTriangles
    )

    let localDesc = try requireObject(mesh, "p1_local2global_i32")
    let p1Local2Global = try readI32(
        try descriptorPath(root: root, descriptor: localDesc),
        expectedCount: 3 * nTriangles
    )

    let areasDesc = try requireObject(mesh, "triangle_areas_f32")
    let areas = try readF32(
        try descriptorPath(root: root, descriptor: areasDesc),
        expectedCount: nTriangles
    )

    let normalsDesc = try requireObject(mesh, "triangle_normals_3xm_f32")
    let normals = try readF32(
        try descriptorPath(root: root, descriptor: normalsDesc),
        expectedCount: 3 * nTriangles
    )
    let symmetryPlane = try parseSymmetryPlane(manifest)

    return Geometry(
        root: root,
        manifest: manifest,
        px: Array(vertices[0..<nVertices]),
        py: Array(vertices[nVertices..<(2 * nVertices)]),
        pz: Array(vertices[(2 * nVertices)..<(3 * nVertices)]),
        triangles: triangles,
        physicalTags: physicalTags,
        p1Local2Global: p1Local2Global,
        areas: areas,
        normals: normals,
        nVertices: nVertices,
        nTriangles: nTriangles,
        p1DofCount: try requireInt(space, "p1_dof_count"),
        dp0DofCount: try requireInt(space, "dp0_dof_count"),
        symmetryPlane: symmetryPlane
    )
}

func averageSurfacePressureForTag(
    geom: Geometry,
    pressure: [Complex32],
    tag: Int
) -> (re: Double, im: Double, area: Double) {
    var sumRe = 0.0
    var sumIm = 0.0
    var totalArea = 0.0
    for tri in 0..<geom.nTriangles where Int(geom.physicalTags[tri]) == tag {
        let area = Double(geom.areas[tri])
        if area <= 0.0 {
            continue
        }
        let dof0 = geom.p1Dof(tri, 0)
        let dof1 = geom.p1Dof(tri, 1)
        let dof2 = geom.p1Dof(tri, 2)
        let avgRe = (
            Double(pressure[dof0].re)
                + Double(pressure[dof1].re)
                + Double(pressure[dof2].re)
        ) / 3.0
        let avgIm = (
            Double(pressure[dof0].im)
                + Double(pressure[dof1].im)
                + Double(pressure[dof2].im)
        ) / 3.0
        sumRe += avgRe * area
        sumIm += avgIm * area
        totalArea += area
    }
    if totalArea < 1.0e-30 {
        return (0.0, 0.0, 0.0)
    }
    return (sumRe / totalArea, sumIm / totalArea, totalArea)
}

func nativePressureReductionPayload(
    geom: Geometry,
    pressure: [Complex32],
    sourceTags: [Int]?,
    impedanceSourceTag: Int?
) -> [String: Any] {
    var payload: [String: Any] = [:]
    if let impedanceSourceTag {
        let avg = averageSurfacePressureForTag(
            geom: geom,
            pressure: pressure,
            tag: impedanceSourceTag
        )
        payload["impedance"] = [avg.re, avg.im]
    }
    if let sourceTags {
        var pavg: [String: Any] = [:]
        for tag in sourceTags {
            let avg = averageSurfacePressureForTag(
                geom: geom,
                pressure: pressure,
                tag: tag
            )
            pavg[String(tag)] = [avg.re, avg.im]
        }
        payload["surface_pressure_avg"] = pavg
    }
    return payload
}

func triangleRule6() -> ([Float], [Float], [Float]) {
    let qx: [Float] = [
        0.4459484909159651, 0.0915762135097710,
        0.1081030181680700, 0.4459484909159651,
        0.8168475729804590, 0.0915762135097710,
    ]
    let qy: [Float] = [
        0.4459484909159651, 0.0915762135097700,
        0.4459484909159651, 0.1081030181680700,
        0.0915762135097700, 0.8168475729804580,
    ]
    let qw: [Float] = [
        0.5 * 0.2233815896780110, 0.5 * 0.1099517436553220,
        0.5 * 0.2233815896780110, 0.5 * 0.2233815896780110,
        0.5 * 0.1099517436553220, 0.5 * 0.1099517436553220,
    ]
    return (qx, qy, qw)
}

func localBasis(_ xi: Float, _ eta: Float) -> (Float, Float, Float) {
    (1.0 - xi - eta, xi, eta)
}

func pointOnTriangle(_ geom: Geometry, _ triangle: Int, _ xi: Float, _ eta: Float)
    -> (Float, Float, Float)
{
    let (b1, b2, b3) = localBasis(xi, eta)
    let i1 = geom.triangleVertex(triangle, 0)
    let i2 = geom.triangleVertex(triangle, 1)
    let i3 = geom.triangleVertex(triangle, 2)
    return (
        b1 * geom.px[i1] + b2 * geom.px[i2] + b3 * geom.px[i3],
        b1 * geom.py[i1] + b2 * geom.py[i2] + b3 * geom.py[i3],
        b1 * geom.pz[i1] + b2 * geom.pz[i2] + b3 * geom.pz[i3]
    )
}

func helmholtzG(_ dx: Float, _ dy: Float, _ dz: Float, _ k: Float) -> Complex32 {
    helmholtzGComplex(dx, dy, dz, kReal: k, kImag: 0.0)
}

func helmholtzGComplex(
    _ dx: Float,
    _ dy: Float,
    _ dz: Float,
    kReal: Float,
    kImag: Float
) -> Complex32 {
    let r2 = dx * dx + dy * dy + dz * dz
    if r2 <= 0 {
        return .zero
    }
    let r = sqrt(r2)
    let phase = kReal * r
    let scale = exp(-kImag * r) * Float(0.07957747154594767) / r
    return Complex32(re: cos(phase) * scale, im: sin(phase) * scale)
}

func helmholtzDlp(
    _ dx: Float,
    _ dy: Float,
    _ dz: Float,
    _ nx: Float,
    _ ny: Float,
    _ nz: Float,
    _ k: Float
) -> Complex32 {
    helmholtzDlpComplex(dx, dy, dz, nx, ny, nz, kReal: k, kImag: 0.0)
}

func helmholtzDlpComplex(
    _ dx: Float,
    _ dy: Float,
    _ dz: Float,
    _ nx: Float,
    _ ny: Float,
    _ nz: Float,
    kReal: Float,
    kImag: Float
) -> Complex32 {
    let r2 = dx * dx + dy * dy + dz * dz
    if r2 <= 0 {
        return .zero
    }
    let r = sqrt(r2)
    let phase = kReal * r
    let scale = exp(-kImag * r) * Float(0.07957747154594767) / r
    let gre = cos(phase) * scale
    let gim = sin(phase) * scale
    let projection = (dx * nx + dy * ny + dz * nz) / r
    let fre = -1.0 / r - kImag
    let fim = kReal
    return Complex32(
        re: (gre * fre - gim * fim) * projection,
        im: (gre * fim + gim * fre) * projection
    )
}

func mirrorMasks(_ symmetryPlane: String?) -> [Int] {
    if symmetryPlane == "yz" {
        return [1]
    }
    if symmetryPlane == "xz" {
        return [2]
    }
    if symmetryPlane == "xy" {
        return [4]
    }
    if symmetryPlane == "yz+xz" {
        return [1, 2, 3]
    }
    return []
}

func mirrorPoint(_ point: (Float, Float, Float), mask: Int) -> (Float, Float, Float) {
    (
        mask & 1 != 0 ? -point.0 : point.0,
        mask & 2 != 0 ? -point.1 : point.1,
        mask & 4 != 0 ? -point.2 : point.2
    )
}

func mirrorNormal(_ normal: (Float, Float, Float), mask: Int) -> (Float, Float, Float) {
    (
        mask & 1 != 0 ? -normal.0 : normal.0,
        mask & 2 != 0 ? -normal.1 : normal.1,
        mask & 4 != 0 ? -normal.2 : normal.2
    )
}

func symmetryImageMasks(_ symmetryPlane: String?) -> [Int] {
    mirrorMasks(symmetryPlane)
}

func readComplexVector(root: String, descriptors: [String: Any], count: Int) throws -> [Complex32] {
    let realDesc = try requireObject(descriptors, "real_f32")
    let imagDesc = try requireObject(descriptors, "imag_f32")
    let re = try readF32(try descriptorPath(root: root, descriptor: realDesc), expectedCount: count)
    let im = try readF32(try descriptorPath(root: root, descriptor: imagDesc), expectedCount: count)
    return zip(re, im).map { Complex32(re: $0.0, im: $0.1) }
}

func robinBetasByTriangle(geom: Geometry, casePayload: [String: Any]) throws -> [Complex32]? {
    guard let raw = casePayload["impedance_sources"] else {
        return nil
    }
    guard let impedanceSources = raw as? [String: Any] else {
        try fail("impedance_sources must be an object")
    }
    var betaByTag: [Int32: Complex32] = [:]
    for (tagString, value) in impedanceSources {
        guard let tag = Int32(tagString) else {
            try fail("impedance_sources keys must be integer tag strings")
        }
        guard let pair = value as? [Any], pair.count == 2 else {
            try fail("impedance_sources values must be [real, imag]")
        }
        guard let reNumber = pair[0] as? NSNumber, let imNumber = pair[1] as? NSNumber else {
            try fail("impedance_sources values must be numeric [real, imag]")
        }
        betaByTag[tag] = Complex32(
            re: reNumber.floatValue,
            im: imNumber.floatValue
        )
    }
    var betas = Array(repeating: Complex32.zero, count: geom.nTriangles)
    for tri in 0..<geom.nTriangles {
        if let beta = betaByTag[geom.physicalTags[tri]] {
            betas[tri] = beta
        }
    }
    return betas
}

func neumannWithRobin(
    geom: Geometry,
    driverNeumann: [Complex32],
    pressure: [Complex32],
    kReal: Float,
    kImag: Float,
    robinBetas: [Complex32]?
) -> [Complex32] {
    guard let robinBetas else {
        return driverNeumann
    }
    let iK = Complex32(re: -kImag, im: kReal)
    var total = driverNeumann
    for tri in 0..<geom.nTriangles {
        let beta = robinBetas[tri]
        if beta.re == 0.0 && beta.im == 0.0 {
            continue
        }
        let dof0 = geom.p1Dof(tri, 0)
        let dof1 = geom.p1Dof(tri, 1)
        let dof2 = geom.p1Dof(tri, 2)
        let pAvg = Complex32(
            re: (pressure[dof0].re + pressure[dof1].re + pressure[dof2].re) / 3.0,
            im: (pressure[dof0].im + pressure[dof1].im + pressure[dof2].im) / 3.0
        )
        total[tri] = total[tri] + (iK * beta * pAvg)
    }
    return total
}

func readObservationPoints(root: String, descriptor: [String: Any]) throws -> [(Float, Float, Float)] {
    let shape = try validateDescriptor(
        descriptor,
        name: "observation_points",
        dtype: "float32",
        rank: 2
    )
    if shape[0] != 3 {
        try fail("observation_points must have shape [3, n_obs]")
    }
    let nObs = shape[1]
    let values = try readF32(
        try descriptorPath(root: root, descriptor: descriptor),
        expectedCount: 3 * nObs
    )
    var points: [(Float, Float, Float)] = []
    points.reserveCapacity(nObs)
    for idx in 0..<nObs {
        points.append((values[idx], values[nObs + idx], values[2 * nObs + idx]))
    }
    return points
}

/// CHIEF interior overdetermination points. Same [3, m] f32 layout as the
/// observation points; given in the reduced (modeled) frame when a symmetry
/// plane is set (the image sum in assembleChiefRows reconstructs the full
/// interior field).
func readChiefPoints(root: String, descriptor: [String: Any]) throws -> [(Float, Float, Float)] {
    let shape = try validateDescriptor(
        descriptor,
        name: "chief_points",
        dtype: "float32",
        rank: 2
    )
    if shape[0] != 3 {
        try fail("chief_points must have shape [3, m]")
    }
    let mPts = shape[1]
    let values = try readF32(
        try descriptorPath(root: root, descriptor: descriptor),
        expectedCount: 3 * mPts
    )
    var points: [(Float, Float, Float)] = []
    points.reserveCapacity(mPts)
    for idx in 0..<mPts {
        points.append((values[idx], values[mPts + idx], values[2 * mPts + idx]))
    }
    return points
}

func assembleRegularReference(
    geom: Geometry,
    neumann: [Complex32],
    k: Float,
    kImag: Float = 0.0,
    robinBetas: [Complex32]? = nil
) -> AssemblyArrays {
    let (qx, qy, qw) = triangleRule6()
    let n = geom.p1DofCount
    var a = Array(repeating: Complex32.zero, count: n * n)
    var rhs = Array(repeating: Complex32.zero, count: n)
    let imageMasks = symmetryImageMasks(geom.symmetryPlane)
    let iK = Complex32(re: -kImag, im: k)

    for trial in 0..<geom.nTriangles {
        let nnx = geom.normal(trial, 0)
        let nny = geom.normal(trial, 1)
        let nnz = geom.normal(trial, 2)
        let trialArea = geom.areas[trial]
        let gTrial = neumann[trial]
        let betaTrial = robinBetas?[trial] ?? Complex32.zero
        let robinCoupling = iK * betaTrial
        let hasRobin = betaTrial.re != 0.0 || betaTrial.im != 0.0
        for test in 0..<geom.nTriangles {
            let jac = (2.0 * geom.areas[test]) * (2.0 * trialArea)
            var block = Array(repeating: Complex32.zero, count: 9)
            var slp = Array(repeating: Complex32.zero, count: 3)
            for qa in 0..<qw.count {
                let (tx, ty, tz) = pointOnTriangle(geom, test, qx[qa], qy[qa])
                let tb = localBasis(qx[qa], qy[qa])
                let tbasis = [tb.0, tb.1, tb.2]
                for qb in 0..<qw.count {
                    let (sx, sy, sz) = pointOnTriangle(geom, trial, qx[qb], qy[qb])
                    let sb = localBasis(qx[qb], qy[qb])
                    let sbasis = [sb.0, sb.1, sb.2]
                    let dx = sx - tx
                    let dy = sy - ty
                    let dz = sz - tz
                    let g = helmholtzGComplex(
                        dx, dy, dz,
                        kReal: k,
                        kImag: kImag
                    )
                    let d = helmholtzDlpComplex(
                        dx, dy, dz,
                        nnx, nny, nnz,
                        kReal: k,
                        kImag: kImag
                    )
                    let w = qw[qa] * qw[qb] * jac
                    for i in 0..<3 {
                        slp[i] = slp[i] + g * (tbasis[i] * w)
                        for j in 0..<3 {
                            block[i * 3 + j] = block[i * 3 + j] + d * (tbasis[i] * sbasis[j] * w)
                        }
                    }
                    for mask in imageMasks {
                        let image = mirrorPoint((sx, sy, sz), mask: mask)
                        let imageNormal = mirrorNormal((nnx, nny, nnz), mask: mask)
                        let idx = image.0 - tx
                        let idy = image.1 - ty
                        let idz = image.2 - tz
                        let imageG = helmholtzGComplex(
                            idx, idy, idz,
                            kReal: k,
                            kImag: kImag
                        )
                        let imageD = helmholtzDlpComplex(
                            idx, idy, idz,
                            imageNormal.0, imageNormal.1, imageNormal.2,
                            kReal: k,
                            kImag: kImag
                        )
                        for i in 0..<3 {
                            slp[i] = slp[i] + imageG * (tbasis[i] * w)
                            for j in 0..<3 {
                                block[i * 3 + j] = block[i * 3 + j] + imageD * (tbasis[i] * sbasis[j] * w)
                            }
                        }
                    }
                }
            }
            for i in 0..<3 {
                let row = geom.p1Dof(test, i)
                let rowWeight = geom.symmetryRowWeight(row)
                rhs[row] = rhs[row] + (slp[i] * gTrial) * rowWeight
                for j in 0..<3 {
                    let col = geom.p1Dof(trial, j)
                    var term = block[i * 3 + j]
                    if hasRobin {
                        term = term - ((slp[i] * robinCoupling) * Float(1.0 / 3.0))
                    }
                    a[row * n + col] = a[row * n + col] + term * rowWeight
                }
            }
            if test == trial {
                for i in 0..<3 {
                    let row = geom.p1Dof(test, i)
                    let rowWeight = geom.symmetryRowWeight(row)
                    for j in 0..<3 {
                        let col = geom.p1Dof(trial, j)
                        let mass = geom.areas[test] * (i == j ? Float(1.0 / 6.0) : Float(1.0 / 12.0))
                        a[row * n + col] = a[row * n + col] - Complex32(re: 0.5 * mass * rowWeight, im: 0)
                    }
                }
            }
        }
    }

    return AssemblyArrays(
        aRe: a.map { $0.re },
        aIm: a.map { $0.im },
        rhsRe: rhs.map { $0.re },
        rhsIm: rhs.map { $0.im }
    )
}

func evaluateExteriorReference(
    geom: Geometry,
    pressure: [Complex32],
    neumann: [Complex32],
    observationPoints: [(Float, Float, Float)],
    k: Float
) -> [Complex32] {
    let (qx, qy, qw) = triangleRule6()
    var out = Array(repeating: Complex32.zero, count: observationPoints.count)
    let imageMasks = symmetryImageMasks(geom.symmetryPlane)

    for obsIdx in observationPoints.indices {
        let (ox, oy, oz) = observationPoints[obsIdx]
        var acc = Complex32.zero
        for tri in 0..<geom.nTriangles {
            let nnx = geom.normal(tri, 0)
            let nny = geom.normal(tri, 1)
            let nnz = geom.normal(tri, 2)
            let jac = 2.0 * geom.areas[tri]
            for qa in 0..<qw.count {
                let (sx, sy, sz) = pointOnTriangle(geom, tri, qx[qa], qy[qa])
                let b = localBasis(qx[qa], qy[qa])
                let basis = [b.0, b.1, b.2]
                var surfacePressure = Complex32.zero
                for local in 0..<3 {
                    surfacePressure = surfacePressure + pressure[geom.p1Dof(tri, local)] * basis[local]
                }
                let dx = sx - ox
                let dy = sy - oy
                let dz = sz - oz
                let d = helmholtzDlp(dx, dy, dz, nnx, nny, nnz, k)
                let g = helmholtzG(dx, dy, dz, k)
                acc = acc + ((d * surfacePressure) - (g * neumann[tri])) * (qw[qa] * jac)
                for mask in imageMasks {
                    let image = mirrorPoint((sx, sy, sz), mask: mask)
                    let imageNormal = mirrorNormal((nnx, nny, nnz), mask: mask)
                    let idx = image.0 - ox
                    let idy = image.1 - oy
                    let idz = image.2 - oz
                    let imageD = helmholtzDlp(
                        idx, idy, idz,
                        imageNormal.0, imageNormal.1, imageNormal.2,
                        k
                    )
                    let imageG = helmholtzG(idx, idy, idz, k)
                    acc = acc + ((imageD * surfacePressure) - (imageG * neumann[tri])) * (qw[qa] * jac)
                }
            }
        }
        out[obsIdx] = acc
    }
    return out
}

/// Assemble the CHIEF (Combined Helmholtz Interior-integral Equation
/// Formulation) constraint rows. Each interior point x_c contributes one
/// equation requiring the exterior representation to evaluate to 0 inside the
/// body (the interior null-field property):
///
///     sum over triangles, images:  D(x_c, y)*p(y) - G(x_c, y)*q(y) = 0,
///         with  q = g_drv + iK*beta*Pi*p   (the Robin total Neumann data).
///
/// Moving p to the LHS and g_drv to the RHS, folding Robin exactly as the
/// boundary rows do (main.swift assembleRegularReference, term -= (slp*iK*beta)/3
/// per P1 column) and the field-side reconstruction neumannWithRobin
/// (q += iK*beta*pAvg, pAvg = sum_local p/3):
///
///     [ D_chief - G_chief*iK*diag(beta)*Pi ] * p  =  G_chief * g_drv
///
/// This is structurally evaluateExteriorReference with the unknown p kept
/// symbolic: per-P1-DOF D coefficients go into a dense row, sum(G)*g_drv into
/// the RHS, and the Robin fold into the same columns.
///
/// CRITICAL subtleties (per the implementation plan):
///   - CHIEF rows are point COLLOCATION of the representation formula, NOT the
///     Galerkin boundary operator. So: RHS = +integral(G*g_drv) (G, not the
///     test-basis-weighted slp), NO -1/2 I jump term, NO symmetryRowWeight.
///   - Uses the REAL-k field-eval kernels (helmholtzG / helmholtzDlp), matching
///     the physical exterior representation that evaluateExterior uses for the
///     field; NOT the complex-k assembly kernels. (The boundary operator may
///     still carry the complex-k shift; CHIEF + complex_k compose fine.)
///   - Includes the symmetry mirror images, so the image sum reconstructs the
///     full interior field from points given in the reduced frame.
///
/// Returns the row-major m x n coefficient matrix (cRe, cIm), the RHS (dRe,
/// dIm), and ||C||_inf (max abs entry) for auto-scaling against ||A||_inf.
func assembleChiefRows(
    geom: Geometry,
    chiefPoints: [(Float, Float, Float)],
    driverNeumann: [Complex32],
    k: Float,
    kImag: Float,
    robinBetas: [Complex32]?
) -> (cRe: [Float], cIm: [Float], dRe: [Float], dIm: [Float], cNormInf: Float) {
    let (qx, qy, qw) = triangleRule6()
    let n = geom.p1DofCount
    let m = chiefPoints.count
    var c = Array(repeating: Complex32.zero, count: m * n)   // row-major m x n
    var d = Array(repeating: Complex32.zero, count: m)
    let imageMasks = symmetryImageMasks(geom.symmetryPlane)
    // iK = i*k_complex with the SAME convention as the boundary fold and the
    // field-side Robin reconstruction (main.swift:1026 / :972).
    let iK = Complex32(re: -kImag, im: k)
    let third = Float(1.0 / 3.0)

    for ci in 0..<m {
        let (ox, oy, oz) = chiefPoints[ci]
        for tri in 0..<geom.nTriangles {
            let nnx = geom.normal(tri, 0)
            let nny = geom.normal(tri, 1)
            let nnz = geom.normal(tri, 2)
            let jac = 2.0 * geom.areas[tri]
            let gTri = driverNeumann[tri]
            let beta = robinBetas?[tri] ?? Complex32.zero
            let hasRobin = beta.re != 0.0 || beta.im != 0.0
            let robinCoupling = iK * beta
            // Per local P1 basis: integral(D*basis_local) coefficients, and the
            // scalar integral(G) (the basis sums to 1, so the G*g_drv RHS uses
            // the basis-independent sum).
            var dCoeff = [Complex32](repeating: .zero, count: 3)
            var gScalar = Complex32.zero
            for qa in 0..<qw.count {
                let (sx, sy, sz) = pointOnTriangle(geom, tri, qx[qa], qy[qa])
                let b = localBasis(qx[qa], qy[qa])
                let basis = [b.0, b.1, b.2]
                let dx = sx - ox
                let dy = sy - oy
                let dz = sz - oz
                let dd = helmholtzDlp(dx, dy, dz, nnx, nny, nnz, k)
                let gg = helmholtzG(dx, dy, dz, k)
                let w = qw[qa] * jac
                for i in 0..<3 {
                    dCoeff[i] = dCoeff[i] + dd * (basis[i] * w)
                }
                gScalar = gScalar + gg * w
                for mask in imageMasks {
                    let image = mirrorPoint((sx, sy, sz), mask: mask)
                    let inrm = mirrorNormal((nnx, nny, nnz), mask: mask)
                    let idx = image.0 - ox
                    let idy = image.1 - oy
                    let idz = image.2 - oz
                    let idd = helmholtzDlp(idx, idy, idz, inrm.0, inrm.1, inrm.2, k)
                    let igg = helmholtzG(idx, idy, idz, k)
                    for i in 0..<3 {
                        dCoeff[i] = dCoeff[i] + idd * (basis[i] * w)
                    }
                    gScalar = gScalar + igg * w
                }
            }
            // RHS: + integral(G) * g_drv  (the +G*g_drv moved from the -G*q term).
            d[ci] = d[ci] + gScalar * gTri
            // LHS columns: D coefficient, minus the Robin fold -G*(iK*beta)/3 per
            // P1 column (same sign as the boundary operator fold).
            for i in 0..<3 {
                let col = geom.p1Dof(tri, i)
                var term = dCoeff[i]
                if hasRobin {
                    term = term - (gScalar * robinCoupling) * third
                }
                c[ci * n + col] = c[ci * n + col] + term
            }
        }
    }

    // True matrix infinity norm of the m x n (row-major) CHIEF block: max over
    // rows of the sum of complex magnitudes, matching ||A||_inf so the auto-scale
    // ratio is dimensionally consistent.
    var cNormInf: Float = 0
    for ci in 0..<m {
        var rowSum: Float = 0
        let base = ci * n
        for col in 0..<n {
            let v = c[base + col]
            rowSum += Float(hypot(Double(v.re), Double(v.im)))
        }
        cNormInf = max(cNormInf, rowSum)
    }
    return (
        c.map { $0.re },
        c.map { $0.im },
        d.map { $0.re },
        d.map { $0.im },
        cNormInf
    )
}

struct P1Incidence {
    let incTri: [Int32]
    let incLoc: [Int32]
    let counts: [Int32]
    let maxInc: Int
}

func buildP1Incidence(_ geom: Geometry) throws -> P1Incidence {
    let n = geom.p1DofCount
    var counts = Array(repeating: Int32(0), count: n)
    for tri in 0..<geom.nTriangles {
        for local in 0..<3 {
            let dof = geom.p1Dof(tri, local)
            if dof < 0 || dof >= n {
                try fail("p1_local2global contains out-of-range DOF \(dof)")
            }
            counts[dof] += 1
        }
    }

    let maxInc = max(1, counts.map { Int($0) }.max() ?? 1)
    var offsets = Array(repeating: Int32(0), count: n)
    var incTri = Array(repeating: Int32(-1), count: n * maxInc)
    var incLoc = Array(repeating: Int32(-1), count: n * maxInc)
    for tri in 0..<geom.nTriangles {
        for local in 0..<3 {
            let dof = geom.p1Dof(tri, local)
            let slot = Int(offsets[dof])
            let cursor = dof * maxInc + slot
            incTri[cursor] = Int32(tri)
            incLoc[cursor] = Int32(local)
            offsets[dof] += 1
        }
    }

    return P1Incidence(
        incTri: incTri,
        incLoc: incLoc,
        counts: counts,
        maxInc: maxInc
    )
}

func sharedLocalIds(_ geom: Geometry, test: Int, trial: Int)
    -> (count: Int, tl1: Int, tl2: Int, sl1: Int, sl2: Int)
{
    var count = 0
    var tl1 = 0
    var tl2 = 0
    var sl1 = 0
    var sl2 = 0
    for i in 0..<3 {
        let tv = geom.triangleVertex(test, i)
        for j in 0..<3 {
            if tv == geom.triangleVertex(trial, j) {
                count += 1
                if count == 1 {
                    tl1 = i
                    sl1 = j
                } else {
                    tl2 = i
                    sl2 = j
                }
            }
        }
    }
    return (count, tl1, tl2, sl1, sl2)
}

func coordinateKey(_ point: (Float, Float, Float), tolerance: Double = 1.0e-6) -> String {
    let x = Int64((Double(point.0) / tolerance).rounded())
    let y = Int64((Double(point.1) / tolerance).rounded())
    let z = Int64((Double(point.2) / tolerance).rounded())
    return "\(x),\(y),\(z)"
}

func vertexPoint(_ geom: Geometry, _ vertex: Int) -> (Float, Float, Float) {
    (geom.px[vertex], geom.py[vertex], geom.pz[vertex])
}

func imageUsesReversedLocalOrder(_ mask: Int) -> Bool {
    var remaining = mask
    var bitCount = 0
    while remaining != 0 {
        bitCount += remaining & 1
        remaining >>= 1
    }
    return bitCount % 2 == 1
}

func imageLocalForOriginalLocal(_ local: Int, mask: Int) -> Int {
    if !imageUsesReversedLocalOrder(mask) {
        return local
    }
    if local == 1 {
        return 2
    }
    if local == 2 {
        return 1
    }
    return local
}

func originalLocalForImageLocal(_ local: Int, mask: Int) -> Int {
    imageLocalForOriginalLocal(local, mask: mask)
}

func imageRefToOriginalRef(_ xi: Float, _ eta: Float, mask: Int) -> (Float, Float) {
    if imageUsesReversedLocalOrder(mask) {
        return (eta, xi)
    }
    return (xi, eta)
}

func imageSharedLocalIds(
    _ geom: Geometry,
    test: Int,
    trial: Int,
    testImageMask: Int,
    trialImageMask: Int
)
    -> (count: Int, tl1: Int, tl2: Int, sl1: Int, sl2: Int)
{
    if testImageMask == 0 && trialImageMask == 0 {
        return sharedLocalIds(geom, test: test, trial: trial)
    }
    var count = 0
    var tl1 = 0
    var tl2 = 0
    var sl1 = 0
    var sl2 = 0
    for testFullLocal in 0..<3 {
        let i = originalLocalForImageLocal(testFullLocal, mask: testImageMask)
        let testPoint = mirrorPoint(
            vertexPoint(geom, geom.triangleVertex(test, i)),
            mask: testImageMask
        )
        let testKey = coordinateKey(testPoint)
        for trialFullLocal in 0..<3 {
            let j = originalLocalForImageLocal(trialFullLocal, mask: trialImageMask)
            let trialPoint = vertexPoint(geom, geom.triangleVertex(trial, j))
            let imageKey = coordinateKey(mirrorPoint(trialPoint, mask: trialImageMask))
            if testKey == imageKey {
                count += 1
                if count == 1 {
                    tl1 = testFullLocal
                    sl1 = trialFullLocal
                } else {
                    tl2 = testFullLocal
                    sl2 = trialFullLocal
                }
            }
        }
    }
    return (count, tl1, tl2, sl1, sl2)
}

func appendDuffyPair(
    pairs: inout [DuffyPair],
    plan: inout DuffyPairPlanAccumulator,
    test: Int,
    trial: Int,
    testImageMask: Int,
    trialImageMask: Int,
    shared: (count: Int, tl1: Int, tl2: Int, sl1: Int, sl2: Int)
) {
    let count = shared.count
    if count <= 0 {
        return
    }
    if count == 3 {
        plan.coincident += 1
        pairs.append(
            DuffyPair(
                test: test,
                trial: trial,
                testImageMask: testImageMask,
                trialImageMask: trialImageMask,
                kind: 1,
                testLocal1: 0,
                testLocal2: 1,
                trialLocal1: 0,
                trialLocal2: 1
            )
        )
    } else if count == 2 {
        plan.edge += 1
        pairs.append(
            DuffyPair(
                test: test,
                trial: trial,
                testImageMask: testImageMask,
                trialImageMask: trialImageMask,
                kind: 2,
                testLocal1: shared.tl1,
                testLocal2: shared.tl2,
                trialLocal1: shared.sl1,
                trialLocal2: shared.sl2
            )
        )
    } else if count == 1 {
        plan.vertex += 1
        pairs.append(
            DuffyPair(
                test: test,
                trial: trial,
                testImageMask: testImageMask,
                trialImageMask: trialImageMask,
                kind: 3,
                testLocal1: shared.tl1,
                testLocal2: shared.tl2,
                trialLocal1: shared.sl1,
                trialLocal2: shared.sl2
            )
        )
    }
}

struct DuffyPairPlanAccumulator {
    var coincident = 0
    var edge = 0
    var vertex = 0

    var plan: DuffyPairPlan {
        DuffyPairPlan(coincident: coincident, edge: edge, vertex: vertex)
    }
}

func buildRealDuffyPairList(_ geom: Geometry) throws -> DuffyPairList {
    var vertexToTriangles = Array(repeating: [Int](), count: geom.nVertices)
    for tri in 0..<geom.nTriangles {
        for local in 0..<3 {
            let vertex = geom.triangleVertex(tri, local)
            if vertex < 0 || vertex >= geom.nVertices {
                try fail("triangles_i32 contains out-of-range vertex \(vertex)")
            }
            vertexToTriangles[vertex].append(tri)
        }
    }

    var plan = DuffyPairPlanAccumulator()
    var seenStamp = Array(repeating: Int32(0), count: geom.nTriangles)
    var stamp = Int32(0)
    var candidates: [Int] = []
    var pairs: [DuffyPair] = []
    pairs.reserveCapacity(geom.nTriangles * 12)

    for trial in 0..<geom.nTriangles {
        stamp += 1
        candidates.removeAll(keepingCapacity: true)
        for local in 0..<3 {
            let v = geom.triangleVertex(trial, local)
            for test in vertexToTriangles[v] where seenStamp[test] != stamp {
                seenStamp[test] = stamp
                candidates.append(test)
            }
        }

        for test in candidates {
            let shared = sharedLocalIds(geom, test: test, trial: trial)
            appendDuffyPair(
                pairs: &pairs,
                plan: &plan,
                test: test,
                trial: trial,
                testImageMask: 0,
                trialImageMask: 0,
                shared: shared
            )
        }
    }

    return DuffyPairList(
        pairs: pairs,
        plan: plan.plan
    )
}

func buildSymmetryDuffyPairList(_ geom: Geometry) throws -> DuffyPairList {
    let imageMasks = [0] + symmetryImageMasks(geom.symmetryPlane)
    var vertexKeyToTrianglesByMask: [Int: [String: [Int]]] = [:]
    for testImageMask in imageMasks {
        var vertexKeyToTriangles: [String: [Int]] = [:]
        for tri in 0..<geom.nTriangles {
            for local in 0..<3 {
                let vertex = geom.triangleVertex(tri, local)
                if vertex < 0 || vertex >= geom.nVertices {
                    try fail("triangles_i32 contains out-of-range vertex \(vertex)")
                }
                let point = mirrorPoint(vertexPoint(geom, vertex), mask: testImageMask)
                let key = coordinateKey(point)
                vertexKeyToTriangles[key, default: []].append(tri)
            }
        }
        vertexKeyToTrianglesByMask[testImageMask] = vertexKeyToTriangles
    }

    var plan = DuffyPairPlanAccumulator()
    var pairs: [DuffyPair] = []
    pairs.reserveCapacity(geom.nTriangles * 72)
    var seenStamp = Array(repeating: Int32(0), count: geom.nTriangles)
    var stamp = Int32(0)
    var candidates: [Int] = []

    for trial in 0..<geom.nTriangles {
        for trialImageMask in imageMasks {
            for testImageMask in imageMasks {
                stamp += 1
                candidates.removeAll(keepingCapacity: true)
                let vertexKeyToTriangles = vertexKeyToTrianglesByMask[testImageMask] ?? [:]
                for local in 0..<3 {
                    let vertex = geom.triangleVertex(trial, local)
                    let point = mirrorPoint(vertexPoint(geom, vertex), mask: trialImageMask)
                    let key = coordinateKey(point)
                    for test in vertexKeyToTriangles[key] ?? [] where seenStamp[test] != stamp {
                        seenStamp[test] = stamp
                        candidates.append(test)
                    }
                }
                for test in candidates {
                    let shared = imageSharedLocalIds(
                        geom,
                        test: test,
                        trial: trial,
                        testImageMask: testImageMask,
                        trialImageMask: trialImageMask
                    )
                    appendDuffyPair(
                        pairs: &pairs,
                        plan: &plan,
                        test: test,
                        trial: trial,
                        testImageMask: testImageMask,
                        trialImageMask: trialImageMask,
                        shared: shared
                    )
                }
            }
        }
    }

    return DuffyPairList(pairs: pairs, plan: plan.plan)
}

func buildDuffyPairList(_ geom: Geometry) throws -> DuffyPairList {
    if geom.symmetryPlane == nil {
        return try buildRealDuffyPairList(geom)
    }
    return try buildSymmetryDuffyPairList(geom)
}

func buildDuffyPairPlan(_ geom: Geometry) throws -> DuffyPairPlan {
    try buildDuffyPairList(geom).plan
}

struct TriangleNearMetrics {
    let centroid: (Float, Float, Float)
    let minPoint: (Float, Float, Float)
    let maxPoint: (Float, Float, Float)
    let longestEdge: Float
}

func distanceSquared(
    _ a: (Float, Float, Float),
    _ b: (Float, Float, Float)
) -> Float {
    let dx = a.0 - b.0
    let dy = a.1 - b.1
    let dz = a.2 - b.2
    return dx * dx + dy * dy + dz * dz
}

func triangleNearMetrics(geom: Geometry, triangle: Int, mask: Int) throws -> TriangleNearMetrics {
    var points: [(Float, Float, Float)] = []
    points.reserveCapacity(3)
    for local in 0..<3 {
        let vertex = geom.triangleVertex(triangle, local)
        if vertex < 0 || vertex >= geom.nVertices {
            try fail("triangles_i32 contains out-of-range vertex \(vertex)")
        }
        points.append(mirrorPoint(vertexPoint(geom, vertex), mask: mask))
    }

    let centroid = (
        (points[0].0 + points[1].0 + points[2].0) / 3.0,
        (points[0].1 + points[1].1 + points[2].1) / 3.0,
        (points[0].2 + points[1].2 + points[2].2) / 3.0
    )
    let minPoint = (
        min(points[0].0, min(points[1].0, points[2].0)),
        min(points[0].1, min(points[1].1, points[2].1)),
        min(points[0].2, min(points[1].2, points[2].2))
    )
    let maxPoint = (
        max(points[0].0, max(points[1].0, points[2].0)),
        max(points[0].1, max(points[1].1, points[2].1)),
        max(points[0].2, max(points[1].2, points[2].2))
    )
    let longestEdgeSquared = max(
        distanceSquared(points[0], points[1]),
        max(distanceSquared(points[1], points[2]), distanceSquared(points[2], points[0]))
    )
    return TriangleNearMetrics(
        centroid: centroid,
        minPoint: minPoint,
        maxPoint: maxPoint,
        longestEdge: sqrt(longestEdgeSquared)
    )
}

func bboxDistanceSquared(_ a: TriangleNearMetrics, _ b: TriangleNearMetrics) -> Double {
    func axisDistance(_ amin: Float, _ amax: Float, _ bmin: Float, _ bmax: Float) -> Double {
        if amax < bmin {
            return Double(bmin - amax)
        }
        if bmax < amin {
            return Double(amin - bmax)
        }
        return 0.0
    }
    let dx = axisDistance(a.minPoint.0, a.maxPoint.0, b.minPoint.0, b.maxPoint.0)
    let dy = axisDistance(a.minPoint.1, a.maxPoint.1, b.minPoint.1, b.maxPoint.1)
    let dz = axisDistance(a.minPoint.2, a.maxPoint.2, b.minPoint.2, b.maxPoint.2)
    return dx * dx + dy * dy + dz * dz
}

func centroidDistanceSquared(_ a: TriangleNearMetrics, _ b: TriangleNearMetrics) -> Double {
    let dx = Double(a.centroid.0 - b.centroid.0)
    let dy = Double(a.centroid.1 - b.centroid.1)
    let dz = Double(a.centroid.2 - b.centroid.2)
    return dx * dx + dy * dy + dz * dz
}

func buildNearPairList(geom: Geometry, threshold: Double) throws -> NearPairList {
    let imageMasks: [Int]
    if geom.symmetryPlane == nil {
        imageMasks = [0]
    } else {
        imageMasks = [0] + symmetryImageMasks(geom.symmetryPlane)
    }
    var metricsByMask: [Int: [TriangleNearMetrics]] = [:]
    for mask in imageMasks {
        var metrics: [TriangleNearMetrics] = []
        metrics.reserveCapacity(geom.nTriangles)
        for tri in 0..<geom.nTriangles {
            metrics.append(try triangleNearMetrics(geom: geom, triangle: tri, mask: mask))
        }
        metricsByMask[mask] = metrics
    }

    let lock = NSLock()
    var pairs: [NearPair] = []
    pairs.reserveCapacity(geom.nTriangles * imageMasks.count * imageMasks.count)

    // Parallel brute force is enough for the current correction sizes; a
    // spatial grid is the next step if this pair-list build shows up in profiles.
    DispatchQueue.concurrentPerform(iterations: geom.nTriangles) { trial in
        var localPairs: [NearPair] = []
        for trialImageMask in imageMasks {
            guard let trialMetrics = metricsByMask[trialImageMask] else {
                continue
            }
            let trialMetric = trialMetrics[trial]
            for testImageMask in imageMasks {
                guard let testMetrics = metricsByMask[testImageMask] else {
                    continue
                }
                for test in 0..<geom.nTriangles {
                    let testMetric = testMetrics[test]
                    let cutoff = threshold * Double(
                        max(testMetric.longestEdge, trialMetric.longestEdge)
                    )
                    let cutoffSquared = cutoff * cutoff
                    if bboxDistanceSquared(testMetric, trialMetric) > cutoffSquared {
                        continue
                    }
                    if centroidDistanceSquared(testMetric, trialMetric) >= cutoffSquared {
                        continue
                    }
                    let shared = imageSharedLocalIds(
                        geom,
                        test: test,
                        trial: trial,
                        testImageMask: testImageMask,
                        trialImageMask: trialImageMask
                    )
                    if shared.count > 0 {
                        continue
                    }
                    localPairs.append(
                        NearPair(
                            test: test,
                            trial: trial,
                            testImageMask: testImageMask,
                            trialImageMask: trialImageMask
                        )
                    )
                }
            }
        }
        if !localPairs.isEmpty {
            lock.lock()
            pairs.append(contentsOf: localPairs)
            lock.unlock()
        }
    }

    pairs.sort {
        if $0.trial != $1.trial {
            return $0.trial < $1.trial
        }
        if $0.trialImageMask != $1.trialImageMask {
            return $0.trialImageMask < $1.trialImageMask
        }
        if $0.testImageMask != $1.testImageMask {
            return $0.testImageMask < $1.testImageMask
        }
        return $0.test < $1.test
    }
    return NearPairList(pairs: pairs)
}

func buildDuffyReductionPlan(geom: Geometry, pairList: DuffyPairList) -> DuffyReductionPlan {
    let n = geom.p1DofCount
    let pairCount = pairList.pairs.count
    var pairTrialTriangles = Array(repeating: 0, count: pairCount)
    var rhsRows = Array(repeating: 0, count: pairCount * 3)
    var rowWeights = Array(repeating: Float(1.0), count: pairCount * 3)
    var dlpSlots = Array(repeating: 0, count: pairCount * 9)
    var matrixIndices: [Int] = []
    var tripletSlots: [Int64: Int] = [:]
    var imagePairs = 0
    matrixIndices.reserveCapacity(pairList.plan.total * 3)
    tripletSlots.reserveCapacity(pairList.plan.total * 3)

    for pairIndex in pairList.pairs.indices {
        let pair = pairList.pairs[pairIndex]
        pairTrialTriangles[pairIndex] = pair.trial
        if pair.testImageMask != 0 || pair.trialImageMask != 0 {
            imagePairs += 1
        }
        for i in 0..<3 {
            let row = geom.p1Dof(pair.test, i)
            let slpIndex = pairIndex + i * pairCount
            rhsRows[slpIndex] = row
            // Duffy deltas are applied at weight 1: with symmetry, the pair list
            // enumerates (testImageMask, trialImageMask) combinations explicitly,
            // which already produces the same 2^planes factor that p1_row_weight
            // applies to the regular assembly.
            rowWeights[slpIndex] = 1.0
            for j in 0..<3 {
                let col = geom.p1Dof(pair.trial, j)
                let deltaIndex = pairIndex + (i * 3 + j) * pairCount
                let key = Int64(row) + Int64(col) * Int64(n)
                let slot: Int
                if let existing = tripletSlots[key] {
                    slot = existing
                } else {
                    slot = matrixIndices.count
                    tripletSlots[key] = slot
                    matrixIndices.append(row * n + col)
                }
                dlpSlots[deltaIndex] = slot
            }
        }
    }

    return DuffyReductionPlan(
        pairTrialTriangles: pairTrialTriangles,
        rhsRows: rhsRows,
        rowWeights: rowWeights,
        dlpSlots: dlpSlots,
        matrixIndices: matrixIndices,
        imagePairs: imagePairs
    )
}

struct DuffyPoint {
    let x: Float
    let y: Float
}

struct DuffyRule {
    let testPoints: [DuffyPoint]
    let trialPoints: [DuffyPoint]
    let weights: [Float]
}

func gaussRule1D4() -> ([Float], [Float]) {
    let x1 = sqrt(Float(3.0 / 7.0) - Float(2.0 / 7.0) * sqrt(Float(6.0 / 5.0))) / 2.0
    let x2 = sqrt(Float(3.0 / 7.0) + Float(2.0 / 7.0) * sqrt(Float(6.0 / 5.0))) / 2.0
    let w1 = (Float(18.0) + sqrt(Float(30.0))) / 72.0
    let w2 = (Float(18.0) - sqrt(Float(30.0))) / 72.0
    return (
        [0.5 - x2, 0.5 - x1, 0.5 + x1, 0.5 + x2],
        [w2, w1, w1, w2]
    )
}

func appendDuffy(
    testPoints: inout [DuffyPoint],
    trialPoints: inout [DuffyPoint],
    weights: inout [Float],
    tx: Float,
    ty: Float,
    sx: Float,
    sy: Float,
    weight: Float
) {
    testPoints.append(DuffyPoint(x: tx - ty, y: ty))
    trialPoints.append(DuffyPoint(x: sx - sy, y: sy))
    weights.append(weight)
}

func duffyRule(kind: Int) throws -> DuffyRule {
    let (xs, ws) = gaussRule1D4()
    var testPoints: [DuffyPoint] = []
    var trialPoints: [DuffyPoint] = []
    var weights: [Float] = []
    testPoints.reserveCapacity(kind == 1 ? 1536 : (kind == 2 ? 1280 : 512))
    trialPoints.reserveCapacity(testPoints.capacity)
    weights.reserveCapacity(testPoints.capacity)

    for a in xs.indices {
        for b in xs.indices {
            for c in xs.indices {
                for d in xs.indices {
                    let xi = xs[b]
                    let eta1 = xs[a]
                    let eta2 = xs[c]
                    let eta3 = xs[d]
                    let eta12 = eta1 * eta2
                    let eta123 = eta12 * eta3
                    let base = ws[a] * ws[b] * ws[c] * ws[d]
                    if kind == 1 {
                        let weight = base * xi * xi * xi * eta1 * eta1 * eta2
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi, ty: xi * (1 - eta1 + eta12), sx: xi * (1 - eta123), sy: xi * (1 - eta1), weight: weight)
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi * (1 - eta123), ty: xi * (1 - eta1), sx: xi, sy: xi * (1 - eta1 + eta12), weight: weight)
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi, ty: xi * (eta1 - eta12 + eta123), sx: xi * (1 - eta12), sy: xi * (eta1 - eta12), weight: weight)
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi * (1 - eta12), ty: xi * (eta1 - eta12), sx: xi, sy: xi * (eta1 - eta12 + eta123), weight: weight)
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi * (1 - eta123), ty: xi * (eta1 - eta123), sx: xi, sy: xi * (eta1 - eta12), weight: weight)
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi, ty: xi * (eta1 - eta12), sx: xi * (1 - eta123), sy: xi * (eta1 - eta123), weight: weight)
                    } else if kind == 2 {
                        let weight = base * xi * xi * xi * eta1 * eta1
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi, ty: xi * eta1 * eta3, sx: xi * (1 - eta12), sy: xi * eta1 * (1 - eta2), weight: weight)
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi, ty: xi * eta1, sx: xi * (1 - eta123), sy: xi * eta1 * eta2 * (1 - eta3), weight: weight * eta2)
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi * (1 - eta12), ty: xi * eta1 * (1 - eta2), sx: xi, sy: xi * eta123, weight: weight * eta2)
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi * (1 - eta123), ty: xi * eta12 * (1 - eta3), sx: xi, sy: xi * eta1, weight: weight * eta2)
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi * (1 - eta123), ty: xi * eta1 * (1 - eta2 * eta3), sx: xi, sy: xi * eta12, weight: weight * eta2)
                    } else if kind == 3 {
                        let weight = base * xi * xi * xi * eta2
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi, ty: xi * eta1, sx: xi * eta2, sy: xi * eta2 * eta3, weight: weight)
                        appendDuffy(testPoints: &testPoints, trialPoints: &trialPoints, weights: &weights, tx: xi * eta2, ty: xi * eta2 * eta3, sx: xi, sy: xi * eta1, weight: weight)
                    } else {
                        try fail("unsupported Duffy kind \(kind)")
                    }
                }
            }
        }
    }
    return DuffyRule(testPoints: testPoints, trialPoints: trialPoints, weights: weights)
}

func refX(_ local: Int) -> Float {
    local == 1 ? 1.0 : 0.0
}

func refY(_ local: Int) -> Float {
    local == 2 ? 1.0 : 0.0
}

func remapSingular(_ point: DuffyPoint, kind: Int, local1: Int, local2: Int)
    -> (Float, Float)
{
    if kind == 1 {
        return (point.x, point.y)
    }
    if kind == 2 {
        let vc = 3 - local1 - local2
        let ax = refX(local1)
        let ay = refY(local1)
        let bx = refX(local2)
        let by = refY(local2)
        let cx = refX(vc)
        let cy = refY(vc)
        return (
            ax + point.x * (bx - ax) + point.y * (cx - ax),
            ay + point.x * (by - ay) + point.y * (cy - ay)
        )
    }
    if local1 == 0 {
        return (point.x, point.y)
    }
    if local1 == 1 {
        return (1.0 - point.x - point.y, point.y)
    }
    return (point.x, 1.0 - point.x - point.y)
}

struct ReferenceSubtriangle {
    let a: (Float, Float)
    let b: (Float, Float)
    let c: (Float, Float)

    var det: Float {
        abs((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0))
    }
}

func midpoint(_ a: (Float, Float), _ b: (Float, Float)) -> (Float, Float) {
    ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5)
}

func referenceSubtriangles(level: Int) -> [ReferenceSubtriangle] {
    var triangles = [
        ReferenceSubtriangle(a: (0.0, 0.0), b: (1.0, 0.0), c: (0.0, 1.0))
    ]
    if level <= 0 {
        return triangles
    }
    for _ in 0..<level {
        var next: [ReferenceSubtriangle] = []
        next.reserveCapacity(triangles.count * 4)
        for tri in triangles {
            let ab = midpoint(tri.a, tri.b)
            let bc = midpoint(tri.b, tri.c)
            let ca = midpoint(tri.c, tri.a)
            next.append(ReferenceSubtriangle(a: tri.a, b: ab, c: ca))
            next.append(ReferenceSubtriangle(a: ab, b: tri.b, c: bc))
            next.append(ReferenceSubtriangle(a: ca, b: bc, c: tri.c))
            next.append(ReferenceSubtriangle(a: ab, b: bc, c: ca))
        }
        triangles = next
    }
    return triangles
}

func pointInSubtriangle(
    _ subtriangle: ReferenceSubtriangle,
    _ xi: Float,
    _ eta: Float
) -> (Float, Float) {
    (
        subtriangle.a.0 + xi * (subtriangle.b.0 - subtriangle.a.0)
            + eta * (subtriangle.c.0 - subtriangle.a.0),
        subtriangle.a.1 + xi * (subtriangle.b.1 - subtriangle.a.1)
            + eta * (subtriangle.c.1 - subtriangle.a.1)
    )
}

func regularPairBlocks(
    geom: Geometry,
    test: Int,
    trial: Int,
    testImageMask: Int,
    trialImageMask: Int,
    k: Float,
    kImag: Float = 0.0
) -> (slp: [Complex32], dlp: [Complex32]) {
    let (qx, qy, qw) = triangleRule6()
    let normal = mirrorNormal(
        (geom.normal(trial, 0), geom.normal(trial, 1), geom.normal(trial, 2)),
        mask: trialImageMask
    )
    let jac = (2.0 * geom.areas[test]) * (2.0 * geom.areas[trial])
    var slp = Array(repeating: Complex32.zero, count: 3)
    var dlp = Array(repeating: Complex32.zero, count: 9)

    for qa in 0..<qw.count {
        let (testXi, testEta) = imageRefToOriginalRef(qx[qa], qy[qa], mask: testImageMask)
        let realTestPoint = pointOnTriangle(geom, test, testXi, testEta)
        let (tx, ty, tz) = mirrorPoint(realTestPoint, mask: testImageMask)
        let tb = localBasis(testXi, testEta)
        let tbasis = [tb.0, tb.1, tb.2]
        for qb in 0..<qw.count {
            let (trialXi, trialEta) = imageRefToOriginalRef(qx[qb], qy[qb], mask: trialImageMask)
            let realTrialPoint = pointOnTriangle(geom, trial, trialXi, trialEta)
            let (sx, sy, sz) = mirrorPoint(realTrialPoint, mask: trialImageMask)
            let sb = localBasis(trialXi, trialEta)
            let sbasis = [sb.0, sb.1, sb.2]
            let dx = sx - tx
            let dy = sy - ty
            let dz = sz - tz
            let g = helmholtzGComplex(dx, dy, dz, kReal: k, kImag: kImag)
            let d = helmholtzDlpComplex(
                dx, dy, dz,
                normal.0, normal.1, normal.2,
                kReal: k,
                kImag: kImag
            )
            let w = qw[qa] * qw[qb] * jac
            for i in 0..<3 {
                slp[i] = slp[i] + g * (tbasis[i] * w)
                for j in 0..<3 {
                    dlp[i * 3 + j] = dlp[i * 3 + j] + d * (tbasis[i] * sbasis[j] * w)
                }
            }
        }
    }
    return (slp, dlp)
}

func subdividedPairBlocks(
    geom: Geometry,
    test: Int,
    trial: Int,
    testImageMask: Int,
    trialImageMask: Int,
    k: Float,
    kImag: Float = 0.0,
    level: Int
) -> (slp: [Complex32], dlp: [Complex32]) {
    let (qx, qy, qw) = triangleRule6()
    let subtriangles = referenceSubtriangles(level: level)
    let normal = mirrorNormal(
        (geom.normal(trial, 0), geom.normal(trial, 1), geom.normal(trial, 2)),
        mask: trialImageMask
    )
    let testJac = 2.0 * geom.areas[test]
    let trialJac = 2.0 * geom.areas[trial]
    var slp = Array(repeating: Complex32.zero, count: 3)
    var dlp = Array(repeating: Complex32.zero, count: 9)

    for testSub in subtriangles {
        let testSubJac = testJac * testSub.det
        for trialSub in subtriangles {
            let jac = testSubJac * trialJac * trialSub.det
            for qa in 0..<qw.count {
                let testImageRef = pointInSubtriangle(testSub, qx[qa], qy[qa])
                let (testXi, testEta) = imageRefToOriginalRef(
                    testImageRef.0,
                    testImageRef.1,
                    mask: testImageMask
                )
                let realTestPoint = pointOnTriangle(geom, test, testXi, testEta)
                let (tx, ty, tz) = mirrorPoint(realTestPoint, mask: testImageMask)
                let tb = localBasis(testXi, testEta)
                let tbasis = [tb.0, tb.1, tb.2]
                for qb in 0..<qw.count {
                    let trialImageRef = pointInSubtriangle(trialSub, qx[qb], qy[qb])
                    let (trialXi, trialEta) = imageRefToOriginalRef(
                        trialImageRef.0,
                        trialImageRef.1,
                        mask: trialImageMask
                    )
                    let realTrialPoint = pointOnTriangle(geom, trial, trialXi, trialEta)
                    let (sx, sy, sz) = mirrorPoint(realTrialPoint, mask: trialImageMask)
                    let sb = localBasis(trialXi, trialEta)
                    let sbasis = [sb.0, sb.1, sb.2]
                    let dx = sx - tx
                    let dy = sy - ty
                    let dz = sz - tz
                    let g = helmholtzGComplex(dx, dy, dz, kReal: k, kImag: kImag)
                    let d = helmholtzDlpComplex(
                        dx, dy, dz,
                        normal.0, normal.1, normal.2,
                        kReal: k,
                        kImag: kImag
                    )
                    let w = qw[qa] * qw[qb] * jac
                    for i in 0..<3 {
                        slp[i] = slp[i] + g * (tbasis[i] * w)
                        for j in 0..<3 {
                            dlp[i * 3 + j] = dlp[i * 3 + j]
                                + d * (tbasis[i] * sbasis[j] * w)
                        }
                    }
                }
            }
        }
    }
    return (slp, dlp)
}

func singularPairBlocks(
    geom: Geometry,
    pair: DuffyPair,
    rules: [Int: DuffyRule],
    k: Float,
    kImag: Float = 0.0
) throws -> (slp: [Complex32], dlp: [Complex32]) {
    guard let rule = rules[pair.kind] else {
        try fail("missing Duffy rule for kind \(pair.kind)")
    }
    let normal = mirrorNormal(
        (geom.normal(pair.trial, 0), geom.normal(pair.trial, 1), geom.normal(pair.trial, 2)),
        mask: pair.trialImageMask
    )
    let jac = (2.0 * geom.areas[pair.test]) * (2.0 * geom.areas[pair.trial])
    var slp = Array(repeating: Complex32.zero, count: 3)
    var dlp = Array(repeating: Complex32.zero, count: 9)

    for q in rule.weights.indices {
        let (txi, teta) = remapSingular(
            rule.testPoints[q],
            kind: pair.kind,
            local1: pair.testLocal1,
            local2: pair.testLocal2
        )
        let (sxi, seta) = remapSingular(
            rule.trialPoints[q],
            kind: pair.kind,
            local1: pair.trialLocal1,
            local2: pair.trialLocal2
        )
        let (testXi, testEta) = imageRefToOriginalRef(txi, teta, mask: pair.testImageMask)
        let realTestPoint = pointOnTriangle(geom, pair.test, testXi, testEta)
        let (tx, ty, tz) = mirrorPoint(realTestPoint, mask: pair.testImageMask)
        let (trialXi, trialEta) = imageRefToOriginalRef(sxi, seta, mask: pair.trialImageMask)
        let realTrialPoint = pointOnTriangle(geom, pair.trial, trialXi, trialEta)
        let (sx, sy, sz) = mirrorPoint(realTrialPoint, mask: pair.trialImageMask)
        let tb = localBasis(testXi, testEta)
        let sb = localBasis(trialXi, trialEta)
        let tbasis = [tb.0, tb.1, tb.2]
        let sbasis = [sb.0, sb.1, sb.2]
        let dx = sx - tx
        let dy = sy - ty
        let dz = sz - tz
        let g = helmholtzGComplex(dx, dy, dz, kReal: k, kImag: kImag)
        let d = helmholtzDlpComplex(
            dx, dy, dz,
            normal.0, normal.1, normal.2,
            kReal: k,
            kImag: kImag
        )
        let w = rule.weights[q] * jac
        for i in 0..<3 {
            slp[i] = slp[i] + g * (tbasis[i] * w)
            for j in 0..<3 {
                dlp[i * 3 + j] = dlp[i * 3 + j] + d * (tbasis[i] * sbasis[j] * w)
            }
        }
    }
    return (slp, dlp)
}

func applyDuffyCorrectionsCPU(
    to arrays: AssemblyArrays,
    geom: Geometry,
    neumann: [Complex32],
    k: Float,
    kImag: Float = 0.0,
    robinBetas: [Complex32]? = nil
) throws -> (AssemblyArrays, DuffyCorrectionStats) {
    let start = CFAbsoluteTimeGetCurrent()
    let pairList = try buildDuffyPairList(geom)
    let rules = [
        1: try duffyRule(kind: 1),
        2: try duffyRule(kind: 2),
        3: try duffyRule(kind: 3),
    ]
    let n = geom.p1DofCount
    var aRe = arrays.aRe
    var aIm = arrays.aIm
    var rhsRe = arrays.rhsRe
    var rhsIm = arrays.rhsIm
    var triplets: [Int64: (Double, Double)] = [:]
    triplets.reserveCapacity(pairList.plan.total * 3)
    let iK = Complex32(re: -kImag, im: k)

    for pair in pairList.pairs {
        let regular = regularPairBlocks(
            geom: geom,
            test: pair.test,
            trial: pair.trial,
            testImageMask: pair.testImageMask,
            trialImageMask: pair.trialImageMask,
            k: k,
            kImag: kImag
        )
        let singular = try singularPairBlocks(
            geom: geom,
            pair: pair,
            rules: rules,
            k: k,
            kImag: kImag
        )
        let gTrial = neumann[pair.trial]
        let betaTrial = robinBetas?[pair.trial] ?? Complex32.zero
        let robinCoupling = iK * betaTrial
        let hasRobin = betaTrial.re != 0.0 || betaTrial.im != 0.0

        for i in 0..<3 {
            let row = geom.p1Dof(pair.test, i)
            // Weight 1: image-mask pair enumeration already carries the symmetry factor.
            let rowWeight: Float = 1.0
            let slpDelta = singular.slp[i] - regular.slp[i]
            let rhsDelta = (slpDelta * gTrial) * rowWeight
            rhsRe[row] += rhsDelta.re
            rhsIm[row] += rhsDelta.im

            for j in 0..<3 {
                let col = geom.p1Dof(pair.trial, j)
                var delta = (singular.dlp[i * 3 + j] - regular.dlp[i * 3 + j]) * rowWeight
                if hasRobin {
                    delta = delta - ((slpDelta * robinCoupling) * Float(1.0 / 3.0))
                }
                let key = Int64(row) + Int64(col) * Int64(n)
                let current = triplets[key] ?? (0.0, 0.0)
                triplets[key] = (
                    current.0 + Double(delta.re),
                    current.1 + Double(delta.im)
                )
            }
        }
    }

    for (key, value) in triplets {
        let row = Int(key % Int64(n))
        let col = Int(key / Int64(n))
        let idx = row * n + col
        aRe[idx] += Float(value.0)
        aIm[idx] += Float(value.1)
    }

    let stats = DuffyCorrectionStats(
        plan: pairList.plan,
        rawTriplets: pairList.plan.total * 9,
        uniqueTriplets: triplets.count,
        seconds: CFAbsoluteTimeGetCurrent() - start,
        implementation: "cpu_duffy_blocks_cpu_reduction",
        blockSeconds: nil,
        reductionSeconds: nil,
        dispatch: nil,
        imagePairs: pairList.pairs.filter { $0.testImageMask != 0 || $0.trialImageMask != 0 }.count,
        reductionPrecomputed: false,
        reductionPlanBuildSeconds: nil
    )
    return (
        AssemblyArrays(aRe: aRe, aIm: aIm, rhsRe: rhsRe, rhsIm: rhsIm),
        stats
    )
}

func applyNearFieldCorrectionsCPU(
    to arrays: AssemblyArrays,
    geom: Geometry,
    neumann: [Complex32],
    k: Float,
    kImag: Float = 0.0,
    robinBetas: [Complex32]? = nil,
    config: NearQuadratureConfig
) throws -> (AssemblyArrays, NearQuadratureStats) {
    let start = CFAbsoluteTimeGetCurrent()
    let pairList = try buildNearPairList(geom: geom, threshold: config.threshold)
    let n = geom.p1DofCount
    var aRe = arrays.aRe
    var aIm = arrays.aIm
    var rhsRe = arrays.rhsRe
    var rhsIm = arrays.rhsIm
    var triplets: [Int64: (Double, Double)] = [:]
    triplets.reserveCapacity(pairList.pairs.count * 9)
    let iK = Complex32(re: -kImag, im: k)

    for pair in pairList.pairs {
        let regular = regularPairBlocks(
            geom: geom,
            test: pair.test,
            trial: pair.trial,
            testImageMask: pair.testImageMask,
            trialImageMask: pair.trialImageMask,
            k: k,
            kImag: kImag
        )
        let subdivided = subdividedPairBlocks(
            geom: geom,
            test: pair.test,
            trial: pair.trial,
            testImageMask: pair.testImageMask,
            trialImageMask: pair.trialImageMask,
            k: k,
            kImag: kImag,
            level: config.level
        )
        let gTrial = neumann[pair.trial]
        let betaTrial = robinBetas?[pair.trial] ?? Complex32.zero
        let robinCoupling = iK * betaTrial
        let hasRobin = betaTrial.re != 0.0 || betaTrial.im != 0.0

        for i in 0..<3 {
            let row = geom.p1Dof(pair.test, i)
            // Weight 1: image-mask pair enumeration already carries the symmetry factor.
            let rowWeight: Float = 1.0
            let slpDelta = subdivided.slp[i] - regular.slp[i]
            let rhsDelta = (slpDelta * gTrial) * rowWeight
            rhsRe[row] += rhsDelta.re
            rhsIm[row] += rhsDelta.im

            for j in 0..<3 {
                let col = geom.p1Dof(pair.trial, j)
                var delta = (subdivided.dlp[i * 3 + j] - regular.dlp[i * 3 + j])
                    * rowWeight
                if hasRobin {
                    delta = delta - ((slpDelta * robinCoupling) * Float(1.0 / 3.0))
                }
                let key = Int64(row) + Int64(col) * Int64(n)
                let current = triplets[key] ?? (0.0, 0.0)
                triplets[key] = (
                    current.0 + Double(delta.re),
                    current.1 + Double(delta.im)
                )
            }
        }
    }

    for (key, value) in triplets {
        let row = Int(key % Int64(n))
        let col = Int(key / Int64(n))
        let idx = row * n + col
        aRe[idx] += Float(value.0)
        aIm[idx] += Float(value.1)
    }

    let stats = NearQuadratureStats(
        level: config.level,
        threshold: config.threshold,
        pairCount: pairList.pairs.count,
        seconds: CFAbsoluteTimeGetCurrent() - start
    )
    return (
        AssemblyArrays(aRe: aRe, aIm: aIm, rhsRe: rhsRe, rhsIm: rhsIm),
        stats
    )
}

func applyNearFieldCorrectionsIfEnabled(
    to arrays: AssemblyArrays,
    geom: Geometry,
    neumann: [Complex32],
    k: Float,
    kImag: Float = 0.0,
    robinBetas: [Complex32]? = nil
) throws -> (AssemblyArrays, NearQuadratureStats?) {
    guard let config = try requestedNearQuadratureConfig() else {
        return (arrays, nil)
    }
    let (corrected, stats) = try applyNearFieldCorrectionsCPU(
        to: arrays,
        geom: geom,
        neumann: neumann,
        k: k,
        kImag: kImag,
        robinBetas: robinBetas,
        config: config
    )
    return (corrected, stats)
}

func applyDuffyCorrections(
    to arrays: AssemblyArrays,
    geom: Geometry,
    neumann: [Complex32],
    k: Float,
    kImag: Float = 0.0,
    robinBetas: [Complex32]? = nil,
    residentContext: ResidentMetalContext? = nil
) throws -> (AssemblyArrays, DuffyCorrectionStats) {
    let mode = ProcessInfo.processInfo.environment[
        "HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE"
    ] ?? "gpu_blocks"
    if mode == "cpu" {
        return try applyDuffyCorrectionsCPU(
            to: arrays,
            geom: geom,
            neumann: neumann,
            k: k,
            kImag: kImag,
            robinBetas: robinBetas
        )
    }
    if mode != "gpu_blocks" {
        try fail("unsupported native Duffy mode: \(mode)")
    }

    let start = CFAbsoluteTimeGetCurrent()
    let pairList: DuffyPairList
    if let residentContext {
        pairList = residentContext.pairList
    } else {
        pairList = try buildDuffyPairList(geom)
    }
    let blockStart = CFAbsoluteTimeGetCurrent()
    let blocks: MetalDuffyBlockOutput
    if let residentContext {
        blocks = try residentContext.computeDuffyDeltaBlocksMetal(k: k, kImag: kImag)
    } else {
        let rules = [
            1: try duffyRule(kind: 1),
            2: try duffyRule(kind: 2),
            3: try duffyRule(kind: 3),
        ]
        blocks = try computeDuffyDeltaBlocksMetal(
            geom: geom,
            pairList: pairList,
            rules: rules,
            k: k,
            kImag: kImag
        )
    }
    let blockSeconds = CFAbsoluteTimeGetCurrent() - blockStart

    let correctedArrays: AssemblyArrays
    let reductionSeconds: Double
    let uniqueTriplets: Int
    let imagePairs: Int
    if let residentContext {
        (correctedArrays, reductionSeconds) = residentContext.reduceDuffyDeltaBlocks(
            to: arrays,
            neumann: neumann,
            blocks: blocks,
            k: k,
            kImag: kImag,
            robinBetas: robinBetas
        )
        uniqueTriplets = residentContext.duffyReductionPlan.matrixIndices.count
        imagePairs = residentContext.duffyReductionPlan.imagePairs
    } else {
        let reductionStart = CFAbsoluteTimeGetCurrent()
        let n = geom.p1DofCount
        let pairCount = pairList.pairs.count
        var aRe = arrays.aRe
        var aIm = arrays.aIm
        var rhsRe = arrays.rhsRe
        var rhsIm = arrays.rhsIm
        var triplets: [Int64: (Double, Double)] = [:]
        triplets.reserveCapacity(pairList.plan.total * 3)
        let iK = Complex32(re: -kImag, im: k)

        for pairIndex in pairList.pairs.indices {
            let pair = pairList.pairs[pairIndex]
            let gTrial = neumann[pair.trial]
            let betaTrial = robinBetas?[pair.trial] ?? Complex32.zero
            let robinCoupling = iK * betaTrial
            let hasRobin = betaTrial.re != 0.0 || betaTrial.im != 0.0
            for i in 0..<3 {
                let row = geom.p1Dof(pair.test, i)
                // Weight 1: image-mask pair enumeration already carries the symmetry factor.
                let rowWeight: Float = 1.0
                let slpDelta = Complex32(
                    re: blocks.slpRe[pairIndex + i * pairCount],
                    im: blocks.slpIm[pairIndex + i * pairCount]
                )
                let rhsDelta = (slpDelta * gTrial) * rowWeight
                rhsRe[row] += rhsDelta.re
                rhsIm[row] += rhsDelta.im

                for j in 0..<3 {
                    let col = geom.p1Dof(pair.trial, j)
                    let deltaIndex = pairIndex + (i * 3 + j) * pairCount
                    var delta = Complex32(
                        re: blocks.dlpRe[deltaIndex],
                        im: blocks.dlpIm[deltaIndex]
                    )
                    if hasRobin {
                        delta = delta - ((slpDelta * robinCoupling) * Float(1.0 / 3.0))
                    }
                    delta = delta * rowWeight
                    let key = Int64(row) + Int64(col) * Int64(n)
                    let current = triplets[key] ?? (0.0, 0.0)
                    triplets[key] = (
                        current.0 + Double(delta.re),
                        current.1 + Double(delta.im)
                    )
                }
            }
        }

        for (key, value) in triplets {
            let row = Int(key % Int64(n))
            let col = Int(key / Int64(n))
            let idx = row * n + col
            aRe[idx] += Float(value.0)
            aIm[idx] += Float(value.1)
        }

        correctedArrays = AssemblyArrays(aRe: aRe, aIm: aIm, rhsRe: rhsRe, rhsIm: rhsIm)
        reductionSeconds = CFAbsoluteTimeGetCurrent() - reductionStart
        uniqueTriplets = triplets.count
        imagePairs = pairList.pairs.filter { $0.testImageMask != 0 || $0.trialImageMask != 0 }.count
    }
    let stats = DuffyCorrectionStats(
        plan: pairList.plan,
        rawTriplets: pairList.plan.total * 9,
        uniqueTriplets: uniqueTriplets,
        seconds: CFAbsoluteTimeGetCurrent() - start,
        implementation: "metal_duffy_blocks_cpu_reduction",
        blockSeconds: blockSeconds,
        reductionSeconds: reductionSeconds,
        dispatch: blocks.dispatch,
        imagePairs: imagePairs,
        reductionPrecomputed: residentContext != nil,
        reductionPlanBuildSeconds: residentContext?.duffyReductionPlanBuildSeconds
    )
    return (
        correctedArrays,
        stats
    )
}

struct MetalKernelParams {
    var nDof: Int32
    var nTriangles: Int32
    var maxInc: Int32
    var symmetryPlane: Int32
    var k: Float
    var kImag: Float
    var hasRobin: Int32
}

let regularAssemblyMetalSource = """
#include <metal_stdlib>
using namespace metal;

// Specialized per session by specializedAssemblyMetalSource(): the symmetry
// plane is a compile-time constant so the compiler prunes the image-mask
// loops entirely for the planes that are not active.
constant int SYMMETRY_PLANE = 0;

struct Params {
    int nDof;
    int nTriangles;
    int maxInc;
    int symmetryPlane; // unused by kernels (see SYMMETRY_PLANE); kept for layout
    float k;
    float kImag;
    int hasRobin;
};

constant float qx[6] = {
    0.4459484909159651f, 0.0915762135097710f,
    0.1081030181680700f, 0.4459484909159651f,
    0.8168475729804590f, 0.0915762135097710f
};

constant float qy[6] = {
    0.4459484909159651f, 0.0915762135097700f,
    0.4459484909159651f, 0.1081030181680700f,
    0.0915762135097700f, 0.8168475729804580f
};

constant float qw[6] = {
    0.1116907948390055f, 0.0549758718276610f,
    0.1116907948390055f, 0.1116907948390055f,
    0.0549758718276610f, 0.0549758718276610f
};

inline int tri_vertex(device const int *triangles, int nTriangles, int tri, int local) {
    return triangles[local * nTriangles + tri];
}

inline float basis_value(float xi, float eta, int local) {
    if (local == 0) {
        return 1.0f - xi - eta;
    }
    if (local == 1) {
        return xi;
    }
    return eta;
}

inline float3 point_on_triangle(
    device const float *px,
    device const float *py,
    device const float *pz,
    device const int *triangles,
    int nTriangles,
    int tri,
    float xi,
    float eta
) {
    float b1 = 1.0f - xi - eta;
    float b2 = xi;
    float b3 = eta;
    int i1 = tri_vertex(triangles, nTriangles, tri, 0);
    int i2 = tri_vertex(triangles, nTriangles, tri, 1);
    int i3 = tri_vertex(triangles, nTriangles, tri, 2);
    return float3(
        b1 * px[i1] + b2 * px[i2] + b3 * px[i3],
        b1 * py[i1] + b2 * py[i2] + b3 * py[i3],
        b1 * pz[i1] + b2 * pz[i2] + b3 * pz[i3]
    );
}

inline float2 c_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 helmholtz_g(float3 delta, float k, float kImag) {
    float r2 = dot(delta, delta);
    if (r2 <= 1.0e-14f) {
        return float2(0.0f, 0.0f);
    }
    float r = sqrt(r2);
    float phase = k * r;
    float scale = 0.07957747154594767f / r;
    if (kImag != 0.0f) {
        scale *= exp(-kImag * r);
    }
    return float2(cos(phase) * scale, sin(phase) * scale);
}

inline float2 helmholtz_dlp(float3 delta, float3 normal, float k, float kImag) {
    float r2 = dot(delta, delta);
    if (r2 <= 1.0e-14f) {
        return float2(0.0f, 0.0f);
    }
    float r = sqrt(r2);
    float phase = k * r;
    float scale = 0.07957747154594767f / r;
    if (kImag != 0.0f) {
        scale *= exp(-kImag * r);
    }
    float gre = cos(phase) * scale;
    float gim = sin(phase) * scale;
    float projection = dot(delta, normal) / r;
    float fre = -1.0f / r;
    if (kImag != 0.0f) {
        fre -= kImag;
    }
    float fim = k;
    return float2(
        (gre * fre - gim * fim) * projection,
        (gre * fim + gim * fre) * projection
    );
}

inline bool has_image_mask(int symmetryPlane, int mask) {
    return (symmetryPlane & mask) == mask;
}

inline float3 mirror_point(float3 point, int mask) {
    if ((mask & 1) != 0) {
        point.x = -point.x;
    }
    if ((mask & 2) != 0) {
        point.y = -point.y;
    }
    if ((mask & 4) != 0) {
        point.z = -point.z;
    }
    return point;
}

inline float3 mirror_normal(float3 normal, int mask) {
    if ((mask & 1) != 0) {
        normal.x = -normal.x;
    }
    if ((mask & 2) != 0) {
        normal.y = -normal.y;
    }
    if ((mask & 4) != 0) {
        normal.z = -normal.z;
    }
    return normal;
}

inline float p1_row_weight(
    int row,
    device const float *px,
    device const float *py,
    device const int *triangles,
    device const int *incTri,
    device const int *incLoc,
    constant Params &params
) {
    if (SYMMETRY_PLANE == 0) {
        return 1.0f;
    }
    float weight = 1.0f;
    if ((SYMMETRY_PLANE & 1) != 0) {
        weight *= 2.0f;
    }
    if ((SYMMETRY_PLANE & 2) != 0) {
        weight *= 2.0f;
    }
    if ((SYMMETRY_PLANE & 4) != 0) {
        weight *= 2.0f;
    }
    return weight;
}

inline float2 regular_dlp_entry(
    device const float *px,
    device const float *py,
    device const float *pz,
    device const int *triangles,
    device const float *normals,
    device const float *areas,
    int nTriangles,
    int testTri,
    int trialTri,
    int testLocal,
    int trialLocal,
    int symmetryPlane,
    float k,
    float kImag
) {
    float jac = (2.0f * areas[testTri]) * (2.0f * areas[trialTri]);
    float3 normal = float3(
        normals[trialTri],
        normals[nTriangles + trialTri],
        normals[2 * nTriangles + trialTri]
    );
    float2 acc = float2(0.0f, 0.0f);
    for (int a = 0; a < 6; ++a) {
        float3 testPoint = point_on_triangle(
            px, py, pz, triangles, nTriangles, testTri, qx[a], qy[a]);
        float tb = basis_value(qx[a], qy[a], testLocal);
        for (int b = 0; b < 6; ++b) {
            float3 trialPoint = point_on_triangle(
                px, py, pz, triangles, nTriangles, trialTri, qx[b], qy[b]);
            float sb = basis_value(qx[b], qy[b], trialLocal);
            float weight = tb * sb * qw[a] * qw[b] * jac;
            // Coincident self-point excluded by index; see
            // assemble_matrix_pair_atomic for why the r2 guard is not enough.
            if (testTri != trialTri || a != b) {
                acc += helmholtz_dlp(trialPoint - testPoint, normal, k, kImag) * weight;
            }
            if (symmetryPlane != 0) {
                for (int mask = 1; mask <= 7; ++mask) {
                    if (!has_image_mask(symmetryPlane, mask)) {
                        continue;
                    }
                    acc += helmholtz_dlp(
                        mirror_point(trialPoint, mask) - testPoint,
                        mirror_normal(normal, mask),
                        k,
                        kImag
                    ) * weight;
                }
            }
        }
    }
    return acc;
}

inline float2 regular_slp_entry(
    device const float *px,
    device const float *py,
    device const float *pz,
    device const int *triangles,
    device const float *areas,
    int nTriangles,
    int testTri,
    int trialTri,
    int testLocal,
    int symmetryPlane,
    float k,
    float kImag
) {
    float jac = (2.0f * areas[testTri]) * (2.0f * areas[trialTri]);
    float2 acc = float2(0.0f, 0.0f);
    for (int a = 0; a < 6; ++a) {
        float3 testPoint = point_on_triangle(
            px, py, pz, triangles, nTriangles, testTri, qx[a], qy[a]);
        float tb = basis_value(qx[a], qy[a], testLocal);
        for (int b = 0; b < 6; ++b) {
            float3 trialPoint = point_on_triangle(
                px, py, pz, triangles, nTriangles, trialTri, qx[b], qy[b]);
            float weight = tb * qw[a] * qw[b] * jac;
            // Coincident self-point excluded by index; see
            // assemble_matrix_pair_atomic for why the r2 guard is not enough.
            if (testTri != trialTri || a != b) {
                acc += helmholtz_g(trialPoint - testPoint, k, kImag) * weight;
            }
            if (symmetryPlane != 0) {
                for (int mask = 1; mask <= 7; ++mask) {
                    if (!has_image_mask(symmetryPlane, mask)) {
                        continue;
                    }
                    acc += helmholtz_g(
                        mirror_point(trialPoint, mask) - testPoint,
                        k,
                        kImag
                    ) * weight;
                }
            }
        }
    }
    return acc;
}

kernel void assemble_matrix_regular(
    device float *outRe [[buffer(0)]],
    device float *outIm [[buffer(1)]],
    device const float *px [[buffer(2)]],
    device const float *py [[buffer(3)]],
    device const float *pz [[buffer(4)]],
    device const int *triangles [[buffer(5)]],
    device const int *p1Local2Global [[buffer(6)]],
    device const float *normals [[buffer(7)]],
    device const float *areas [[buffer(8)]],
    device const int *incTri [[buffer(9)]],
    device const int *incLoc [[buffer(10)]],
    device const int *counts [[buffer(11)]],
    constant Params &params [[buffer(12)]],
    device const float *robinBetaRe [[buffer(13)]],
    device const float *robinBetaIm [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    int total = params.nDof * params.nDof;
    if (gid >= uint(total)) {
        return;
    }
    int idx = int(gid);
    int row = idx / params.nDof;
    int col = idx - row * params.nDof;
    float2 acc = float2(0.0f, 0.0f);
    float rowWeight = p1_row_weight(
        row, px, py, triangles, incTri, incLoc, params);
    for (int rs = 0; rs < counts[row]; ++rs) {
        int testTri = incTri[row * params.maxInc + rs];
        int testLocal = incLoc[row * params.maxInc + rs];
        for (int cs = 0; cs < counts[col]; ++cs) {
            int trialTri = incTri[col * params.maxInc + cs];
            int trialLocal = incLoc[col * params.maxInc + cs];
            acc += regular_dlp_entry(
                px, py, pz, triangles, normals, areas, params.nTriangles,
                testTri, trialTri, testLocal, trialLocal,
                SYMMETRY_PLANE, params.k, params.kImag);
            if (params.hasRobin != 0) {
                float2 beta = float2(robinBetaRe[trialTri], robinBetaIm[trialTri]);
                if (beta.x != 0.0f || beta.y != 0.0f) {
                    float2 iK = float2(-params.kImag, params.k);
                    float2 robinFactor = c_mul(iK, beta) * (-0.33333333333333333f);
                    float2 slp = regular_slp_entry(
                        px, py, pz, triangles, areas, params.nTriangles,
                        testTri, trialTri, testLocal, SYMMETRY_PLANE,
                        params.k, params.kImag);
                    acc += c_mul(robinFactor, slp);
                }
            }
            if (testTri == trialTri) {
                float mass = areas[testTri] * (testLocal == trialLocal ? 0.16666666666666666f : 0.08333333333333333f);
                acc.x -= 0.5f * mass;
            }
        }
    }
    outRe[idx] = acc.x * rowWeight;
    outIm[idx] = acc.y * rowWeight;
}

kernel void assemble_rhs_source_regular(
    device float *outRe [[buffer(0)]],
    device float *outIm [[buffer(1)]],
    device const float *px [[buffer(2)]],
    device const float *py [[buffer(3)]],
    device const float *pz [[buffer(4)]],
    device const int *triangles [[buffer(5)]],
    device const float *areas [[buffer(6)]],
    device const int *incTri [[buffer(7)]],
    device const int *incLoc [[buffer(8)]],
    device const int *counts [[buffer(9)]],
    device const int *sourceTris [[buffer(10)]],
    device const float *sourceRe [[buffer(11)]],
    device const float *sourceIm [[buffer(12)]],
    constant Params &params [[buffer(13)]],
    constant int &sourceCount [[buffer(14)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(params.nDof)) {
        return;
    }
    int row = int(gid);
    float2 acc = float2(0.0f, 0.0f);
    float rowWeight = p1_row_weight(
        row, px, py, triangles, incTri, incLoc, params);
    for (int rs = 0; rs < counts[row]; ++rs) {
        int testTri = incTri[row * params.maxInc + rs];
        int testLocal = incLoc[row * params.maxInc + rs];
        for (int src = 0; src < sourceCount; ++src) {
            int trialTri = sourceTris[src];
            float2 slp = regular_slp_entry(
                px, py, pz, triangles, areas, params.nTriangles,
                testTri, trialTri, testLocal, SYMMETRY_PLANE, params.k, params.kImag);
            acc += c_mul(slp, float2(sourceRe[src], sourceIm[src]));
        }
    }
    outRe[row] = acc.x * rowWeight;
    outIm[row] = acc.y * rowWeight;
}

kernel void assemble_pair_blocks_regular(
    device float *dlpRe [[buffer(0)]],
    device float *dlpIm [[buffer(1)]],
    device float *slpRe [[buffer(2)]],
    device float *slpIm [[buffer(3)]],
    device const float *px [[buffer(4)]],
    device const float *py [[buffer(5)]],
    device const float *pz [[buffer(6)]],
    device const int *triangles [[buffer(7)]],
    device const float *normals [[buffer(8)]],
    device const float *areas [[buffer(9)]],
    constant Params &params [[buffer(10)]],
    constant int &pairCount [[buffer(11)]],
    device const float *robinBetaRe [[buffer(12)]],
    device const float *robinBetaIm [[buffer(13)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(pairCount)) {
        return;
    }
    int pairIndex = int(gid);
    int testTri = pairIndex / params.nTriangles;
    int trialTri = pairIndex - testTri * params.nTriangles;
    float jac = (2.0f * areas[testTri]) * (2.0f * areas[trialTri]);
    float3 normal = float3(
        normals[trialTri],
        normals[params.nTriangles + trialTri],
        normals[2 * params.nTriangles + trialTri]
    );

    float2 localSlp0 = float2(0.0f, 0.0f);
    float2 localSlp1 = float2(0.0f, 0.0f);
    float2 localSlp2 = float2(0.0f, 0.0f);
    float2 localDlp00 = float2(0.0f, 0.0f);
    float2 localDlp01 = float2(0.0f, 0.0f);
    float2 localDlp02 = float2(0.0f, 0.0f);
    float2 localDlp10 = float2(0.0f, 0.0f);
    float2 localDlp11 = float2(0.0f, 0.0f);
    float2 localDlp12 = float2(0.0f, 0.0f);
    float2 localDlp20 = float2(0.0f, 0.0f);
    float2 localDlp21 = float2(0.0f, 0.0f);
    float2 localDlp22 = float2(0.0f, 0.0f);
    float2 beta = float2(0.0f, 0.0f);
    bool pairHasRobin = false;
    if (params.hasRobin != 0) {
        beta = float2(robinBetaRe[trialTri], robinBetaIm[trialTri]);
        pairHasRobin = (beta.x != 0.0f || beta.y != 0.0f);
    }

    for (int a = 0; a < 6; ++a) {
        float3 testPoint = point_on_triangle(
            px, py, pz, triangles, params.nTriangles, testTri, qx[a], qy[a]);
        float tb0 = basis_value(qx[a], qy[a], 0);
        float tb1 = basis_value(qx[a], qy[a], 1);
        float tb2 = basis_value(qx[a], qy[a], 2);
        for (int b = 0; b < 6; ++b) {
            float3 trialPoint = point_on_triangle(
                px, py, pz, triangles, params.nTriangles, trialTri, qx[b], qy[b]);
            float sb0 = basis_value(qx[b], qy[b], 0);
            float sb1 = basis_value(qx[b], qy[b], 1);
            float sb2 = basis_value(qx[b], qy[b], 2);
            float weight = qw[a] * qw[b] * jac;
            // Coincident self-point excluded by index; see
            // assemble_matrix_pair_atomic for why the r2 guard is not enough.
            float2 g = float2(0.0f, 0.0f);
            float2 d = float2(0.0f, 0.0f);
            if (testTri != trialTri || a != b) {
                g = helmholtz_g(trialPoint - testPoint, params.k, params.kImag) * weight;
                d = helmholtz_dlp(trialPoint - testPoint, normal, params.k, params.kImag) * weight;
            }
            localSlp0 += g * tb0;
            localSlp1 += g * tb1;
            localSlp2 += g * tb2;
            localDlp00 += d * (tb0 * sb0);
            localDlp01 += d * (tb0 * sb1);
            localDlp02 += d * (tb0 * sb2);
            localDlp10 += d * (tb1 * sb0);
            localDlp11 += d * (tb1 * sb1);
            localDlp12 += d * (tb1 * sb2);
            localDlp20 += d * (tb2 * sb0);
            localDlp21 += d * (tb2 * sb1);
            localDlp22 += d * (tb2 * sb2);
            if (SYMMETRY_PLANE != 0) {
                for (int mask = 1; mask <= 7; ++mask) {
                    if (!has_image_mask(SYMMETRY_PLANE, mask)) {
                        continue;
                    }
                    float3 imagePoint = mirror_point(trialPoint, mask);
                    float3 imageNormal = mirror_normal(normal, mask);
                    float2 imageG = helmholtz_g(imagePoint - testPoint, params.k, params.kImag) * weight;
                    float2 imageD = helmholtz_dlp(imagePoint - testPoint, imageNormal, params.k, params.kImag) * weight;
                    localSlp0 += imageG * tb0;
                    localSlp1 += imageG * tb1;
                    localSlp2 += imageG * tb2;
                    localDlp00 += imageD * (tb0 * sb0);
                    localDlp01 += imageD * (tb0 * sb1);
                    localDlp02 += imageD * (tb0 * sb2);
                    localDlp10 += imageD * (tb1 * sb0);
                    localDlp11 += imageD * (tb1 * sb1);
                    localDlp12 += imageD * (tb1 * sb2);
                    localDlp20 += imageD * (tb2 * sb0);
                    localDlp21 += imageD * (tb2 * sb1);
                    localDlp22 += imageD * (tb2 * sb2);
                }
            }
        }
    }

    if (pairHasRobin) {
        float2 iK = float2(-params.kImag, params.k);
        float2 robinFactor = c_mul(iK, beta) * (-0.33333333333333333f);
        float2 slpValues[3] = { localSlp0, localSlp1, localSlp2 };
        float2 localDlp[9] = {
            localDlp00,
            localDlp01,
            localDlp02,
            localDlp10,
            localDlp11,
            localDlp12,
            localDlp20,
            localDlp21,
            localDlp22
        };
        for (int i = 0; i < 3; ++i) {
            float2 contrib = c_mul(robinFactor, slpValues[i]);
            for (int j = 0; j < 3; ++j) {
                localDlp[i * 3 + j] += contrib;
            }
        }
        localDlp00 = localDlp[0];
        localDlp01 = localDlp[1];
        localDlp02 = localDlp[2];
        localDlp10 = localDlp[3];
        localDlp11 = localDlp[4];
        localDlp12 = localDlp[5];
        localDlp20 = localDlp[6];
        localDlp21 = localDlp[7];
        localDlp22 = localDlp[8];
    }

    slpRe[pairIndex] = localSlp0.x;
    slpIm[pairIndex] = localSlp0.y;
    slpRe[pairIndex + pairCount] = localSlp1.x;
    slpIm[pairIndex + pairCount] = localSlp1.y;
    slpRe[pairIndex + 2 * pairCount] = localSlp2.x;
    slpIm[pairIndex + 2 * pairCount] = localSlp2.y;

    float2 dlpValues[9] = {
        localDlp00,
        localDlp01,
        localDlp02,
        localDlp10,
        localDlp11,
        localDlp12,
        localDlp20,
        localDlp21,
        localDlp22
    };
    for (int i = 0; i < 9; ++i) {
        int outIdx = pairIndex + i * pairCount;
        dlpRe[outIdx] = dlpValues[i].x;
        dlpIm[outIdx] = dlpValues[i].y;
    }
}

inline float symmetry_row_weight() {
    float weight = 1.0f;
    if ((SYMMETRY_PLANE & 1) != 0) {
        weight *= 2.0f;
    }
    if ((SYMMETRY_PLANE & 2) != 0) {
        weight *= 2.0f;
    }
    if ((SYMMETRY_PLANE & 4) != 0) {
        weight *= 2.0f;
    }
    return weight;
}

// One thread per triangle pair: the 36 helmholtz_dlp evaluations shared by
// the pair's nine (row, col) entries are computed once and the 3x3 block is
// scattered into A with relaxed atomic float adds, instead of one thread per
// matrix entry recomputing the pair quadrature for every entry. The output
// matrix buffers must be zero-filled before dispatch.
kernel void assemble_matrix_pair_atomic(
    device atomic_float *outRe [[buffer(0)]],
    device atomic_float *outIm [[buffer(1)]],
    device const float *px [[buffer(2)]],
    device const float *py [[buffer(3)]],
    device const float *pz [[buffer(4)]],
    device const int *triangles [[buffer(5)]],
    device const int *p1Local2Global [[buffer(6)]],
    device const float *normals [[buffer(7)]],
    device const float *areas [[buffer(8)]],
    constant Params &params [[buffer(9)]],
    constant int &pairCount [[buffer(10)]],
    device const float *robinBetaRe [[buffer(11)]],
    device const float *robinBetaIm [[buffer(12)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(pairCount)) {
        return;
    }
    int pairIndex = int(gid);
    int testTri = pairIndex / params.nTriangles;
    int trialTri = pairIndex - testTri * params.nTriangles;
    float jac = (2.0f * areas[testTri]) * (2.0f * areas[trialTri]);
    float3 normal = float3(
        normals[trialTri],
        normals[params.nTriangles + trialTri],
        normals[2 * params.nTriangles + trialTri]
    );

    float2 dlp[9];
    for (int i = 0; i < 9; ++i) {
        dlp[i] = float2(0.0f, 0.0f);
    }
    float2 slp[3];
    for (int i = 0; i < 3; ++i) {
        slp[i] = float2(0.0f, 0.0f);
    }
    float2 beta = float2(0.0f, 0.0f);
    bool pairHasRobin = false;
    if (params.hasRobin != 0) {
        beta = float2(robinBetaRe[trialTri], robinBetaIm[trialTri]);
        pairHasRobin = (beta.x != 0.0f || beta.y != 0.0f);
    }
    for (int a = 0; a < 6; ++a) {
        float3 testPoint = point_on_triangle(
            px, py, pz, triangles, params.nTriangles, testTri, qx[a], qy[a]);
        float tb[3] = {
            basis_value(qx[a], qy[a], 0),
            basis_value(qx[a], qy[a], 1),
            basis_value(qx[a], qy[a], 2)
        };
        for (int b = 0; b < 6; ++b) {
            float3 trialPoint = point_on_triangle(
                px, py, pz, triangles, params.nTriangles, trialTri, qx[b], qy[b]);
            float sb[3] = {
                basis_value(qx[b], qy[b], 0),
                basis_value(qx[b], qy[b], 1),
                basis_value(qx[b], qy[b], 2)
            };
            float weight = qw[a] * qw[b] * jac;
            // For the coincident pair the a == b evaluation sits on the
            // kernel singularity and must be excluded by INDEX, not by the
            // r2 guard inside helmholtz_dlp: fast-math FMA contraction can
            // round testPoint and trialPoint differently, leaving a few-ulp
            // garbage delta whose r2 lands just above the guard and whose
            // 1/r2 blows the entry up by ~1e10 (observed on bempp
            // regular_sphere(3)). The CPU paths compute both points with
            // identical arithmetic, get an exactly-zero delta, and return
            // zero from the guard, so skipping the evaluation matches them.
            float2 d = float2(0.0f, 0.0f);
            float2 g = float2(0.0f, 0.0f);
            if (testTri != trialTri || a != b) {
                d = helmholtz_dlp(trialPoint - testPoint, normal, params.k, params.kImag) * weight;
                if (pairHasRobin) {
                    g = helmholtz_g(trialPoint - testPoint, params.k, params.kImag) * weight;
                }
            }
            if (SYMMETRY_PLANE != 0) {
                for (int mask = 1; mask <= 7; ++mask) {
                    if (!has_image_mask(SYMMETRY_PLANE, mask)) {
                        continue;
                    }
                    d += helmholtz_dlp(
                        mirror_point(trialPoint, mask) - testPoint,
                        mirror_normal(normal, mask),
                        params.k,
                        params.kImag
                    ) * weight;
                    if (pairHasRobin) {
                        g += helmholtz_g(
                            mirror_point(trialPoint, mask) - testPoint,
                            params.k,
                            params.kImag
                        ) * weight;
                    }
                }
            }
            for (int i = 0; i < 3; ++i) {
                if (pairHasRobin) {
                    slp[i] += g * tb[i];
                }
                for (int j = 0; j < 3; ++j) {
                    dlp[i * 3 + j] += d * (tb[i] * sb[j]);
                }
            }
        }
    }

    if (pairHasRobin) {
        float2 iK = float2(-params.kImag, params.k);
        float2 robinFactor = c_mul(iK, beta) * (-0.33333333333333333f);
        for (int i = 0; i < 3; ++i) {
            float2 contrib = c_mul(robinFactor, slp[i]);
            for (int j = 0; j < 3; ++j) {
                dlp[i * 3 + j] += contrib;
            }
        }
    }

    if (testTri == trialTri) {
        float area = areas[testTri];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float mass = area * (i == j ? 0.16666666666666666f : 0.08333333333333333f);
                dlp[i * 3 + j].x -= 0.5f * mass;
            }
        }
    }

    float rowWeight = symmetry_row_weight();
    for (int i = 0; i < 3; ++i) {
        int row = p1Local2Global[testTri * 3 + i];
        for (int j = 0; j < 3; ++j) {
            int col = p1Local2Global[trialTri * 3 + j];
            float2 value = dlp[i * 3 + j] * rowWeight;
            int outIdx = row * params.nDof + col;
            atomic_fetch_add_explicit(&outRe[outIdx], value.x, memory_order_relaxed);
            atomic_fetch_add_explicit(&outIm[outIdx], value.y, memory_order_relaxed);
        }
    }
}

kernel void evaluate_field_regular(
    device float *outRe [[buffer(0)]],
    device float *outIm [[buffer(1)]],
    device const float *obs [[buffer(2)]],
    device const float *pressureRe [[buffer(3)]],
    device const float *pressureIm [[buffer(4)]],
    device const float *neumannRe [[buffer(5)]],
    device const float *neumannIm [[buffer(6)]],
    device const float *px [[buffer(7)]],
    device const float *py [[buffer(8)]],
    device const float *pz [[buffer(9)]],
    device const int *triangles [[buffer(10)]],
    device const int *p1Local2Global [[buffer(11)]],
    device const float *normals [[buffer(12)]],
    device const float *areas [[buffer(13)]],
    constant Params &params [[buffer(14)]],
    constant int &nObs [[buffer(15)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(nObs)) {
        return;
    }
    int obsIdx = int(gid);
    float3 obsPoint = float3(obs[obsIdx], obs[nObs + obsIdx], obs[2 * nObs + obsIdx]);
    float2 acc = float2(0.0f, 0.0f);
    for (int tri = 0; tri < params.nTriangles; ++tri) {
        float3 normal = float3(
            normals[tri],
            normals[params.nTriangles + tri],
            normals[2 * params.nTriangles + tri]
        );
        float jac = 2.0f * areas[tri];
        float2 gTri = float2(neumannRe[tri], neumannIm[tri]);
        int dof0 = p1Local2Global[tri * 3];
        int dof1 = p1Local2Global[tri * 3 + 1];
        int dof2 = p1Local2Global[tri * 3 + 2];
        for (int q = 0; q < 6; ++q) {
            float xi = qx[q];
            float eta = qy[q];
            float b0 = 1.0f - xi - eta;
            float b1 = xi;
            float b2 = eta;
            float3 sourcePoint = point_on_triangle(
                px, py, pz, triangles, params.nTriangles, tri, xi, eta);
            float2 pressure = float2(
                b0 * pressureRe[dof0] + b1 * pressureRe[dof1] + b2 * pressureRe[dof2],
                b0 * pressureIm[dof0] + b1 * pressureIm[dof1] + b2 * pressureIm[dof2]
            );
            float2 dlp = helmholtz_dlp(sourcePoint - obsPoint, normal, params.k, 0.0f);
            float2 slp = helmholtz_g(sourcePoint - obsPoint, params.k, 0.0f);
            float weight = qw[q] * jac;
            acc += (c_mul(dlp, pressure) - c_mul(slp, gTri)) * weight;
            if (SYMMETRY_PLANE != 0) {
                for (int mask = 1; mask <= 7; ++mask) {
                    if (!has_image_mask(SYMMETRY_PLANE, mask)) {
                        continue;
                    }
                    float3 imagePoint = mirror_point(sourcePoint, mask);
                    float3 imageNormal = mirror_normal(normal, mask);
                    float2 imageDlp = helmholtz_dlp(imagePoint - obsPoint, imageNormal, params.k, 0.0f);
                    float2 imageSlp = helmholtz_g(imagePoint - obsPoint, params.k, 0.0f);
                    acc += (c_mul(imageDlp, pressure) - c_mul(imageSlp, gTri)) * weight;
                }
            }
        }
    }
    outRe[obsIdx] = acc.x;
    outIm[obsIdx] = acc.y;
}

inline float ref_x(int local) {
    return local == 1 ? 1.0f : 0.0f;
}

inline float ref_y(int local) {
    return local == 2 ? 1.0f : 0.0f;
}

inline float2 remap_duffy_point(float2 point, int kind, int local1, int local2) {
    if (kind == 1) {
        return point;
    }
    if (kind == 2) {
        int vc = 3 - local1 - local2;
        float2 a = float2(ref_x(local1), ref_y(local1));
        float2 b = float2(ref_x(local2), ref_y(local2));
        float2 c = float2(ref_x(vc), ref_y(vc));
        return a + point.x * (b - a) + point.y * (c - a);
    }
    if (local1 == 0) {
        return point;
    }
    if (local1 == 1) {
        return float2(1.0f - point.x - point.y, point.y);
    }
    return float2(point.x, 1.0f - point.x - point.y);
}

inline float2 image_ref_to_original_ref(float2 point, int mask) {
    int bitCount = 0;
    int remaining = mask;
    while (remaining != 0) {
        bitCount += remaining & 1;
        remaining >>= 1;
    }
    if ((bitCount & 1) != 0) {
        return float2(point.y, point.x);
    }
    return point;
}

kernel void duffy_delta_blocks(
    device float *slpRe [[buffer(0)]],
    device float *slpIm [[buffer(1)]],
    device float *dlpRe [[buffer(2)]],
    device float *dlpIm [[buffer(3)]],
    device const float *px [[buffer(4)]],
    device const float *py [[buffer(5)]],
    device const float *pz [[buffer(6)]],
    device const int *triangles [[buffer(7)]],
    device const float *normals [[buffer(8)]],
    device const float *areas [[buffer(9)]],
    device const int *pairTest [[buffer(10)]],
    device const int *pairTrial [[buffer(11)]],
    device const int *pairKind [[buffer(12)]],
    device const int *pairTestImageMask [[buffer(13)]],
    device const int *pairTrialImageMask [[buffer(14)]],
    device const int *pairTestLocal1 [[buffer(15)]],
    device const int *pairTestLocal2 [[buffer(16)]],
    device const int *pairTrialLocal1 [[buffer(17)]],
    device const int *pairTrialLocal2 [[buffer(18)]],
    device const float *ruleTestX [[buffer(19)]],
    device const float *ruleTestY [[buffer(20)]],
    device const float *ruleTrialX [[buffer(21)]],
    device const float *ruleTrialY [[buffer(22)]],
    device const float *ruleWeights [[buffer(23)]],
    device const int *ruleOffsets [[buffer(24)]],
    device const int *ruleCounts [[buffer(25)]],
    constant Params &params [[buffer(26)]],
    constant int &pairCount [[buffer(27)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= uint(pairCount)) {
        return;
    }
    int pairIndex = int(gid);
    int testTri = pairTest[pairIndex];
    int trialTri = pairTrial[pairIndex];
    int kind = pairKind[pairIndex];
    int testImageMask = pairTestImageMask[pairIndex];
    int trialImageMask = pairTrialImageMask[pairIndex];
    int testLocal1 = pairTestLocal1[pairIndex];
    int testLocal2 = pairTestLocal2[pairIndex];
    int trialLocal1 = pairTrialLocal1[pairIndex];
    int trialLocal2 = pairTrialLocal2[pairIndex];
    float jac = (2.0f * areas[testTri]) * (2.0f * areas[trialTri]);
    float3 normal = mirror_normal(float3(
        normals[trialTri],
        normals[params.nTriangles + trialTri],
        normals[2 * params.nTriangles + trialTri]
    ), trialImageMask);

    float2 regSlp0 = float2(0.0f, 0.0f);
    float2 regSlp1 = float2(0.0f, 0.0f);
    float2 regSlp2 = float2(0.0f, 0.0f);
    float2 regDlp00 = float2(0.0f, 0.0f);
    float2 regDlp01 = float2(0.0f, 0.0f);
    float2 regDlp02 = float2(0.0f, 0.0f);
    float2 regDlp10 = float2(0.0f, 0.0f);
    float2 regDlp11 = float2(0.0f, 0.0f);
    float2 regDlp12 = float2(0.0f, 0.0f);
    float2 regDlp20 = float2(0.0f, 0.0f);
    float2 regDlp21 = float2(0.0f, 0.0f);
    float2 regDlp22 = float2(0.0f, 0.0f);

    for (int a = 0; a < 6; ++a) {
        float2 testRef = image_ref_to_original_ref(float2(qx[a], qy[a]), testImageMask);
        float3 testPoint = point_on_triangle(
            px, py, pz, triangles, params.nTriangles, testTri, testRef.x, testRef.y);
        testPoint = mirror_point(testPoint, testImageMask);
        float tb0 = basis_value(testRef.x, testRef.y, 0);
        float tb1 = basis_value(testRef.x, testRef.y, 1);
        float tb2 = basis_value(testRef.x, testRef.y, 2);
        for (int b = 0; b < 6; ++b) {
            float2 trialRef = image_ref_to_original_ref(float2(qx[b], qy[b]), trialImageMask);
            float3 trialPoint = point_on_triangle(
                px, py, pz, triangles, params.nTriangles, trialTri, trialRef.x, trialRef.y);
            trialPoint = mirror_point(trialPoint, trialImageMask);
            float sb0 = basis_value(trialRef.x, trialRef.y, 0);
            float sb1 = basis_value(trialRef.x, trialRef.y, 1);
            float sb2 = basis_value(trialRef.x, trialRef.y, 2);
            float w = qw[a] * qw[b] * jac;
            // The regular half of the delta must exclude the same coincident
            // self-point the regular assembly kernels exclude, or the
            // subtraction no longer cancels the regular contribution; see
            // assemble_matrix_pair_atomic.
            float2 g = float2(0.0f, 0.0f);
            float2 d = float2(0.0f, 0.0f);
            if (testTri != trialTri || testImageMask != trialImageMask || a != b) {
                g = helmholtz_g(trialPoint - testPoint, params.k, params.kImag) * w;
                d = helmholtz_dlp(trialPoint - testPoint, normal, params.k, params.kImag) * w;
            }
            regSlp0 += g * tb0;
            regSlp1 += g * tb1;
            regSlp2 += g * tb2;
            regDlp00 += d * (tb0 * sb0);
            regDlp01 += d * (tb0 * sb1);
            regDlp02 += d * (tb0 * sb2);
            regDlp10 += d * (tb1 * sb0);
            regDlp11 += d * (tb1 * sb1);
            regDlp12 += d * (tb1 * sb2);
            regDlp20 += d * (tb2 * sb0);
            regDlp21 += d * (tb2 * sb1);
            regDlp22 += d * (tb2 * sb2);
        }
    }

    float2 singSlp0 = float2(0.0f, 0.0f);
    float2 singSlp1 = float2(0.0f, 0.0f);
    float2 singSlp2 = float2(0.0f, 0.0f);
    float2 singDlp00 = float2(0.0f, 0.0f);
    float2 singDlp01 = float2(0.0f, 0.0f);
    float2 singDlp02 = float2(0.0f, 0.0f);
    float2 singDlp10 = float2(0.0f, 0.0f);
    float2 singDlp11 = float2(0.0f, 0.0f);
    float2 singDlp12 = float2(0.0f, 0.0f);
    float2 singDlp20 = float2(0.0f, 0.0f);
    float2 singDlp21 = float2(0.0f, 0.0f);
    float2 singDlp22 = float2(0.0f, 0.0f);

    int ruleIndex = kind - 1;
    int offset = ruleOffsets[ruleIndex];
    int count = ruleCounts[ruleIndex];
    for (int q = 0; q < count; ++q) {
        int idx = offset + q;
        float2 testRef = remap_duffy_point(
            float2(ruleTestX[idx], ruleTestY[idx]), kind, testLocal1, testLocal2);
        float2 trialRef = remap_duffy_point(
            float2(ruleTrialX[idx], ruleTrialY[idx]), kind, trialLocal1, trialLocal2);
        float2 testOrigRef = image_ref_to_original_ref(testRef, testImageMask);
        float2 trialOrigRef = image_ref_to_original_ref(trialRef, trialImageMask);
        float3 testPoint = point_on_triangle(
            px, py, pz, triangles, params.nTriangles, testTri, testOrigRef.x, testOrigRef.y);
        testPoint = mirror_point(testPoint, testImageMask);
        float3 trialPoint = point_on_triangle(
            px, py, pz, triangles, params.nTriangles, trialTri, trialOrigRef.x, trialOrigRef.y);
        trialPoint = mirror_point(trialPoint, trialImageMask);
        float tb0 = basis_value(testOrigRef.x, testOrigRef.y, 0);
        float tb1 = basis_value(testOrigRef.x, testOrigRef.y, 1);
        float tb2 = basis_value(testOrigRef.x, testOrigRef.y, 2);
        float sb0 = basis_value(trialOrigRef.x, trialOrigRef.y, 0);
        float sb1 = basis_value(trialOrigRef.x, trialOrigRef.y, 1);
        float sb2 = basis_value(trialOrigRef.x, trialOrigRef.y, 2);
        float w = ruleWeights[idx] * jac;
        float2 g = helmholtz_g(trialPoint - testPoint, params.k, params.kImag) * w;
        float2 d = helmholtz_dlp(trialPoint - testPoint, normal, params.k, params.kImag) * w;
        singSlp0 += g * tb0;
        singSlp1 += g * tb1;
        singSlp2 += g * tb2;
        singDlp00 += d * (tb0 * sb0);
        singDlp01 += d * (tb0 * sb1);
        singDlp02 += d * (tb0 * sb2);
        singDlp10 += d * (tb1 * sb0);
        singDlp11 += d * (tb1 * sb1);
        singDlp12 += d * (tb1 * sb2);
        singDlp20 += d * (tb2 * sb0);
        singDlp21 += d * (tb2 * sb1);
        singDlp22 += d * (tb2 * sb2);
    }

    float2 dSlp0 = singSlp0 - regSlp0;
    float2 dSlp1 = singSlp1 - regSlp1;
    float2 dSlp2 = singSlp2 - regSlp2;
    slpRe[pairIndex] = dSlp0.x;
    slpIm[pairIndex] = dSlp0.y;
    slpRe[pairIndex + pairCount] = dSlp1.x;
    slpIm[pairIndex + pairCount] = dSlp1.y;
    slpRe[pairIndex + 2 * pairCount] = dSlp2.x;
    slpIm[pairIndex + 2 * pairCount] = dSlp2.y;

    float2 dlpValues[9] = {
        singDlp00 - regDlp00,
        singDlp01 - regDlp01,
        singDlp02 - regDlp02,
        singDlp10 - regDlp10,
        singDlp11 - regDlp11,
        singDlp12 - regDlp12,
        singDlp20 - regDlp20,
        singDlp21 - regDlp21,
        singDlp22 - regDlp22
    };
    for (int i = 0; i < 9; ++i) {
        int outIdx = pairIndex + i * pairCount;
        dlpRe[outIdx] = dlpValues[i].x;
        dlpIm[outIdx] = dlpValues[i].y;
    }
}
"""

func specializedAssemblyMetalSource(symmetryPlaneCode: Int32) -> String {
    regularAssemblyMetalSource.replacingOccurrences(
        of: "constant int SYMMETRY_PLANE = 0;",
        with: "constant int SYMMETRY_PLANE = \(symmetryPlaneCode);"
    )
}

final class MetalWarmup: @unchecked Sendable {
    static let shared = MetalWarmup()

    private let lock = NSLock()
    private let semaphore = DispatchSemaphore(value: 0)
    private var started = false
    private var completed = false
    private var cachedDevice: MTLDevice?

    func begin() {
        lock.lock()
        if started {
            lock.unlock()
            return
        }
        started = true
        lock.unlock()

        Thread.detachNewThread { [self] in
            let device = MTLCreateSystemDefaultDevice()
            lock.lock()
            cachedDevice = device
            completed = true
            lock.unlock()
            semaphore.signal()
        }
    }

    func device() throws -> MTLDevice {
        lock.lock()
        if let cachedDevice {
            lock.unlock()
            return cachedDevice
        }
        if started {
            if completed {
                lock.unlock()
                try fail("Metal device unavailable")
            }
            lock.unlock()
            semaphore.wait()
            semaphore.signal()
            lock.lock()
            let device = cachedDevice
            lock.unlock()
            guard let device else {
                try fail("Metal device unavailable")
            }
            return device
        }
        started = true
        lock.unlock()

        let device = MTLCreateSystemDefaultDevice()
        lock.lock()
        cachedDevice = device
        completed = true
        lock.unlock()
        semaphore.signal()
        guard let device else {
            try fail("Metal device unavailable")
        }
        return device
    }
}

final class AssemblyLibraryCache: @unchecked Sendable {
    static let shared = AssemblyLibraryCache()

    private let lock = NSLock()
    private var libraries: [Int32: MTLLibrary] = [:]

    func library(device: MTLDevice, symmetryPlaneCode: Int32) throws -> MTLLibrary {
        lock.lock()
        defer {
            lock.unlock()
        }
        if let library = libraries[symmetryPlaneCode] {
            return library
        }

        let library = try device.makeLibrary(
            source: specializedAssemblyMetalSource(symmetryPlaneCode: symmetryPlaneCode),
            options: nil
        )
        libraries[symmetryPlaneCode] = library
        return library
    }
}

func assemblyLibrary(device: MTLDevice, symmetryPlaneCode: Int32) throws -> MTLLibrary {
    try AssemblyLibraryCache.shared.library(device: device, symmetryPlaneCode: symmetryPlaneCode)
}

func makeBuffer<T>(_ device: MTLDevice, _ values: [T], label: String) throws -> MTLBuffer {
    let byteCount = values.count * MemoryLayout<T>.stride
    if byteCount <= 0 {
        try fail("\(label) buffer must not be empty")
    }
    let buffer = values.withUnsafeBufferPointer { ptr -> MTLBuffer? in
        device.makeBuffer(
            bytes: ptr.baseAddress!,
            length: byteCount,
            options: .storageModeShared
        )
    }
    guard let buffer else {
        try fail("failed to allocate Metal buffer \(label)")
    }
    buffer.label = label
    return buffer
}

func makeOutputBuffer(_ device: MTLDevice, count: Int, label: String) throws -> MTLBuffer {
    let byteCount = count * MemoryLayout<Float>.stride
    guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
        try fail("failed to allocate Metal output buffer \(label)")
    }
    buffer.label = label
    memset(buffer.contents(), 0, byteCount)
    return buffer
}

func readFloatBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
    let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: ptr, count: count))
}

func makeRobinBetaBuffers(
    device: MTLDevice,
    robinBetas: [Complex32]?,
    nTriangles: Int,
    labelPrefix: String
) throws -> (re: MTLBuffer, im: MTLBuffer, hasRobin: Int32) {
    guard let robinBetas else {
        let zero = [Float(0.0)]
        return (
            try makeBuffer(device, zero, label: "\(labelPrefix)_robin_beta_re_zero"),
            try makeBuffer(device, zero, label: "\(labelPrefix)_robin_beta_im_zero"),
            0
        )
    }
    if robinBetas.count != nTriangles {
        try fail("robin beta count \(robinBetas.count) does not match triangle count \(nTriangles)")
    }
    let hasRobin = robinBetas.contains { $0.re != 0.0 || $0.im != 0.0 }
    if !hasRobin {
        let zero = [Float(0.0)]
        return (
            try makeBuffer(device, zero, label: "\(labelPrefix)_robin_beta_re_zero"),
            try makeBuffer(device, zero, label: "\(labelPrefix)_robin_beta_im_zero"),
            0
        )
    }
    return (
        try makeBuffer(device, robinBetas.map { $0.re }, label: "\(labelPrefix)_robin_beta_re"),
        try makeBuffer(device, robinBetas.map { $0.im }, label: "\(labelPrefix)_robin_beta_im"),
        1
    )
}

func computeDuffyDeltaBlocksMetal(
    geom: Geometry,
    pairList: DuffyPairList,
    rules: [Int: DuffyRule],
    k: Float,
    kImag: Float = 0.0
) throws -> MetalDuffyBlockOutput {
    guard !pairList.pairs.isEmpty else {
        return MetalDuffyBlockOutput(
            slpRe: [],
            slpIm: [],
            dlpRe: [],
            dlpIm: [],
            dispatch: [
                "pairs": 0,
                "kernel": "duffy_delta_blocks",
            ]
        )
    }
    let device = try MetalWarmup.shared.device()
    guard let commandQueue = device.makeCommandQueue() else {
        try fail("failed to create Metal command queue")
    }
    let library = try assemblyLibrary(device: device, symmetryPlaneCode: geom.symmetryPlaneCode)
    guard let function = library.makeFunction(name: "duffy_delta_blocks") else {
        try fail("failed to load Metal Duffy kernel")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    let pairCount = pairList.pairs.count

    let pairTest = pairList.pairs.map { Int32($0.test) }
    let pairTrial = pairList.pairs.map { Int32($0.trial) }
    let pairKind = pairList.pairs.map { Int32($0.kind) }
    let pairTestImageMask = pairList.pairs.map { Int32($0.testImageMask) }
    let pairTrialImageMask = pairList.pairs.map { Int32($0.trialImageMask) }
    let pairTestLocal1 = pairList.pairs.map { Int32($0.testLocal1) }
    let pairTestLocal2 = pairList.pairs.map { Int32($0.testLocal2) }
    let pairTrialLocal1 = pairList.pairs.map { Int32($0.trialLocal1) }
    let pairTrialLocal2 = pairList.pairs.map { Int32($0.trialLocal2) }

    var ruleTestX: [Float] = []
    var ruleTestY: [Float] = []
    var ruleTrialX: [Float] = []
    var ruleTrialY: [Float] = []
    var ruleWeights: [Float] = []
    var ruleOffsets: [Int32] = []
    var ruleCounts: [Int32] = []
    for kind in 1...3 {
        guard let rule = rules[kind] else {
            try fail("missing Duffy rule for kind \(kind)")
        }
        ruleOffsets.append(Int32(ruleWeights.count))
        ruleCounts.append(Int32(rule.weights.count))
        for i in rule.weights.indices {
            ruleTestX.append(rule.testPoints[i].x)
            ruleTestY.append(rule.testPoints[i].y)
            ruleTrialX.append(rule.trialPoints[i].x)
            ruleTrialY.append(rule.trialPoints[i].y)
            ruleWeights.append(rule.weights[i])
        }
    }

    var params = MetalKernelParams(
        nDof: Int32(geom.p1DofCount),
        nTriangles: Int32(geom.nTriangles),
        maxInc: 0,
        symmetryPlane: geom.symmetryPlaneCode,
        k: k,
        kImag: kImag,
        hasRobin: 0
    )
    var pairCountI32 = Int32(pairCount)
    let slpCount = pairCount * 3
    let dlpCount = pairCount * 9

    let slpRe = try makeOutputBuffer(device, count: slpCount, label: "duffy_slp_re")
    let slpIm = try makeOutputBuffer(device, count: slpCount, label: "duffy_slp_im")
    let dlpRe = try makeOutputBuffer(device, count: dlpCount, label: "duffy_dlp_re")
    let dlpIm = try makeOutputBuffer(device, count: dlpCount, label: "duffy_dlp_im")
    let px = try makeBuffer(device, geom.px, label: "duffy_px")
    let py = try makeBuffer(device, geom.py, label: "duffy_py")
    let pz = try makeBuffer(device, geom.pz, label: "duffy_pz")
    let triangles = try makeBuffer(device, geom.triangles, label: "duffy_triangles")
    let normals = try makeBuffer(device, geom.normals, label: "duffy_normals")
    let areas = try makeBuffer(device, geom.areas, label: "duffy_areas")
    let pairTestBuffer = try makeBuffer(device, pairTest, label: "duffy_pair_test")
    let pairTrialBuffer = try makeBuffer(device, pairTrial, label: "duffy_pair_trial")
    let pairKindBuffer = try makeBuffer(device, pairKind, label: "duffy_pair_kind")
    let pairTestImageMaskBuffer = try makeBuffer(device, pairTestImageMask, label: "duffy_pair_test_image_mask")
    let pairTrialImageMaskBuffer = try makeBuffer(device, pairTrialImageMask, label: "duffy_pair_trial_image_mask")
    let pairTestLocal1Buffer = try makeBuffer(device, pairTestLocal1, label: "duffy_pair_test_local1")
    let pairTestLocal2Buffer = try makeBuffer(device, pairTestLocal2, label: "duffy_pair_test_local2")
    let pairTrialLocal1Buffer = try makeBuffer(device, pairTrialLocal1, label: "duffy_pair_trial_local1")
    let pairTrialLocal2Buffer = try makeBuffer(device, pairTrialLocal2, label: "duffy_pair_trial_local2")
    let ruleTestXBuffer = try makeBuffer(device, ruleTestX, label: "duffy_rule_test_x")
    let ruleTestYBuffer = try makeBuffer(device, ruleTestY, label: "duffy_rule_test_y")
    let ruleTrialXBuffer = try makeBuffer(device, ruleTrialX, label: "duffy_rule_trial_x")
    let ruleTrialYBuffer = try makeBuffer(device, ruleTrialY, label: "duffy_rule_trial_y")
    let ruleWeightsBuffer = try makeBuffer(device, ruleWeights, label: "duffy_rule_weights")
    let ruleOffsetsBuffer = try makeBuffer(device, ruleOffsets, label: "duffy_rule_offsets")
    let ruleCountsBuffer = try makeBuffer(device, ruleCounts, label: "duffy_rule_counts")

    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        try fail("failed to create Metal command buffer")
    }
    commandBuffer.label = "hornlab Duffy delta blocks"
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        try fail("failed to create Metal Duffy encoder")
    }
    encoder.label = "Duffy delta blocks"
    encoder.setBuffer(slpRe, offset: 0, index: 0)
    encoder.setBuffer(slpIm, offset: 0, index: 1)
    encoder.setBuffer(dlpRe, offset: 0, index: 2)
    encoder.setBuffer(dlpIm, offset: 0, index: 3)
    encoder.setBuffer(px, offset: 0, index: 4)
    encoder.setBuffer(py, offset: 0, index: 5)
    encoder.setBuffer(pz, offset: 0, index: 6)
    encoder.setBuffer(triangles, offset: 0, index: 7)
    encoder.setBuffer(normals, offset: 0, index: 8)
    encoder.setBuffer(areas, offset: 0, index: 9)
    encoder.setBuffer(pairTestBuffer, offset: 0, index: 10)
    encoder.setBuffer(pairTrialBuffer, offset: 0, index: 11)
    encoder.setBuffer(pairKindBuffer, offset: 0, index: 12)
    encoder.setBuffer(pairTestImageMaskBuffer, offset: 0, index: 13)
    encoder.setBuffer(pairTrialImageMaskBuffer, offset: 0, index: 14)
    encoder.setBuffer(pairTestLocal1Buffer, offset: 0, index: 15)
    encoder.setBuffer(pairTestLocal2Buffer, offset: 0, index: 16)
    encoder.setBuffer(pairTrialLocal1Buffer, offset: 0, index: 17)
    encoder.setBuffer(pairTrialLocal2Buffer, offset: 0, index: 18)
    encoder.setBuffer(ruleTestXBuffer, offset: 0, index: 19)
    encoder.setBuffer(ruleTestYBuffer, offset: 0, index: 20)
    encoder.setBuffer(ruleTrialXBuffer, offset: 0, index: 21)
    encoder.setBuffer(ruleTrialYBuffer, offset: 0, index: 22)
    encoder.setBuffer(ruleWeightsBuffer, offset: 0, index: 23)
    encoder.setBuffer(ruleOffsetsBuffer, offset: 0, index: 24)
    encoder.setBuffer(ruleCountsBuffer, offset: 0, index: 25)
    encoder.setBytes(&params, length: MemoryLayout<MetalKernelParams>.stride, index: 26)
    encoder.setBytes(&pairCountI32, length: MemoryLayout<Int32>.stride, index: 27)
    let dispatch = try dispatch1D(
        encoder: encoder,
        pipeline: pipeline,
        count: pairCount,
        kernel: "duffy"
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        try fail("Metal Duffy correction failed: \(error)")
    }

    var dispatchReport = dispatch
    dispatchReport["kernel"] = "duffy_delta_blocks"
    dispatchReport["pairs"] = pairCount
    dispatchReport["slp_values"] = slpCount
    dispatchReport["dlp_values"] = dlpCount

    return MetalDuffyBlockOutput(
        slpRe: readFloatBuffer(slpRe, count: slpCount),
        slpIm: readFloatBuffer(slpIm, count: slpCount),
        dlpRe: readFloatBuffer(dlpRe, count: dlpCount),
        dlpIm: readFloatBuffer(dlpIm, count: dlpCount),
        dispatch: dispatchReport
    )
}

let metalThreadsPerGroupEnv = "HORNLAB_METAL_BEM_NATIVE_THREADS_PER_GROUP"
let metalMatrixThreadsPerGroupEnv = "HORNLAB_METAL_BEM_NATIVE_MATRIX_THREADS_PER_GROUP"
let metalRhsThreadsPerGroupEnv = "HORNLAB_METAL_BEM_NATIVE_RHS_THREADS_PER_GROUP"
let metalDuffyThreadsPerGroupEnv = "HORNLAB_METAL_BEM_NATIVE_DUFFY_THREADS_PER_GROUP"
let metalFieldThreadsPerGroupEnv = "HORNLAB_METAL_BEM_NATIVE_FIELD_THREADS_PER_GROUP"
let metalRegularAssemblyImplementationEnv = "HORNLAB_METAL_BEM_NATIVE_REGULAR_ASSEMBLY_IMPL"
let metalDenseSolveImplementationEnv = "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_IMPL"
// Mixed-precision iterative refinement passes after the float32 LU solve;
// 0 (default) disables refinement.
let metalDenseSolveRefineEnv = "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_REFINE"
let metalSolveConcurrencyEnv = "HORNLAB_METAL_BEM_NATIVE_SOLVE_CONCURRENCY"
// Dense factor/solve precision: "float32" (default, Complex32 LU) or "float64"
// (complex128 zgesv, result narrowed back to f32). Mixed precision.
let metalDenseSolveDtypeEnv = "HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_DTYPE"
let nearQuadratureEnv = "HORNLAB_METAL_BEM_NATIVE_NEAR_QUADRATURE"
let defaultMetalThreadsPerThreadgroup = 64

func parseMetalThreadsPerGroupEnv(_ envName: String) throws -> Int? {
    guard let raw = ProcessInfo.processInfo.environment[envName],
          !raw.isEmpty else {
        return nil
    }
    guard let value = Int(raw), value > 0 else {
        try fail("\(envName) must be a positive integer")
    }
    return value
}

func metalThreadsPerGroupEnv(for kernel: String) -> String? {
    switch kernel {
    case "matrix":
        return metalMatrixThreadsPerGroupEnv
    case "rhs":
        return metalRhsThreadsPerGroupEnv
    case "duffy":
        return metalDuffyThreadsPerGroupEnv
    case "field":
        return metalFieldThreadsPerGroupEnv
    default:
        return nil
    }
}

func requestedMetalThreadsPerGroup(kernel: String) throws -> (value: Int?, envName: String?) {
    if let specificEnv = metalThreadsPerGroupEnv(for: kernel),
       let specific = try parseMetalThreadsPerGroupEnv(specificEnv) {
        return (specific, specificEnv)
    }
    if let global = try parseMetalThreadsPerGroupEnv(metalThreadsPerGroupEnv) {
        return (global, metalThreadsPerGroupEnv)
    }
    return (nil, nil)
}

func dispatchConfig(for pipeline: MTLComputePipelineState, kernel: String) throws -> [String: Any] {
    let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
    let request = try requestedMetalThreadsPerGroup(kernel: kernel)
    let requested = request.value
    let defaultThreads = min(defaultMetalThreadsPerThreadgroup, maxThreads)
    let actual = min(requested ?? defaultThreads, maxThreads)
    return [
        "kernel": kernel,
        "env": request.envName ?? NSNull(),
        "global_env": metalThreadsPerGroupEnv,
        "specific_env": metalThreadsPerGroupEnv(for: kernel) ?? NSNull(),
        "default_threads_per_threadgroup": defaultThreads,
        "requested_threads_per_threadgroup": requested.map { $0 as Any } ?? NSNull(),
        "threads_per_threadgroup": actual,
        "max_total_threads_per_threadgroup": maxThreads,
        "thread_execution_width": pipeline.threadExecutionWidth,
        "capped_to_pipeline_max": requested.map { $0 > maxThreads } ?? false,
    ]
}

@discardableResult
func dispatch1D(
    encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState,
    count: Int,
    kernel: String
) throws -> [String: Any] {
    let config = try dispatchConfig(for: pipeline, kernel: kernel)
    let threads = config["threads_per_threadgroup"] as! Int
    encoder.setComputePipelineState(pipeline)
    encoder.dispatchThreads(
        MTLSize(width: count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
    )
    return config
}

func requestedRegularAssemblyImplementation() throws -> String {
    // pair_atomic computes each triangle pair's quadrature once and scatters
    // 3x3 blocks with atomic float adds (~2x faster assembly than entrywise
    // on production meshes, parity-verified). Atomic accumulation order makes
    // it nondeterministic at float32 rounding level run to run; select
    // entrywise when bit-reproducible assembly matters more than speed.
    let raw = ProcessInfo.processInfo.environment[
        metalRegularAssemblyImplementationEnv
    ] ?? "pair_atomic"
    if raw == "entrywise" || raw == "block_staged" || raw == "pair_atomic" {
        return raw
    }
    try fail(
        "\(metalRegularAssemblyImplementationEnv) must be 'entrywise', "
            + "'block_staged', or 'pair_atomic'"
    )
}

func requestedSolveConcurrency() throws -> Int {
    // Default 6: measured on a 10-core/64GB M-series machine, solve-bound
    // batches (n_dof ~4500) keep improving up to 6-8 concurrent cgesv calls
    // (60.9s serial -> 17.1s at 6 -> 16.0s at 8) while assembly-bound batches
    // (n_dof ~1000) are flat for 2-8; 6 leaves headroom for the consumer
    // thread and GPU scheduling. Memory in flight scales with the value:
    // roughly (concurrency + 3) dense systems resident at once.
    let raw = ProcessInfo.processInfo.environment[
        metalSolveConcurrencyEnv
    ] ?? "6"
    guard let value = Int(raw), (1...8).contains(value) else {
        try fail("\(metalSolveConcurrencyEnv) must be an integer in 1...8")
    }
    return value
}

func requestedDenseSolveImplementation() throws -> String {
    let raw = ProcessInfo.processInfo.environment[
        metalDenseSolveImplementationEnv
    ] ?? "cgesv"
    if raw == "cgesv" || raw == "cgetrf_cgetrs" {
        return raw
    }
    try fail("\(metalDenseSolveImplementationEnv) must be 'cgesv' or 'cgetrf_cgetrs'")
}

func requestedDenseSolveDtype() throws -> String {
    let raw = ProcessInfo.processInfo.environment[
        metalDenseSolveDtypeEnv
    ] ?? "float32"
    if raw == "float32" || raw == "float64" {
        return raw
    }
    try fail("\(metalDenseSolveDtypeEnv) must be 'float32' or 'float64'")
}

func requestedDenseSolveRefineIterations() throws -> Int {
    guard let raw = ProcessInfo.processInfo.environment[
        metalDenseSolveRefineEnv
    ], !raw.isEmpty else {
        return 0
    }
    guard let value = Int(raw), (0...10).contains(value) else {
        try fail("\(metalDenseSolveRefineEnv) must be an integer in 0...10")
    }
    return value
}

func requestedNearQuadratureConfig() throws -> NearQuadratureConfig? {
    guard let raw = ProcessInfo.processInfo.environment[nearQuadratureEnv],
          !raw.isEmpty,
          raw != "0" else {
        return nil
    }

    func parseLevel(_ value: String) -> Int? {
        guard let level = Int(value), (1...2).contains(level) else {
            return nil
        }
        return level
    }

    if let level = parseLevel(raw) {
        return NearQuadratureConfig(level: level, threshold: 1.5)
    }

    let parts = raw.split(separator: ":", omittingEmptySubsequences: false)
    if parts.count == 2,
       let level = parseLevel(String(parts[0])),
       let threshold = Double(String(parts[1])),
       threshold.isFinite,
       threshold > 0.0 {
        return NearQuadratureConfig(level: level, threshold: threshold)
    }

    try fail(
        "\(nearQuadratureEnv) must be unset, '0', '1', '2', "
            + "or '<level>:<positive threshold>' with level 1...2"
    )
}

func assembleRegularMetalSelected(
    geom: Geometry,
    neumann: [Complex32],
    k: Float,
    kImag: Float = 0.0,
    robinBetas: [Complex32]? = nil,
    residentContext: ResidentMetalContext?
) throws -> MetalAssemblyOutput {
    let implementation = try requestedRegularAssemblyImplementation()
    if implementation == "block_staged" {
        if let residentContext {
            return try residentContext.assembleRegularBlockStagedMetal(
                neumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas
            )
        }
        let context = try ResidentMetalContext(geom: geom)
        return try context.assembleRegularBlockStagedMetal(
            neumann: neumann,
            k: k,
            kImag: kImag,
            robinBetas: robinBetas
        )
    }
    // entrywise and pair_atomic are both encoded by the resident context.
    if let residentContext {
        return try residentContext.assembleRegularMetal(
            neumann: neumann,
            k: k,
            kImag: kImag,
            robinBetas: robinBetas
        )
    }
    if implementation == "pair_atomic" {
        let context = try ResidentMetalContext(geom: geom)
        return try context.assembleRegularMetal(
            neumann: neumann,
            k: k,
            kImag: kImag,
            robinBetas: robinBetas
        )
    }
    return try assembleRegularMetal(
        geom: geom,
        neumann: neumann,
        k: k,
        kImag: kImag,
        robinBetas: robinBetas
    )
}

func regularMetalImplementationName(_ output: MetalAssemblyOutput) -> String {
    let implementation = output.dispatch["regular_assembly_implementation"] as? String
    if implementation == "block_staged" {
        return "swift_native_metal_block_staged_regular_quadrature"
    }
    if implementation == "pair_atomic" {
        return "swift_native_metal_pair_atomic_regular_quadrature"
    }
    return "swift_native_metal_regular_quadrature"
}

func correctedMetalImplementationName(_ output: MetalAssemblyOutput, stats: DuffyCorrectionStats) -> String {
    let implementation = output.dispatch["regular_assembly_implementation"] as? String
    let regularPrefix: String
    if implementation == "block_staged" {
        regularPrefix = "swift_native_metal_block_staged_regular"
    } else if implementation == "pair_atomic" {
        regularPrefix = "swift_native_metal_pair_atomic_regular"
    } else {
        regularPrefix = "swift_native_metal_regular"
    }
    if stats.implementation == "metal_duffy_blocks_cpu_reduction" {
        return "\(regularPrefix)_plus_metal_duffy_blocks"
    }
    return "\(regularPrefix)_plus_cpu_duffy"
}

struct ResidentMetalPipelines {
    let library: MTLLibrary
    let matrixPipeline: MTLComputePipelineState
    let pairAtomicPipeline: MTLComputePipelineState
    let rhsPipeline: MTLComputePipelineState
    let pairBlockPipeline: MTLComputePipelineState
    let fieldPipeline: MTLComputePipelineState
    let duffyPipeline: MTLComputePipelineState
}

final class ResidentMetalPipelineBox: @unchecked Sendable {
    var result: Result<ResidentMetalPipelines, Error>?
}

final class ResidentMetalContext {
    let geom: Geometry
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    let matrixPipeline: MTLComputePipelineState
    let pairAtomicPipeline: MTLComputePipelineState
    let rhsPipeline: MTLComputePipelineState
    let pairBlockPipeline: MTLComputePipelineState
    let fieldPipeline: MTLComputePipelineState
    let duffyPipeline: MTLComputePipelineState
    let incidence: P1Incidence
    let pairList: DuffyPairList
    let duffyReductionPlan: DuffyReductionPlan
    let duffyReductionPlanBuildSeconds: Double
    let rules: [Int: DuffyRule]
    let px: MTLBuffer
    let py: MTLBuffer
    let pz: MTLBuffer
    let triangles: MTLBuffer
    let p1Local2Global: MTLBuffer
    let normals: MTLBuffer
    let areas: MTLBuffer
    let incTri: MTLBuffer
    let incLoc: MTLBuffer
    let counts: MTLBuffer
    let aRe: MTLBuffer
    let aIm: MTLBuffer
    let rhsRe: MTLBuffer
    let rhsIm: MTLBuffer
    let pairTestBuffer: MTLBuffer?
    let pairTrialBuffer: MTLBuffer?
    let pairKindBuffer: MTLBuffer?
    let pairTestImageMaskBuffer: MTLBuffer?
    let pairTrialImageMaskBuffer: MTLBuffer?
    let pairTestLocal1Buffer: MTLBuffer?
    let pairTestLocal2Buffer: MTLBuffer?
    let pairTrialLocal1Buffer: MTLBuffer?
    let pairTrialLocal2Buffer: MTLBuffer?
    let ruleTestXBuffer: MTLBuffer
    let ruleTestYBuffer: MTLBuffer
    let ruleTrialXBuffer: MTLBuffer
    let ruleTrialYBuffer: MTLBuffer
    let ruleWeightsBuffer: MTLBuffer
    let ruleOffsetsBuffer: MTLBuffer
    let ruleCountsBuffer: MTLBuffer
    let duffySlpRe: MTLBuffer?
    let duffySlpIm: MTLBuffer?
    let duffyDlpRe: MTLBuffer?
    let duffyDlpIm: MTLBuffer?
    var fieldOutRe: MTLBuffer?
    var fieldOutIm: MTLBuffer?
    var fieldOutCount = 0
    private var alternateOutputSlot: AssemblyOutputSlot?

    init(geom: Geometry) throws {
        self.geom = geom
        let pipelineBox = ResidentMetalPipelineBox()
        let pipelineSemaphore = DispatchSemaphore(value: 0)
        let symmetryPlaneCode = geom.symmetryPlaneCode
        Thread.detachNewThread {
            do {
                let device = try MetalWarmup.shared.device()
                let library = try assemblyLibrary(
                    device: device,
                    symmetryPlaneCode: symmetryPlaneCode
                )
                guard let matrixFunction = library.makeFunction(name: "assemble_matrix_regular"),
                      let pairAtomicFunction = library.makeFunction(name: "assemble_matrix_pair_atomic"),
                      let rhsFunction = library.makeFunction(name: "assemble_rhs_source_regular"),
                      let pairBlockFunction = library.makeFunction(name: "assemble_pair_blocks_regular"),
                      let fieldFunction = library.makeFunction(name: "evaluate_field_regular"),
                      let duffyFunction = library.makeFunction(name: "duffy_delta_blocks") else {
                    try fail("failed to load resident Metal kernels")
                }
                pipelineBox.result = .success(ResidentMetalPipelines(
                    library: library,
                    matrixPipeline: try device.makeComputePipelineState(function: matrixFunction),
                    pairAtomicPipeline: try device.makeComputePipelineState(function: pairAtomicFunction),
                    rhsPipeline: try device.makeComputePipelineState(function: rhsFunction),
                    pairBlockPipeline: try device.makeComputePipelineState(function: pairBlockFunction),
                    fieldPipeline: try device.makeComputePipelineState(function: fieldFunction),
                    duffyPipeline: try device.makeComputePipelineState(function: duffyFunction)
                ))
            } catch {
                pipelineBox.result = .failure(error)
            }
            pipelineSemaphore.signal()
        }
        let incidence = try buildP1Incidence(geom)
        let pairList = try buildDuffyPairList(geom)
        let reductionPlanStart = CFAbsoluteTimeGetCurrent()
        let duffyReductionPlan = buildDuffyReductionPlan(geom: geom, pairList: pairList)
        let duffyReductionPlanBuildSeconds = CFAbsoluteTimeGetCurrent() - reductionPlanStart
        let rules = [
            1: try duffyRule(kind: 1),
            2: try duffyRule(kind: 2),
            3: try duffyRule(kind: 3),
        ]
        var ruleTestX: [Float] = []
        var ruleTestY: [Float] = []
        var ruleTrialX: [Float] = []
        var ruleTrialY: [Float] = []
        var ruleWeights: [Float] = []
        var ruleOffsets: [Int32] = []
        var ruleCounts: [Int32] = []
        for kind in 1...3 {
            guard let rule = rules[kind] else {
                try fail("missing Duffy rule for kind \(kind)")
            }
            ruleOffsets.append(Int32(ruleWeights.count))
            ruleCounts.append(Int32(rule.weights.count))
            for idx in rule.weights.indices {
                ruleTestX.append(rule.testPoints[idx].x)
                ruleTestY.append(rule.testPoints[idx].y)
                ruleTrialX.append(rule.trialPoints[idx].x)
                ruleTrialY.append(rule.trialPoints[idx].y)
                ruleWeights.append(rule.weights[idx])
            }
        }

        let device = try MetalWarmup.shared.device()
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            try fail("failed to create Metal command queue")
        }
        self.commandQueue = commandQueue
        self.incidence = incidence
        self.pairList = pairList
        self.duffyReductionPlan = duffyReductionPlan
        self.duffyReductionPlanBuildSeconds = duffyReductionPlanBuildSeconds
        self.rules = rules
        self.px = try makeBuffer(device, geom.px, label: "resident_px")
        self.py = try makeBuffer(device, geom.py, label: "resident_py")
        self.pz = try makeBuffer(device, geom.pz, label: "resident_pz")
        self.triangles = try makeBuffer(device, geom.triangles, label: "resident_triangles")
        self.p1Local2Global = try makeBuffer(device, geom.p1Local2Global, label: "resident_p1_local2global")
        self.normals = try makeBuffer(device, geom.normals, label: "resident_normals")
        self.areas = try makeBuffer(device, geom.areas, label: "resident_areas")
        self.incTri = try makeBuffer(device, incidence.incTri, label: "resident_inc_tri")
        self.incLoc = try makeBuffer(device, incidence.incLoc, label: "resident_inc_loc")
        self.counts = try makeBuffer(device, incidence.counts, label: "resident_counts")
        let n = geom.p1DofCount
        self.aRe = try makeOutputBuffer(device, count: n * n, label: "resident_A_re")
        self.aIm = try makeOutputBuffer(device, count: n * n, label: "resident_A_im")
        self.rhsRe = try makeOutputBuffer(device, count: n, label: "resident_rhs_re")
        self.rhsIm = try makeOutputBuffer(device, count: n, label: "resident_rhs_im")
        self.ruleTestXBuffer = try makeBuffer(device, ruleTestX, label: "resident_duffy_rule_test_x")
        self.ruleTestYBuffer = try makeBuffer(device, ruleTestY, label: "resident_duffy_rule_test_y")
        self.ruleTrialXBuffer = try makeBuffer(device, ruleTrialX, label: "resident_duffy_rule_trial_x")
        self.ruleTrialYBuffer = try makeBuffer(device, ruleTrialY, label: "resident_duffy_rule_trial_y")
        self.ruleWeightsBuffer = try makeBuffer(device, ruleWeights, label: "resident_duffy_rule_weights")
        self.ruleOffsetsBuffer = try makeBuffer(device, ruleOffsets, label: "resident_duffy_rule_offsets")
        self.ruleCountsBuffer = try makeBuffer(device, ruleCounts, label: "resident_duffy_rule_counts")

        if pairList.pairs.isEmpty {
            self.pairTestBuffer = nil
            self.pairTrialBuffer = nil
            self.pairKindBuffer = nil
            self.pairTestImageMaskBuffer = nil
            self.pairTrialImageMaskBuffer = nil
            self.pairTestLocal1Buffer = nil
            self.pairTestLocal2Buffer = nil
            self.pairTrialLocal1Buffer = nil
            self.pairTrialLocal2Buffer = nil
            self.duffySlpRe = nil
            self.duffySlpIm = nil
            self.duffyDlpRe = nil
            self.duffyDlpIm = nil
        } else {
            let pairCount = pairList.pairs.count
            self.pairTestBuffer = try makeBuffer(device, pairList.pairs.map { Int32($0.test) }, label: "resident_duffy_pair_test")
            self.pairTrialBuffer = try makeBuffer(device, pairList.pairs.map { Int32($0.trial) }, label: "resident_duffy_pair_trial")
            self.pairKindBuffer = try makeBuffer(device, pairList.pairs.map { Int32($0.kind) }, label: "resident_duffy_pair_kind")
            self.pairTestImageMaskBuffer = try makeBuffer(device, pairList.pairs.map { Int32($0.testImageMask) }, label: "resident_duffy_pair_test_image_mask")
            self.pairTrialImageMaskBuffer = try makeBuffer(device, pairList.pairs.map { Int32($0.trialImageMask) }, label: "resident_duffy_pair_trial_image_mask")
            self.pairTestLocal1Buffer = try makeBuffer(device, pairList.pairs.map { Int32($0.testLocal1) }, label: "resident_duffy_pair_test_local1")
            self.pairTestLocal2Buffer = try makeBuffer(device, pairList.pairs.map { Int32($0.testLocal2) }, label: "resident_duffy_pair_test_local2")
            self.pairTrialLocal1Buffer = try makeBuffer(device, pairList.pairs.map { Int32($0.trialLocal1) }, label: "resident_duffy_pair_trial_local1")
            self.pairTrialLocal2Buffer = try makeBuffer(device, pairList.pairs.map { Int32($0.trialLocal2) }, label: "resident_duffy_pair_trial_local2")
            self.duffySlpRe = try makeOutputBuffer(device, count: pairCount * 3, label: "resident_duffy_slp_re")
            self.duffySlpIm = try makeOutputBuffer(device, count: pairCount * 3, label: "resident_duffy_slp_im")
            self.duffyDlpRe = try makeOutputBuffer(device, count: pairCount * 9, label: "resident_duffy_dlp_re")
            self.duffyDlpIm = try makeOutputBuffer(device, count: pairCount * 9, label: "resident_duffy_dlp_im")
        }
        pipelineSemaphore.wait()
        guard let pipelineResult = pipelineBox.result else {
            try fail("failed to load resident Metal kernels")
        }
        let pipelines = try pipelineResult.get()
        self.library = pipelines.library
        self.matrixPipeline = pipelines.matrixPipeline
        self.pairAtomicPipeline = pipelines.pairAtomicPipeline
        self.rhsPipeline = pipelines.rhsPipeline
        self.pairBlockPipeline = pipelines.pairBlockPipeline
        self.fieldPipeline = pipelines.fieldPipeline
        self.duffyPipeline = pipelines.duffyPipeline
    }

    struct AssemblyOutputSlot {
        let aRe: MTLBuffer
        let aIm: MTLBuffer
        let rhsRe: MTLBuffer
        let rhsIm: MTLBuffer
        let duffySlpRe: MTLBuffer?
        let duffySlpIm: MTLBuffer?
        let duffyDlpRe: MTLBuffer?
        let duffyDlpIm: MTLBuffer?
    }

    /// Even slot indices alias the primary resident output buffers; odd
    /// indices use a lazily allocated second set so one case's GPU assembly
    /// can run while the previous case's outputs are still being consumed
    /// on the CPU.
    func outputSlot(_ slotIndex: Int) throws -> AssemblyOutputSlot {
        if slotIndex % 2 == 0 {
            return AssemblyOutputSlot(
                aRe: aRe,
                aIm: aIm,
                rhsRe: rhsRe,
                rhsIm: rhsIm,
                duffySlpRe: duffySlpRe,
                duffySlpIm: duffySlpIm,
                duffyDlpRe: duffyDlpRe,
                duffyDlpIm: duffyDlpIm
            )
        }
        if let alternateOutputSlot {
            return alternateOutputSlot
        }
        let n = geom.p1DofCount
        let pairCount = pairList.pairs.count
        let slot = AssemblyOutputSlot(
            aRe: try makeOutputBuffer(device, count: n * n, label: "resident_A_re_alt"),
            aIm: try makeOutputBuffer(device, count: n * n, label: "resident_A_im_alt"),
            rhsRe: try makeOutputBuffer(device, count: n, label: "resident_rhs_re_alt"),
            rhsIm: try makeOutputBuffer(device, count: n, label: "resident_rhs_im_alt"),
            duffySlpRe: pairCount == 0
                ? nil
                : try makeOutputBuffer(device, count: pairCount * 3, label: "resident_duffy_slp_re_alt"),
            duffySlpIm: pairCount == 0
                ? nil
                : try makeOutputBuffer(device, count: pairCount * 3, label: "resident_duffy_slp_im_alt"),
            duffyDlpRe: pairCount == 0
                ? nil
                : try makeOutputBuffer(device, count: pairCount * 9, label: "resident_duffy_dlp_re_alt"),
            duffyDlpIm: pairCount == 0
                ? nil
                : try makeOutputBuffer(device, count: pairCount * 9, label: "resident_duffy_dlp_im_alt")
        )
        alternateOutputSlot = slot
        return slot
    }

    private func readAssemblyArrays(slot: AssemblyOutputSlot) -> AssemblyArrays {
        let n = geom.p1DofCount
        return AssemblyArrays(
            aRe: readFloatBuffer(slot.aRe, count: n * n),
            aIm: readFloatBuffer(slot.aIm, count: n * n),
            rhsRe: readFloatBuffer(slot.rhsRe, count: n),
            rhsIm: readFloatBuffer(slot.rhsIm, count: n)
        )
    }

    private func encodeRegularAssembly(
        commandBuffer: MTLCommandBuffer,
        slot: AssemblyOutputSlot,
        neumann: [Complex32],
        k: Float,
        kImag: Float = 0.0,
        robinBetas: [Complex32]? = nil
    ) throws -> (implementation: String, matrix: [String: Any], rhs: [String: Any]) {
        let requested = try requestedRegularAssemblyImplementation()
        // block_staged is routed to assembleRegularBlockStagedMetal before
        // this encoder is reached; anything else here means entrywise.
        let implementation = requested == "pair_atomic" ? "pair_atomic" : "entrywise"
        var params = MetalKernelParams(
            nDof: Int32(geom.p1DofCount),
            nTriangles: Int32(geom.nTriangles),
            maxInc: Int32(incidence.maxInc),
            symmetryPlane: geom.symmetryPlaneCode,
            k: k,
            kImag: kImag,
            hasRobin: 0
        )
        let robinBuffers = try makeRobinBetaBuffers(
            device: device,
            robinBetas: robinBetas,
            nTriangles: geom.nTriangles,
            labelPrefix: "resident"
        )
        params.hasRobin = robinBuffers.hasRobin
        let n = geom.p1DofCount
        let matrixCount = n * n
        var sourceTrisArray: [Int32] = []
        var sourceReArray: [Float] = []
        var sourceImArray: [Float] = []
        sourceTrisArray.reserveCapacity(neumann.count)
        sourceReArray.reserveCapacity(neumann.count)
        sourceImArray.reserveCapacity(neumann.count)
        for tri in 0..<neumann.count {
            let value = neumann[tri]
            if value.re != 0.0 || value.im != 0.0 {
                sourceTrisArray.append(Int32(tri))
                sourceReArray.append(value.re)
                sourceImArray.append(value.im)
            }
        }
        if sourceTrisArray.isEmpty {
            sourceTrisArray.append(0)
            sourceReArray.append(0.0)
            sourceImArray.append(0.0)
        }
        var sourceCount = Int32(sourceTrisArray.count)
        let sourceTris = try makeBuffer(device, sourceTrisArray, label: "resident_source_tris")
        let sourceRe = try makeBuffer(device, sourceReArray, label: "resident_source_re")
        let sourceIm = try makeBuffer(device, sourceImArray, label: "resident_source_im")

        let matrixDispatch: [String: Any]
        if implementation == "pair_atomic" {
            // The pair-atomic kernel accumulates into A, so the matrix
            // buffers must start from zero on the GPU timeline.
            guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
                try fail("failed to create Metal blit encoder")
            }
            blitEncoder.label = "pair-atomic matrix zero fill"
            let matrixByteCount = matrixCount * MemoryLayout<Float>.stride
            blitEncoder.fill(buffer: slot.aRe, range: 0..<matrixByteCount, value: 0)
            blitEncoder.fill(buffer: slot.aIm, range: 0..<matrixByteCount, value: 0)
            blitEncoder.endEncoding()

            guard let matrixEncoder = commandBuffer.makeComputeCommandEncoder() else {
                try fail("failed to create Metal matrix encoder")
            }
            matrixEncoder.label = "resident pair-atomic P1/P1 DLP matrix"
            var pairCount = Int32(geom.nTriangles * geom.nTriangles)
            matrixEncoder.setBuffer(slot.aRe, offset: 0, index: 0)
            matrixEncoder.setBuffer(slot.aIm, offset: 0, index: 1)
            matrixEncoder.setBuffer(px, offset: 0, index: 2)
            matrixEncoder.setBuffer(py, offset: 0, index: 3)
            matrixEncoder.setBuffer(pz, offset: 0, index: 4)
            matrixEncoder.setBuffer(triangles, offset: 0, index: 5)
            matrixEncoder.setBuffer(p1Local2Global, offset: 0, index: 6)
            matrixEncoder.setBuffer(normals, offset: 0, index: 7)
            matrixEncoder.setBuffer(areas, offset: 0, index: 8)
            matrixEncoder.setBytes(&params, length: MemoryLayout<MetalKernelParams>.stride, index: 9)
            matrixEncoder.setBytes(&pairCount, length: MemoryLayout<Int32>.stride, index: 10)
            matrixEncoder.setBuffer(robinBuffers.re, offset: 0, index: 11)
            matrixEncoder.setBuffer(robinBuffers.im, offset: 0, index: 12)
            var dispatch = try dispatch1D(
                encoder: matrixEncoder,
                pipeline: pairAtomicPipeline,
                count: Int(pairCount),
                kernel: "matrix"
            )
            matrixEncoder.endEncoding()
            dispatch["triangle_pairs"] = Int(pairCount)
            matrixDispatch = dispatch
        } else {
            guard let matrixEncoder = commandBuffer.makeComputeCommandEncoder() else {
                try fail("failed to create Metal matrix encoder")
            }
            matrixEncoder.label = "resident regular P1/P1 DLP matrix"
            matrixEncoder.setBuffer(slot.aRe, offset: 0, index: 0)
            matrixEncoder.setBuffer(slot.aIm, offset: 0, index: 1)
            matrixEncoder.setBuffer(px, offset: 0, index: 2)
            matrixEncoder.setBuffer(py, offset: 0, index: 3)
            matrixEncoder.setBuffer(pz, offset: 0, index: 4)
            matrixEncoder.setBuffer(triangles, offset: 0, index: 5)
            matrixEncoder.setBuffer(p1Local2Global, offset: 0, index: 6)
            matrixEncoder.setBuffer(normals, offset: 0, index: 7)
            matrixEncoder.setBuffer(areas, offset: 0, index: 8)
            matrixEncoder.setBuffer(incTri, offset: 0, index: 9)
            matrixEncoder.setBuffer(incLoc, offset: 0, index: 10)
            matrixEncoder.setBuffer(counts, offset: 0, index: 11)
            matrixEncoder.setBytes(&params, length: MemoryLayout<MetalKernelParams>.stride, index: 12)
            matrixEncoder.setBuffer(robinBuffers.re, offset: 0, index: 13)
            matrixEncoder.setBuffer(robinBuffers.im, offset: 0, index: 14)
            matrixDispatch = try dispatch1D(
                encoder: matrixEncoder,
                pipeline: matrixPipeline,
                count: matrixCount,
                kernel: "matrix"
            )
            matrixEncoder.endEncoding()
        }

        guard let rhsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            try fail("failed to create Metal RHS encoder")
        }
        rhsEncoder.label = "resident regular DP0 Neumann RHS"
        rhsEncoder.setBuffer(slot.rhsRe, offset: 0, index: 0)
        rhsEncoder.setBuffer(slot.rhsIm, offset: 0, index: 1)
        rhsEncoder.setBuffer(px, offset: 0, index: 2)
        rhsEncoder.setBuffer(py, offset: 0, index: 3)
        rhsEncoder.setBuffer(pz, offset: 0, index: 4)
        rhsEncoder.setBuffer(triangles, offset: 0, index: 5)
        rhsEncoder.setBuffer(areas, offset: 0, index: 6)
        rhsEncoder.setBuffer(incTri, offset: 0, index: 7)
        rhsEncoder.setBuffer(incLoc, offset: 0, index: 8)
        rhsEncoder.setBuffer(counts, offset: 0, index: 9)
        rhsEncoder.setBuffer(sourceTris, offset: 0, index: 10)
        rhsEncoder.setBuffer(sourceRe, offset: 0, index: 11)
        rhsEncoder.setBuffer(sourceIm, offset: 0, index: 12)
        rhsEncoder.setBytes(&params, length: MemoryLayout<MetalKernelParams>.stride, index: 13)
        rhsEncoder.setBytes(&sourceCount, length: MemoryLayout<Int32>.stride, index: 14)
        let rhsDispatch = try dispatch1D(
            encoder: rhsEncoder,
            pipeline: rhsPipeline,
            count: n,
            kernel: "rhs"
        )
        rhsEncoder.endEncoding()
        return (implementation: implementation, matrix: matrixDispatch, rhs: rhsDispatch)
    }

    func assembleRegularMetal(
        neumann: [Complex32],
        k: Float,
        kImag: Float = 0.0,
        robinBetas: [Complex32]? = nil
    ) throws -> MetalAssemblyOutput {
        let slot = try outputSlot(0)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            try fail("failed to create Metal command buffer")
        }
        commandBuffer.label = "hornlab resident regular dense assembly"
        let dispatch = try encodeRegularAssembly(
            commandBuffer: commandBuffer,
            slot: slot,
            neumann: neumann,
            k: k,
            kImag: kImag,
            robinBetas: robinBetas
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            try fail("resident Metal regular assembly failed: \(error)")
        }
        return MetalAssemblyOutput(
            arrays: readAssemblyArrays(slot: slot),
            dispatch: [
                "regular_assembly_implementation": dispatch.implementation,
                "matrix": dispatch.matrix,
                "rhs": dispatch.rhs,
            ]
        )
    }

    func assembleRegularBlockStagedMetal(
        neumann: [Complex32],
        k: Float,
        kImag: Float = 0.0,
        robinBetas: [Complex32]? = nil
    ) throws -> MetalAssemblyOutput {
        let n = geom.p1DofCount
        let nTri = geom.nTriangles
        let pairCount = nTri * nTri
        let dlpCount = pairCount * 9
        let slpCount = pairCount * 3

        let pairDlpRe = try makeOutputBuffer(device, count: dlpCount, label: "block_pair_dlp_re")
        let pairDlpIm = try makeOutputBuffer(device, count: dlpCount, label: "block_pair_dlp_im")
        let pairSlpRe = try makeOutputBuffer(device, count: slpCount, label: "block_pair_slp_re")
        let pairSlpIm = try makeOutputBuffer(device, count: slpCount, label: "block_pair_slp_im")
        var params = MetalKernelParams(
            nDof: Int32(n),
            nTriangles: Int32(nTri),
            maxInc: Int32(incidence.maxInc),
            symmetryPlane: geom.symmetryPlaneCode,
            k: k,
            kImag: kImag,
            hasRobin: 0
        )
        let robinBuffers = try makeRobinBetaBuffers(
            device: device,
            robinBetas: robinBetas,
            nTriangles: nTri,
            labelPrefix: "block"
        )
        params.hasRobin = robinBuffers.hasRobin
        var pairCountI32 = Int32(pairCount)

        let blockStart = CFAbsoluteTimeGetCurrent()
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            try fail("failed to create Metal pair-block command buffer")
        }
        commandBuffer.label = "hornlab resident pair-block regular assembly"
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            try fail("failed to create Metal pair-block encoder")
        }
        encoder.label = "resident regular triangle-pair blocks"
        encoder.setBuffer(pairDlpRe, offset: 0, index: 0)
        encoder.setBuffer(pairDlpIm, offset: 0, index: 1)
        encoder.setBuffer(pairSlpRe, offset: 0, index: 2)
        encoder.setBuffer(pairSlpIm, offset: 0, index: 3)
        encoder.setBuffer(px, offset: 0, index: 4)
        encoder.setBuffer(py, offset: 0, index: 5)
        encoder.setBuffer(pz, offset: 0, index: 6)
        encoder.setBuffer(triangles, offset: 0, index: 7)
        encoder.setBuffer(normals, offset: 0, index: 8)
        encoder.setBuffer(areas, offset: 0, index: 9)
        encoder.setBytes(&params, length: MemoryLayout<MetalKernelParams>.stride, index: 10)
        encoder.setBytes(&pairCountI32, length: MemoryLayout<Int32>.stride, index: 11)
        encoder.setBuffer(robinBuffers.re, offset: 0, index: 12)
        encoder.setBuffer(robinBuffers.im, offset: 0, index: 13)
        let blockDispatch = try dispatch1D(
            encoder: encoder,
            pipeline: pairBlockPipeline,
            count: pairCount,
            kernel: "matrix"
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            try fail("resident Metal pair-block regular assembly failed: \(error)")
        }
        let blockSeconds = CFAbsoluteTimeGetCurrent() - blockStart

        let dlpReValues = readFloatBuffer(pairDlpRe, count: dlpCount)
        let dlpImValues = readFloatBuffer(pairDlpIm, count: dlpCount)
        let slpReValues = readFloatBuffer(pairSlpRe, count: slpCount)
        let slpImValues = readFloatBuffer(pairSlpIm, count: slpCount)

        let reductionStart = CFAbsoluteTimeGetCurrent()
        var aReValues = Array(repeating: Float(0.0), count: n * n)
        var aImValues = Array(repeating: Float(0.0), count: n * n)
        var rhsReValues = Array(repeating: Float(0.0), count: n)
        var rhsImValues = Array(repeating: Float(0.0), count: n)
        for testTri in 0..<nTri {
            for trialTri in 0..<nTri {
                let pairIndex = testTri * nTri + trialTri
                let gTrial = neumann[trialTri]
                for i in 0..<3 {
                    let row = geom.p1Dof(testTri, i)
                    let rowWeight = geom.symmetryRowWeight(row)
                    let slpIndex = pairIndex + i * pairCount
                    let sre = slpReValues[slpIndex]
                    let sim = slpImValues[slpIndex]
                    rhsReValues[row] += (sre * gTrial.re - sim * gTrial.im) * rowWeight
                    rhsImValues[row] += (sre * gTrial.im + sim * gTrial.re) * rowWeight
                    for j in 0..<3 {
                        let col = geom.p1Dof(trialTri, j)
                        let localIndex = i * 3 + j
                        let dlpIndex = pairIndex + localIndex * pairCount
                        let outIndex = row * n + col
                        aReValues[outIndex] += dlpReValues[dlpIndex] * rowWeight
                        aImValues[outIndex] += dlpImValues[dlpIndex] * rowWeight
                    }
                }
            }
            for i in 0..<3 {
                let row = geom.p1Dof(testTri, i)
                let rowWeight = geom.symmetryRowWeight(row)
                for j in 0..<3 {
                    let col = geom.p1Dof(testTri, j)
                    let mass = geom.areas[testTri] * (i == j ? Float(1.0 / 6.0) : Float(1.0 / 12.0))
                    aReValues[row * n + col] -= 0.5 * mass * rowWeight
                }
            }
        }
        let reductionSeconds = CFAbsoluteTimeGetCurrent() - reductionStart

        var dispatchReport = blockDispatch
        dispatchReport["kernel"] = "assemble_pair_blocks_regular"
        dispatchReport["triangle_pairs"] = pairCount
        dispatchReport["dlp_values"] = dlpCount
        dispatchReport["slp_values"] = slpCount
        dispatchReport["block_seconds"] = blockSeconds
        dispatchReport["cpu_reduction_seconds"] = reductionSeconds
        return MetalAssemblyOutput(
            arrays: AssemblyArrays(
                aRe: aReValues,
                aIm: aImValues,
                rhsRe: rhsReValues,
                rhsIm: rhsImValues
            ),
            dispatch: [
                "regular_assembly_implementation": "block_staged",
                "pair_blocks": dispatchReport,
            ]
        )
    }

    func computeDuffyDeltaBlocksMetal(k: Float, kImag: Float = 0.0) throws -> MetalDuffyBlockOutput {
        guard !pairList.pairs.isEmpty else {
            return MetalDuffyBlockOutput(slpRe: [], slpIm: [], dlpRe: [], dlpIm: [], dispatch: ["pairs": 0, "kernel": "duffy_delta_blocks"])
        }
        let slot = try outputSlot(0)
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            try fail("failed to create Metal command buffer")
        }
        commandBuffer.label = "hornlab resident Duffy delta blocks"
        let dispatchReport = try encodeDuffyDeltaBlocks(
            commandBuffer: commandBuffer,
            slot: slot,
            k: k,
            kImag: kImag
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            try fail("resident Metal Duffy correction failed: \(error)")
        }
        return try readDuffyBlocks(slot: slot, dispatchReport: dispatchReport)
    }

    private func encodeDuffyDeltaBlocks(
        commandBuffer: MTLCommandBuffer,
        slot: AssemblyOutputSlot,
        k: Float,
        kImag: Float = 0.0
    ) throws -> [String: Any] {
        guard let pairTestBuffer,
              let pairTrialBuffer,
              let pairKindBuffer,
              let pairTestImageMaskBuffer,
              let pairTrialImageMaskBuffer,
              let pairTestLocal1Buffer,
              let pairTestLocal2Buffer,
              let pairTrialLocal1Buffer,
              let pairTrialLocal2Buffer,
              let duffySlpRe = slot.duffySlpRe,
              let duffySlpIm = slot.duffySlpIm,
              let duffyDlpRe = slot.duffyDlpRe,
              let duffyDlpIm = slot.duffyDlpIm else {
            try fail("resident Duffy buffers are unavailable")
        }
        let pairCount = pairList.pairs.count
        var params = MetalKernelParams(
            nDof: Int32(geom.p1DofCount),
            nTriangles: Int32(geom.nTriangles),
            maxInc: 0,
            symmetryPlane: geom.symmetryPlaneCode,
            k: k,
            kImag: kImag,
            hasRobin: 0
        )
        var pairCountI32 = Int32(pairCount)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            try fail("failed to create Metal Duffy encoder")
        }
        encoder.label = "resident Duffy delta blocks"
        encoder.setBuffer(duffySlpRe, offset: 0, index: 0)
        encoder.setBuffer(duffySlpIm, offset: 0, index: 1)
        encoder.setBuffer(duffyDlpRe, offset: 0, index: 2)
        encoder.setBuffer(duffyDlpIm, offset: 0, index: 3)
        encoder.setBuffer(px, offset: 0, index: 4)
        encoder.setBuffer(py, offset: 0, index: 5)
        encoder.setBuffer(pz, offset: 0, index: 6)
        encoder.setBuffer(triangles, offset: 0, index: 7)
        encoder.setBuffer(normals, offset: 0, index: 8)
        encoder.setBuffer(areas, offset: 0, index: 9)
        encoder.setBuffer(pairTestBuffer, offset: 0, index: 10)
        encoder.setBuffer(pairTrialBuffer, offset: 0, index: 11)
        encoder.setBuffer(pairKindBuffer, offset: 0, index: 12)
        encoder.setBuffer(pairTestImageMaskBuffer, offset: 0, index: 13)
        encoder.setBuffer(pairTrialImageMaskBuffer, offset: 0, index: 14)
        encoder.setBuffer(pairTestLocal1Buffer, offset: 0, index: 15)
        encoder.setBuffer(pairTestLocal2Buffer, offset: 0, index: 16)
        encoder.setBuffer(pairTrialLocal1Buffer, offset: 0, index: 17)
        encoder.setBuffer(pairTrialLocal2Buffer, offset: 0, index: 18)
        encoder.setBuffer(ruleTestXBuffer, offset: 0, index: 19)
        encoder.setBuffer(ruleTestYBuffer, offset: 0, index: 20)
        encoder.setBuffer(ruleTrialXBuffer, offset: 0, index: 21)
        encoder.setBuffer(ruleTrialYBuffer, offset: 0, index: 22)
        encoder.setBuffer(ruleWeightsBuffer, offset: 0, index: 23)
        encoder.setBuffer(ruleOffsetsBuffer, offset: 0, index: 24)
        encoder.setBuffer(ruleCountsBuffer, offset: 0, index: 25)
        encoder.setBytes(&params, length: MemoryLayout<MetalKernelParams>.stride, index: 26)
        encoder.setBytes(&pairCountI32, length: MemoryLayout<Int32>.stride, index: 27)
        let dispatch = try dispatch1D(
            encoder: encoder,
            pipeline: duffyPipeline,
            count: pairCount,
            kernel: "duffy"
        )
        encoder.endEncoding()
        var dispatchReport = dispatch
        dispatchReport["kernel"] = "duffy_delta_blocks"
        dispatchReport["pairs"] = pairCount
        dispatchReport["slp_values"] = pairCount * 3
        dispatchReport["dlp_values"] = pairCount * 9
        return dispatchReport
    }

    private func readDuffyBlocks(
        slot: AssemblyOutputSlot,
        dispatchReport: [String: Any]
    ) throws -> MetalDuffyBlockOutput {
        guard let duffySlpRe = slot.duffySlpRe,
              let duffySlpIm = slot.duffySlpIm,
              let duffyDlpRe = slot.duffyDlpRe,
              let duffyDlpIm = slot.duffyDlpIm else {
            try fail("resident Duffy buffers are unavailable")
        }
        let pairCount = pairList.pairs.count
        let slpCount = pairCount * 3
        let dlpCount = pairCount * 9
        return MetalDuffyBlockOutput(
            slpRe: readFloatBuffer(duffySlpRe, count: slpCount),
            slpIm: readFloatBuffer(duffySlpIm, count: slpCount),
            dlpRe: readFloatBuffer(duffyDlpRe, count: dlpCount),
            dlpIm: readFloatBuffer(duffyDlpIm, count: dlpCount),
            dispatch: dispatchReport
        )
    }

    struct PendingAssembly {
        let caseIndex: Int
        let slot: AssemblyOutputSlot
        let regularCommandBuffer: MTLCommandBuffer
        let duffyCommandBuffer: MTLCommandBuffer?
        let includesDuffyBlocks: Bool
        let implementation: String
        let matrixDispatch: [String: Any]
        let rhsDispatch: [String: Any]
        let duffyDispatchReport: [String: Any]?
    }

    struct FinishedAssembly {
        let regular: MetalAssemblyOutput
        let duffyBlocks: MetalDuffyBlockOutput?
        let regularGpuSeconds: Double
        let duffyGpuSeconds: Double
        let readbackSeconds: Double
    }

    /// Encode and commit one case's regular assembly (and Duffy delta blocks
    /// when requested) without waiting, so the GPU works on case `caseIndex`
    /// while the CPU is still solving earlier cases. Even and odd cases write
    /// to distinct output slots; callers must finish case i before beginning
    /// case i + 2 so a slot is never written while it is still being read.
    func beginAssembly(
        caseIndex: Int,
        neumann: [Complex32],
        k: Float,
        kImag: Float = 0.0,
        robinBetas: [Complex32]? = nil,
        includeDuffyBlocks: Bool
    ) throws -> PendingAssembly {
        let slot = try outputSlot(caseIndex % 2)
        guard let regularCommandBuffer = commandQueue.makeCommandBuffer() else {
            try fail("failed to create Metal command buffer")
        }
        regularCommandBuffer.label = "hornlab resident regular dense assembly case \(caseIndex)"
        let dispatch = try encodeRegularAssembly(
            commandBuffer: regularCommandBuffer,
            slot: slot,
            neumann: neumann,
            k: k,
            kImag: kImag,
            robinBetas: robinBetas
        )
        regularCommandBuffer.commit()

        var duffyCommandBuffer: MTLCommandBuffer? = nil
        var duffyDispatchReport: [String: Any]? = nil
        if includeDuffyBlocks && !pairList.pairs.isEmpty {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                try fail("failed to create Metal command buffer")
            }
            commandBuffer.label = "hornlab resident Duffy delta blocks case \(caseIndex)"
            duffyDispatchReport = try encodeDuffyDeltaBlocks(
                commandBuffer: commandBuffer,
                slot: slot,
                k: k,
                kImag: kImag
            )
            commandBuffer.commit()
            duffyCommandBuffer = commandBuffer
        }
        return PendingAssembly(
            caseIndex: caseIndex,
            slot: slot,
            regularCommandBuffer: regularCommandBuffer,
            duffyCommandBuffer: duffyCommandBuffer,
            includesDuffyBlocks: includeDuffyBlocks,
            implementation: dispatch.implementation,
            matrixDispatch: dispatch.matrix,
            rhsDispatch: dispatch.rhs,
            duffyDispatchReport: duffyDispatchReport
        )
    }

    func finishAssembly(_ pending: PendingAssembly) throws -> FinishedAssembly {
        pending.regularCommandBuffer.waitUntilCompleted()
        if let error = pending.regularCommandBuffer.error {
            try fail("resident Metal regular assembly failed: \(error)")
        }
        var duffyGpuSeconds = 0.0
        if let duffyCommandBuffer = pending.duffyCommandBuffer {
            duffyCommandBuffer.waitUntilCompleted()
            if let error = duffyCommandBuffer.error {
                try fail("resident Metal Duffy correction failed: \(error)")
            }
            duffyGpuSeconds = max(
                0.0,
                duffyCommandBuffer.gpuEndTime - duffyCommandBuffer.gpuStartTime
            )
        }
        let regularGpuSeconds = max(
            0.0,
            pending.regularCommandBuffer.gpuEndTime - pending.regularCommandBuffer.gpuStartTime
        )
        let readbackStart = CFAbsoluteTimeGetCurrent()
        let regular = MetalAssemblyOutput(
            arrays: readAssemblyArrays(slot: pending.slot),
            dispatch: [
                "regular_assembly_implementation": pending.implementation,
                "matrix": pending.matrixDispatch,
                "rhs": pending.rhsDispatch,
            ]
        )
        var duffyBlocks: MetalDuffyBlockOutput? = nil
        if let duffyDispatchReport = pending.duffyDispatchReport {
            duffyBlocks = try readDuffyBlocks(
                slot: pending.slot,
                dispatchReport: duffyDispatchReport
            )
        } else if pending.includesDuffyBlocks {
            duffyBlocks = MetalDuffyBlockOutput(
                slpRe: [],
                slpIm: [],
                dlpRe: [],
                dlpIm: [],
                dispatch: ["pairs": 0, "kernel": "duffy_delta_blocks"]
            )
        }
        return FinishedAssembly(
            regular: regular,
            duffyBlocks: duffyBlocks,
            regularGpuSeconds: regularGpuSeconds,
            duffyGpuSeconds: duffyGpuSeconds,
            readbackSeconds: CFAbsoluteTimeGetCurrent() - readbackStart
        )
    }

    func reduceDuffyDeltaBlocks(
        to arrays: AssemblyArrays,
        neumann: [Complex32],
        blocks: MetalDuffyBlockOutput,
        k: Float,
        kImag: Float = 0.0,
        robinBetas: [Complex32]? = nil
    ) -> (AssemblyArrays, Double) {
        let reductionStart = CFAbsoluteTimeGetCurrent()
        let pairCount = pairList.pairs.count
        var aRe = arrays.aRe
        var aIm = arrays.aIm
        var rhsRe = arrays.rhsRe
        var rhsIm = arrays.rhsIm
        var matrixDeltaRe = Array(repeating: 0.0, count: duffyReductionPlan.matrixIndices.count)
        var matrixDeltaIm = Array(repeating: 0.0, count: duffyReductionPlan.matrixIndices.count)
        let iK = Complex32(re: -kImag, im: k)

        for pairIndex in 0..<pairCount {
            let trialTri = duffyReductionPlan.pairTrialTriangles[pairIndex]
            let gTrial = neumann[trialTri]
            let betaTrial = robinBetas?[trialTri] ?? Complex32.zero
            let robinCoupling = iK * betaTrial
            let hasRobin = betaTrial.re != 0.0 || betaTrial.im != 0.0
            for i in 0..<3 {
                let slpIndex = pairIndex + i * pairCount
                let row = duffyReductionPlan.rhsRows[slpIndex]
                let rowWeight = duffyReductionPlan.rowWeights[slpIndex]
                let slpDelta = Complex32(
                    re: blocks.slpRe[slpIndex],
                    im: blocks.slpIm[slpIndex]
                )
                let rhsDelta = (slpDelta * gTrial) * rowWeight
                rhsRe[row] += rhsDelta.re
                rhsIm[row] += rhsDelta.im

                for j in 0..<3 {
                    let deltaIndex = pairIndex + (i * 3 + j) * pairCount
                    let slot = duffyReductionPlan.dlpSlots[deltaIndex]
                    var delta = Complex32(
                        re: blocks.dlpRe[deltaIndex],
                        im: blocks.dlpIm[deltaIndex]
                    )
                    if hasRobin {
                        delta = delta - ((slpDelta * robinCoupling) * Float(1.0 / 3.0))
                    }
                    delta = delta * rowWeight
                    matrixDeltaRe[slot] += Double(delta.re)
                    matrixDeltaIm[slot] += Double(delta.im)
                }
            }
        }

        for slot in duffyReductionPlan.matrixIndices.indices {
            let idx = duffyReductionPlan.matrixIndices[slot]
            aRe[idx] += Float(matrixDeltaRe[slot])
            aIm[idx] += Float(matrixDeltaIm[slot])
        }

        return (
            AssemblyArrays(aRe: aRe, aIm: aIm, rhsRe: rhsRe, rhsIm: rhsIm),
            CFAbsoluteTimeGetCurrent() - reductionStart
        )
    }

    func evaluateExteriorMetal(
        pressure: [Complex32],
        neumann: [Complex32],
        observationPoints: [(Float, Float, Float)],
        k: Float,
        cachedObservationBuffer: MTLBuffer? = nil,
        cachedObservationCount: Int? = nil
    ) throws -> MetalFieldOutput {
        let observationCount = cachedObservationCount ?? observationPoints.count
        var params = MetalKernelParams(
            nDof: Int32(geom.p1DofCount),
            nTriangles: Int32(geom.nTriangles),
            maxInc: 1,
            symmetryPlane: geom.symmetryPlaneCode,
            k: k,
            kImag: 0.0,
            hasRobin: 0
        )
        var nObs = Int32(observationCount)
        if fieldOutCount != observationCount {
            fieldOutRe = try makeOutputBuffer(device, count: observationCount, label: "resident_field_re")
            fieldOutIm = try makeOutputBuffer(device, count: observationCount, label: "resident_field_im")
            fieldOutCount = observationCount
        }
        guard let outRe = fieldOutRe, let outIm = fieldOutIm else {
            try fail("resident field output buffers are unavailable")
        }
        let obsBuffer: MTLBuffer
        if let cachedObservationBuffer {
            obsBuffer = cachedObservationBuffer
        } else {
            obsBuffer = try makeObservationBuffer(
                observationPoints: observationPoints
            ).buffer
        }
        let pressureRe = try makeBuffer(device, pressure.map { $0.re }, label: "resident_pressure_re")
        let pressureIm = try makeBuffer(device, pressure.map { $0.im }, label: "resident_pressure_im")
        let neumannRe = try makeBuffer(device, neumann.map { $0.re }, label: "resident_neumann_re")
        let neumannIm = try makeBuffer(device, neumann.map { $0.im }, label: "resident_neumann_im")

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            try fail("failed to create Metal command buffer")
        }
        commandBuffer.label = "hornlab resident regular field evaluation"
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            try fail("failed to create Metal field encoder")
        }
        encoder.label = "resident regular exterior field"
        encoder.setBuffer(outRe, offset: 0, index: 0)
        encoder.setBuffer(outIm, offset: 0, index: 1)
        encoder.setBuffer(obsBuffer, offset: 0, index: 2)
        encoder.setBuffer(pressureRe, offset: 0, index: 3)
        encoder.setBuffer(pressureIm, offset: 0, index: 4)
        encoder.setBuffer(neumannRe, offset: 0, index: 5)
        encoder.setBuffer(neumannIm, offset: 0, index: 6)
        encoder.setBuffer(px, offset: 0, index: 7)
        encoder.setBuffer(py, offset: 0, index: 8)
        encoder.setBuffer(pz, offset: 0, index: 9)
        encoder.setBuffer(triangles, offset: 0, index: 10)
        encoder.setBuffer(p1Local2Global, offset: 0, index: 11)
        encoder.setBuffer(normals, offset: 0, index: 12)
        encoder.setBuffer(areas, offset: 0, index: 13)
        encoder.setBytes(&params, length: MemoryLayout<MetalKernelParams>.stride, index: 14)
        encoder.setBytes(&nObs, length: MemoryLayout<Int32>.stride, index: 15)
        let fieldDispatch = try dispatch1D(
            encoder: encoder,
            pipeline: fieldPipeline,
            count: observationCount,
            kernel: "field"
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            try fail("resident Metal field evaluation failed: \(error)")
        }
        let gpuSeconds = max(0.0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
        let readbackStart = CFAbsoluteTimeGetCurrent()
        let re = readFloatBuffer(outRe, count: observationCount)
        let im = readFloatBuffer(outIm, count: observationCount)
        return MetalFieldOutput(
            values: zip(re, im).map { Complex32(re: $0.0, im: $0.1) },
            dispatch: ["field": fieldDispatch],
            gpuSeconds: gpuSeconds + (CFAbsoluteTimeGetCurrent() - readbackStart)
        )
    }

    func makeObservationBuffer(
        observationPoints: [(Float, Float, Float)]
    ) throws -> (buffer: MTLBuffer, count: Int) {
        var obs = Array(repeating: Float(0), count: observationPoints.count * 3)
        for idx in observationPoints.indices {
            let (x, y, z) = observationPoints[idx]
            obs[idx] = x
            obs[observationPoints.count + idx] = y
            obs[2 * observationPoints.count + idx] = z
        }
        return (
            try makeBuffer(device, obs, label: "resident_obs_shared"),
            observationPoints.count
        )
    }
}

func assembleRegularMetal(
    geom: Geometry,
    neumann: [Complex32],
    k: Float,
    kImag: Float = 0.0,
    robinBetas: [Complex32]? = nil
) throws -> MetalAssemblyOutput {
    let device = try MetalWarmup.shared.device()
    guard let commandQueue = device.makeCommandQueue() else {
        try fail("failed to create Metal command queue")
    }
    let library = try assemblyLibrary(device: device, symmetryPlaneCode: geom.symmetryPlaneCode)
    guard let matrixFunction = library.makeFunction(name: "assemble_matrix_regular"),
          let rhsFunction = library.makeFunction(name: "assemble_rhs_source_regular") else {
        try fail("failed to load Metal regular assembly kernels")
    }
    let matrixPipeline = try device.makeComputePipelineState(function: matrixFunction)
    let rhsPipeline = try device.makeComputePipelineState(function: rhsFunction)
    let incidence = try buildP1Incidence(geom)
    var params = MetalKernelParams(
        nDof: Int32(geom.p1DofCount),
        nTriangles: Int32(geom.nTriangles),
        maxInc: Int32(incidence.maxInc),
        symmetryPlane: geom.symmetryPlaneCode,
        k: k,
        kImag: kImag,
        hasRobin: 0
    )
    let robinBuffers = try makeRobinBetaBuffers(
        device: device,
        robinBetas: robinBetas,
        nTriangles: geom.nTriangles,
        labelPrefix: "entrywise"
    )
    params.hasRobin = robinBuffers.hasRobin
    let n = geom.p1DofCount
    let matrixCount = n * n

    let aRe = try makeOutputBuffer(device, count: matrixCount, label: "A_re")
    let aIm = try makeOutputBuffer(device, count: matrixCount, label: "A_im")
    let rhsRe = try makeOutputBuffer(device, count: n, label: "rhs_re")
    let rhsIm = try makeOutputBuffer(device, count: n, label: "rhs_im")
    let px = try makeBuffer(device, geom.px, label: "px")
    let py = try makeBuffer(device, geom.py, label: "py")
    let pz = try makeBuffer(device, geom.pz, label: "pz")
    let triangles = try makeBuffer(device, geom.triangles, label: "triangles")
    let p1Local2Global = try makeBuffer(device, geom.p1Local2Global, label: "p1_local2global")
    let normals = try makeBuffer(device, geom.normals, label: "normals")
    let areas = try makeBuffer(device, geom.areas, label: "areas")
    let incTri = try makeBuffer(device, incidence.incTri, label: "inc_tri")
    let incLoc = try makeBuffer(device, incidence.incLoc, label: "inc_loc")
    let counts = try makeBuffer(device, incidence.counts, label: "counts")
    var sourceTrisArray: [Int32] = []
    var sourceReArray: [Float] = []
    var sourceImArray: [Float] = []
    sourceTrisArray.reserveCapacity(neumann.count)
    sourceReArray.reserveCapacity(neumann.count)
    sourceImArray.reserveCapacity(neumann.count)
    for tri in 0..<neumann.count {
        let value = neumann[tri]
        if value.re != 0.0 || value.im != 0.0 {
            sourceTrisArray.append(Int32(tri))
            sourceReArray.append(value.re)
            sourceImArray.append(value.im)
        }
    }
    if sourceTrisArray.isEmpty {
        sourceTrisArray.append(0)
        sourceReArray.append(0.0)
        sourceImArray.append(0.0)
    }
    var sourceCount = Int32(sourceTrisArray.count)
    let sourceTris = try makeBuffer(device, sourceTrisArray, label: "source_tris")
    let sourceRe = try makeBuffer(device, sourceReArray, label: "source_re")
    let sourceIm = try makeBuffer(device, sourceImArray, label: "source_im")

    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        try fail("failed to create Metal command buffer")
    }
    commandBuffer.label = "hornlab regular dense assembly"

    guard let matrixEncoder = commandBuffer.makeComputeCommandEncoder() else {
        try fail("failed to create Metal matrix encoder")
    }
    matrixEncoder.label = "regular P1/P1 DLP matrix"
    matrixEncoder.setBuffer(aRe, offset: 0, index: 0)
    matrixEncoder.setBuffer(aIm, offset: 0, index: 1)
    matrixEncoder.setBuffer(px, offset: 0, index: 2)
    matrixEncoder.setBuffer(py, offset: 0, index: 3)
    matrixEncoder.setBuffer(pz, offset: 0, index: 4)
    matrixEncoder.setBuffer(triangles, offset: 0, index: 5)
    matrixEncoder.setBuffer(p1Local2Global, offset: 0, index: 6)
    matrixEncoder.setBuffer(normals, offset: 0, index: 7)
    matrixEncoder.setBuffer(areas, offset: 0, index: 8)
    matrixEncoder.setBuffer(incTri, offset: 0, index: 9)
    matrixEncoder.setBuffer(incLoc, offset: 0, index: 10)
    matrixEncoder.setBuffer(counts, offset: 0, index: 11)
    matrixEncoder.setBytes(&params, length: MemoryLayout<MetalKernelParams>.stride, index: 12)
    matrixEncoder.setBuffer(robinBuffers.re, offset: 0, index: 13)
    matrixEncoder.setBuffer(robinBuffers.im, offset: 0, index: 14)
    let matrixDispatch = try dispatch1D(
        encoder: matrixEncoder,
        pipeline: matrixPipeline,
        count: matrixCount,
        kernel: "matrix"
    )
    matrixEncoder.endEncoding()

    guard let rhsEncoder = commandBuffer.makeComputeCommandEncoder() else {
        try fail("failed to create Metal RHS encoder")
    }
    rhsEncoder.label = "regular DP0 Neumann RHS"
    rhsEncoder.setBuffer(rhsRe, offset: 0, index: 0)
    rhsEncoder.setBuffer(rhsIm, offset: 0, index: 1)
    rhsEncoder.setBuffer(px, offset: 0, index: 2)
    rhsEncoder.setBuffer(py, offset: 0, index: 3)
    rhsEncoder.setBuffer(pz, offset: 0, index: 4)
    rhsEncoder.setBuffer(triangles, offset: 0, index: 5)
    rhsEncoder.setBuffer(areas, offset: 0, index: 6)
    rhsEncoder.setBuffer(incTri, offset: 0, index: 7)
    rhsEncoder.setBuffer(incLoc, offset: 0, index: 8)
    rhsEncoder.setBuffer(counts, offset: 0, index: 9)
    rhsEncoder.setBuffer(sourceTris, offset: 0, index: 10)
    rhsEncoder.setBuffer(sourceRe, offset: 0, index: 11)
    rhsEncoder.setBuffer(sourceIm, offset: 0, index: 12)
    rhsEncoder.setBytes(&params, length: MemoryLayout<MetalKernelParams>.stride, index: 13)
    rhsEncoder.setBytes(&sourceCount, length: MemoryLayout<Int32>.stride, index: 14)
    let rhsDispatch = try dispatch1D(
        encoder: rhsEncoder,
        pipeline: rhsPipeline,
        count: n,
        kernel: "rhs"
    )
    rhsEncoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        try fail("Metal regular assembly failed: \(error)")
    }

    return MetalAssemblyOutput(
        arrays: AssemblyArrays(
            aRe: readFloatBuffer(aRe, count: matrixCount),
            aIm: readFloatBuffer(aIm, count: matrixCount),
            rhsRe: readFloatBuffer(rhsRe, count: n),
            rhsIm: readFloatBuffer(rhsIm, count: n)
        ),
        dispatch: [
            "regular_assembly_implementation": "entrywise",
            "matrix": matrixDispatch,
            "rhs": rhsDispatch,
        ]
    )
}

func evaluateExteriorMetal(
    geom: Geometry,
    pressure: [Complex32],
    neumann: [Complex32],
    observationPoints: [(Float, Float, Float)],
    k: Float
) throws -> MetalFieldOutput {
    let device = try MetalWarmup.shared.device()
    guard let commandQueue = device.makeCommandQueue() else {
        try fail("failed to create Metal command queue")
    }
    let library = try assemblyLibrary(device: device, symmetryPlaneCode: geom.symmetryPlaneCode)
    guard let fieldFunction = library.makeFunction(name: "evaluate_field_regular") else {
        try fail("failed to load Metal field kernel")
    }
    let fieldPipeline = try device.makeComputePipelineState(function: fieldFunction)
    var params = MetalKernelParams(
        nDof: Int32(geom.p1DofCount),
        nTriangles: Int32(geom.nTriangles),
        maxInc: 1,
        symmetryPlane: geom.symmetryPlaneCode,
        k: k,
        kImag: 0.0,
        hasRobin: 0
    )
    var nObs = Int32(observationPoints.count)
    var obs = Array(repeating: Float(0), count: observationPoints.count * 3)
    for idx in observationPoints.indices {
        let (x, y, z) = observationPoints[idx]
        obs[idx] = x
        obs[observationPoints.count + idx] = y
        obs[2 * observationPoints.count + idx] = z
    }
    let pressureReValues = pressure.map { $0.re }
    let pressureImValues = pressure.map { $0.im }
    let neumannReValues = neumann.map { $0.re }
    let neumannImValues = neumann.map { $0.im }

    let outRe = try makeOutputBuffer(device, count: observationPoints.count, label: "field_re")
    let outIm = try makeOutputBuffer(device, count: observationPoints.count, label: "field_im")
    let obsBuffer = try makeBuffer(device, obs, label: "obs")
    let pressureRe = try makeBuffer(device, pressureReValues, label: "pressure_re")
    let pressureIm = try makeBuffer(device, pressureImValues, label: "pressure_im")
    let neumannRe = try makeBuffer(device, neumannReValues, label: "neumann_re")
    let neumannIm = try makeBuffer(device, neumannImValues, label: "neumann_im")
    let px = try makeBuffer(device, geom.px, label: "px")
    let py = try makeBuffer(device, geom.py, label: "py")
    let pz = try makeBuffer(device, geom.pz, label: "pz")
    let triangles = try makeBuffer(device, geom.triangles, label: "triangles")
    let p1Local2Global = try makeBuffer(device, geom.p1Local2Global, label: "p1_local2global")
    let normals = try makeBuffer(device, geom.normals, label: "normals")
    let areas = try makeBuffer(device, geom.areas, label: "areas")

    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        try fail("failed to create Metal command buffer")
    }
    commandBuffer.label = "hornlab regular field evaluation"
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        try fail("failed to create Metal field encoder")
    }
    encoder.label = "regular exterior field"
    encoder.setBuffer(outRe, offset: 0, index: 0)
    encoder.setBuffer(outIm, offset: 0, index: 1)
    encoder.setBuffer(obsBuffer, offset: 0, index: 2)
    encoder.setBuffer(pressureRe, offset: 0, index: 3)
    encoder.setBuffer(pressureIm, offset: 0, index: 4)
    encoder.setBuffer(neumannRe, offset: 0, index: 5)
    encoder.setBuffer(neumannIm, offset: 0, index: 6)
    encoder.setBuffer(px, offset: 0, index: 7)
    encoder.setBuffer(py, offset: 0, index: 8)
    encoder.setBuffer(pz, offset: 0, index: 9)
    encoder.setBuffer(triangles, offset: 0, index: 10)
    encoder.setBuffer(p1Local2Global, offset: 0, index: 11)
    encoder.setBuffer(normals, offset: 0, index: 12)
    encoder.setBuffer(areas, offset: 0, index: 13)
    encoder.setBytes(&params, length: MemoryLayout<MetalKernelParams>.stride, index: 14)
    encoder.setBytes(&nObs, length: MemoryLayout<Int32>.stride, index: 15)
    let fieldDispatch = try dispatch1D(
        encoder: encoder,
        pipeline: fieldPipeline,
        count: observationPoints.count,
        kernel: "field"
    )
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        try fail("Metal field evaluation failed: \(error)")
    }

    let re = readFloatBuffer(outRe, count: observationPoints.count)
    let im = readFloatBuffer(outIm, count: observationPoints.count)
    return MetalFieldOutput(
        values: zip(re, im).map { Complex32(re: $0.0, im: $0.1) },
        dispatch: [
            "field": fieldDispatch,
        ]
    )
}

func relativeL2(_ lhsRe: [Float], _ lhsIm: [Float], _ rhsRe: [Float], _ rhsIm: [Float]) -> Double {
    var diff = 0.0
    var ref = 0.0
    for idx in lhsRe.indices {
        let dr = Double(lhsRe[idx] - rhsRe[idx])
        let di = Double(lhsIm[idx] - rhsIm[idx])
        diff += dr * dr + di * di
        let rr = Double(rhsRe[idx])
        let ri = Double(rhsIm[idx])
        ref += rr * rr + ri * ri
    }
    if ref <= 0.0 {
        return sqrt(diff)
    }
    return sqrt(diff / ref)
}

func timedRun<T>(_ body: () throws -> T) rethrows -> (T, Double) {
    let start = CFAbsoluteTimeGetCurrent()
    let value = try body()
    return (value, CFAbsoluteTimeGetCurrent() - start)
}

/// Mixed-precision iterative refinement on float32 LU factors. The residual
/// r = b - A·x is accumulated in float64 against the original row-major
/// float32 operator (which survives the LAPACK calls untouched), the
/// correction is solved through the existing single-precision LU via cgetrs,
/// and the solution is accumulated in float64. Stops early once the
/// infinity-norm relative residual reaches the single-precision floor or
/// stops improving. Corrects LU/rounding error only — float32 assembly and
/// quadrature error are untouched, and this is not an interior-resonance
/// (CHIEF/Burton-Miller) substitute.
func refineDenseSolveSolution(
    aReRowMajor: [Float],
    aImRowMajor: [Float],
    rhsRe: [Float],
    rhsIm: [Float],
    factored: inout [__CLPK_complex],
    pivots: inout [__CLPK_integer],
    solution: inout [__CLPK_complex],
    n: Int,
    maxIterations: Int
) -> (iterations: Int, residualRel: Double) {
    let singlePrecisionFloor = 1.0e-7
    var xRe = [Double](repeating: 0.0, count: n)
    var xIm = [Double](repeating: 0.0, count: n)
    for i in 0..<n {
        xRe[i] = Double(solution[i].r)
        xIm[i] = Double(solution[i].i)
    }
    var bNormInf = 0.0
    for i in 0..<n {
        bNormInf = max(bNormInf, abs(Double(rhsRe[i])), abs(Double(rhsIm[i])))
    }

    var resRe = [Double](repeating: 0.0, count: n)
    var resIm = [Double](repeating: 0.0, count: n)

    func residualRelativeNorm() -> Double {
        xRe.withUnsafeBufferPointer { xReBuf in
            xIm.withUnsafeBufferPointer { xImBuf in
                aReRowMajor.withUnsafeBufferPointer { aReBuf in
                    aImRowMajor.withUnsafeBufferPointer { aImBuf in
                        resRe.withUnsafeMutableBufferPointer { resReBuf in
                            resIm.withUnsafeMutableBufferPointer { resImBuf in
                                DispatchQueue.concurrentPerform(iterations: n) { row in
                                    var accRe = 0.0
                                    var accIm = 0.0
                                    let base = row * n
                                    for col in 0..<n {
                                        let are = Double(aReBuf[base + col])
                                        let aim = Double(aImBuf[base + col])
                                        let xre = xReBuf[col]
                                        let xim = xImBuf[col]
                                        accRe += are * xre - aim * xim
                                        accIm += are * xim + aim * xre
                                    }
                                    resReBuf[row] = Double(rhsRe[row]) - accRe
                                    resImBuf[row] = Double(rhsIm[row]) - accIm
                                }
                            }
                        }
                    }
                }
            }
        }
        var rNorm = 0.0
        for i in 0..<n {
            rNorm = max(rNorm, abs(resRe[i]), abs(resIm[i]))
        }
        return bNormInf > 0.0 ? rNorm / bNormInf : rNorm
    }

    var bestRel = residualRelativeNorm()
    var iterationsApplied = 0
    var nClpk = __CLPK_integer(n)
    var nrhs = __CLPK_integer(1)
    var lda = __CLPK_integer(n)
    var ldb = __CLPK_integer(n)
    var trans = Int8(78) // "N"

    for _ in 0..<maxIterations {
        if bestRel <= singlePrecisionFloor {
            break
        }
        var correction = [__CLPK_complex](
            repeating: __CLPK_complex(r: 0.0, i: 0.0),
            count: n
        )
        for i in 0..<n {
            correction[i] = __CLPK_complex(r: Float(resRe[i]), i: Float(resIm[i]))
        }
        var info = __CLPK_integer(0)
        cgetrs_(&trans, &nClpk, &nrhs, &factored, &lda, &pivots, &correction, &ldb, &info)
        if info != 0 {
            break
        }
        let prevRe = xRe
        let prevIm = xIm
        for i in 0..<n {
            xRe[i] += Double(correction[i].r)
            xIm[i] += Double(correction[i].i)
        }
        let rel = residualRelativeNorm()
        if rel >= bestRel {
            // Diverged or stalled: keep the previous iterate.
            xRe = prevRe
            xIm = prevIm
            break
        }
        bestRel = rel
        iterationsApplied += 1
    }

    for i in 0..<n {
        solution[i] = __CLPK_complex(r: Float(xRe[i]), i: Float(xIm[i]))
    }
    return (iterationsApplied, bestRel)
}

/// Mixed-precision dense solve: float32 row-major operator widened to complex128
/// column-major, factored/solved with Accelerate `zgesv`, then narrowed back to
/// Complex32 so the rest of the pipeline (field eval, surface-pressure
/// reductions, all `[Complex32]` buffers, on-disk f32 outputs) is unchanged.
/// This recovers the 3-4 digits the float32 LU loses near a near-singular
/// system. The float32 iterative refinement (`refineDenseSolveSolution`) is
/// deliberately NOT run here: it corrects only against the float32 operator and
/// would cap accuracy at the single-precision floor while wasting a widen/narrow
/// round-trip.
func solveDenseAccelerateZgesv(
    aReRowMajor: [Float],
    aImRowMajor: [Float],
    rhsRe: [Float],
    rhsIm: [Float],
    n: Int
) throws -> DenseSolveRun {
    if aReRowMajor.count != n * n || aImRowMajor.count != n * n {
        try fail("dense solve matrix size mismatch")
    }
    if rhsRe.count != n || rhsIm.count != n {
        try fail("dense solve RHS size mismatch")
    }
    let start = CFAbsoluteTimeGetCurrent()

    // Widen float32 row-major -> complex128 column-major (LAPACK layout).
    var matrix = Array(
        repeating: __CLPK_doublecomplex(r: 0.0, i: 0.0),
        count: n * n
    )
    for row in 0..<n {
        for col in 0..<n {
            let source = row * n + col
            let dest = col * n + row
            matrix[dest] = __CLPK_doublecomplex(
                r: Double(aReRowMajor[source]),
                i: Double(aImRowMajor[source])
            )
        }
    }

    var rhs = Array(
        repeating: __CLPK_doublecomplex(r: 0.0, i: 0.0),
        count: n
    )
    for i in 0..<n {
        rhs[i] = __CLPK_doublecomplex(r: Double(rhsRe[i]), i: Double(rhsIm[i]))
    }

    var nClpk = __CLPK_integer(n)
    var nrhs = __CLPK_integer(1)
    var lda = __CLPK_integer(n)
    var ldb = __CLPK_integer(n)
    var info = __CLPK_integer(0)
    var pivots = Array(repeating: __CLPK_integer(0), count: n)
    let anorm = matrixOneNormZ(&matrix, n: n)
    zgesv_(&nClpk, &nrhs, &matrix, &lda, &pivots, &rhs, &ldb, &info)

    if info != 0 {
        return DenseSolveRun(
            pressure: [],
            implementation: "accelerate_lapack_zgesv",
            seconds: CFAbsoluteTimeGetCurrent() - start,
            lapackInfo: Int32(info),
            rcond: nil,
            dtype: "float64"
        )
    }
    let rcond = estimateReciprocalConditionZ(factored: &matrix, n: n, anorm: anorm)
    // Narrow the complex128 solution back to Complex32 for the f32 pipeline.
    return DenseSolveRun(
        pressure: rhs.map { Complex32(re: Float($0.r), im: Float($0.i)) },
        implementation: "accelerate_lapack_zgesv",
        seconds: CFAbsoluteTimeGetCurrent() - start,
        lapackInfo: Int32(info),
        rcond: rcond,
        dtype: "float64"
    )
}

/// True matrix infinity norm (max row sum of complex magnitudes) of a row-major
/// (rows x cols) complex operator, the norm the plan uses to auto-scale the CHIEF
/// rows against the boundary block: ||M||_inf = max_i sum_j |M_ij|. Magnitudes use
/// the complex absolute value hypot(re, im), not the max scalar component.
func matrixInfNormRowMajor(re: [Float], im: [Float], rows: Int, cols: Int) -> Float {
    var maxRowSum: Float = 0
    for row in 0..<rows {
        var rowSum: Float = 0
        let base = row * cols
        for col in 0..<cols {
            let idx = base + col
            rowSum += Float(hypot(Double(re[idx]), Double(im[idx])))
        }
        maxRowSum = max(maxRowSum, rowSum)
    }
    return maxRowSum
}

/// Solve the CHIEF-overdetermined system in complex128 by least squares (zgels,
/// QR): stack the n x n boundary operator A (rhs b) on top of scale*C (the m
/// CHIEF rows, rhs scale*d) and minimize ||M*p - r||_2. The scale rescales the
/// collocation CHIEF rows so they are numerically comparable to the Galerkin
/// boundary rows: scale = chief_weight * ||A||_inf / max(||C||_inf, eps).
///
/// The solved pressure (first n elements of the LS solution) is narrowed back to
/// Complex32 for the f32 pipeline; the float64 path piggybacks on Feature 1's
/// complex128 LAPACK machinery (the point of CHIEF is to resolve a near-singular
/// system, so doing the LS in f32 would partially defeat it).
///
/// chief_residual_rel is computed EXPLICITLY as ||scale*(C*p - d)||_2 / ||b||_2
/// (the plan's robust fallback) rather than relying on zgels leaving the residual
/// in the trailing rows of the overwritten RHS, so it does not depend on the
/// exact Accelerate zgels residual-row semantics.
func solveDenseLeastSquaresZgels(
    aReRowMajor: [Float],
    aImRowMajor: [Float],
    rhsRe: [Float],
    rhsIm: [Float],
    cReRowMajor: [Float],
    cImRowMajor: [Float],
    dRe: [Float],
    dIm: [Float],
    weight: Float,
    cNormInf: Float,
    aNormInf: Float,
    n: Int,
    m: Int
) throws -> DenseSolveRun {
    if aReRowMajor.count != n * n || aImRowMajor.count != n * n {
        try fail("CHIEF least-squares matrix size mismatch")
    }
    if rhsRe.count != n || rhsIm.count != n {
        try fail("CHIEF least-squares RHS size mismatch")
    }
    if cReRowMajor.count != m * n || cImRowMajor.count != m * n {
        try fail("CHIEF row block size mismatch")
    }
    if dRe.count != m || dIm.count != m {
        try fail("CHIEF row RHS size mismatch")
    }
    if m < 1 {
        try fail("CHIEF least-squares requires at least one constraint row")
    }
    let start = CFAbsoluteTimeGetCurrent()
    let rows = n + m
    let scale = Double(weight) * (cNormInf > 0 ? Double(aNormInf) / Double(cNormInf) : 1.0)

    // Column-major (rows x n) complex128: A on top, scale*C below.
    var matrix = Array(
        repeating: __CLPK_doublecomplex(r: 0.0, i: 0.0),
        count: rows * n
    )
    for row in 0..<n {
        for col in 0..<n {
            let source = row * n + col
            matrix[col * rows + row] = __CLPK_doublecomplex(
                r: Double(aReRowMajor[source]),
                i: Double(aImRowMajor[source])
            )
        }
    }
    for r in 0..<m {
        for col in 0..<n {
            let source = r * n + col
            matrix[col * rows + (n + r)] = __CLPK_doublecomplex(
                r: Double(cReRowMajor[source]) * scale,
                i: Double(cImRowMajor[source]) * scale
            )
        }
    }

    // RHS length = max(rows, n) = rows; first n = b, next m = scale*d.
    var b = Array(
        repeating: __CLPK_doublecomplex(r: 0.0, i: 0.0),
        count: rows
    )
    for i in 0..<n {
        b[i] = __CLPK_doublecomplex(r: Double(rhsRe[i]), i: Double(rhsIm[i]))
    }
    for r in 0..<m {
        b[n + r] = __CLPK_doublecomplex(
            r: Double(dRe[r]) * scale,
            i: Double(dIm[r]) * scale
        )
    }

    var trans = Int8(78) // "N"
    var mC = __CLPK_integer(rows)
    var nC = __CLPK_integer(n)
    var nrhs = __CLPK_integer(1)
    var lda = __CLPK_integer(rows)
    var ldb = __CLPK_integer(rows)
    var info = __CLPK_integer(0)

    // Workspace query (lwork = -1), then the real solve.
    var lwork = __CLPK_integer(-1)
    var workQuery = [__CLPK_doublecomplex(r: 0.0, i: 0.0)]
    zgels_(&trans, &mC, &nC, &nrhs, &matrix, &lda, &b, &ldb, &workQuery, &lwork, &info)
    if info != 0 {
        return DenseSolveRun(
            pressure: [],
            implementation: "accelerate_lapack_zgels",
            seconds: CFAbsoluteTimeGetCurrent() - start,
            lapackInfo: Int32(info),
            rcond: nil,
            dtype: "float64",
            chiefResidualRel: nil
        )
    }
    let workSize = max(1, Int(workQuery[0].r))
    lwork = __CLPK_integer(workSize)
    var work = Array(
        repeating: __CLPK_doublecomplex(r: 0.0, i: 0.0),
        count: workSize
    )
    zgels_(&trans, &mC, &nC, &nrhs, &matrix, &lda, &b, &ldb, &work, &lwork, &info)
    if info != 0 {
        return DenseSolveRun(
            pressure: [],
            implementation: "accelerate_lapack_zgels",
            seconds: CFAbsoluteTimeGetCurrent() - start,
            lapackInfo: Int32(info),
            rcond: nil,
            dtype: "float64",
            chiefResidualRel: nil
        )
    }

    // Extract the least-squares solution (first n elements of b).
    var solution = Array(
        repeating: __CLPK_doublecomplex(r: 0.0, i: 0.0),
        count: n
    )
    for i in 0..<n {
        solution[i] = b[i]
    }

    // Recompute the CHIEF residual EXPLICITLY: ||scale*(C*p - d)||_2 / ||b||_2,
    // using the original row-major C/d (not the overwritten LS RHS rows).
    var residSq = 0.0
    for r in 0..<m {
        var accRe = 0.0
        var accIm = 0.0
        for col in 0..<n {
            let cre = Double(cReRowMajor[r * n + col])
            let cim = Double(cImRowMajor[r * n + col])
            let pre = solution[col].r
            let pim = solution[col].i
            accRe += cre * pre - cim * pim
            accIm += cre * pim + cim * pre
        }
        let diffRe = scale * (accRe - Double(dRe[r]))
        let diffIm = scale * (accIm - Double(dIm[r]))
        residSq += diffRe * diffRe + diffIm * diffIm
    }
    var refSq = 0.0
    for i in 0..<n {
        refSq += Double(rhsRe[i]) * Double(rhsRe[i]) + Double(rhsIm[i]) * Double(rhsIm[i])
    }
    let chiefResidualRel = refSq > 0 ? (residSq / refSq).squareRoot() : residSq.squareRoot()

    return DenseSolveRun(
        pressure: solution.map { Complex32(re: Float($0.r), im: Float($0.i)) },
        implementation: "accelerate_lapack_zgels",
        seconds: CFAbsoluteTimeGetCurrent() - start,
        lapackInfo: Int32(info),
        rcond: nil,
        dtype: "float64",
        chiefResidualRel: chiefResidualRel
    )
}

/// Dense solve for one case: the plain square LU/zgesv path, OR the
/// CHIEF-overdetermined least-squares (zgels) path when interior CHIEF points
/// are present. The CHIEF rows reuse the field-eval kernels (real-k), carry the
/// identical Robin fold as the boundary rows, and are auto-scaled by
/// ||A||_inf/||C||_inf * chief_weight. File-scope (not a captured closure) so the
/// concurrent solve pool can call it without capturing the batch's mutable
/// per-case arrays; the sequential and concurrent solve sites share it so the
/// CHIEF math is assembled identically in all paths.
func solveCaseDense(
    arrays: AssemblyArrays,
    geom: Geometry,
    chiefPoints: [(Float, Float, Float)]?,
    chiefWeight: Float,
    driverNeumann: [Complex32],
    k: Float,
    kImag: Float,
    robinBetas: [Complex32]?
) throws -> DenseSolveRun {
    guard let chiefPoints else {
        return try solveDenseAccelerate(
            aReRowMajor: arrays.aRe,
            aImRowMajor: arrays.aIm,
            rhsRe: arrays.rhsRe,
            rhsIm: arrays.rhsIm,
            n: geom.p1DofCount
        )
    }
    let chiefRows = assembleChiefRows(
        geom: geom,
        chiefPoints: chiefPoints,
        driverNeumann: driverNeumann,
        k: k,
        kImag: kImag,
        robinBetas: robinBetas
    )
    let n = geom.p1DofCount
    let aNormInf = matrixInfNormRowMajor(
        re: arrays.aRe, im: arrays.aIm, rows: n, cols: n
    )
    return try solveDenseLeastSquaresZgels(
        aReRowMajor: arrays.aRe,
        aImRowMajor: arrays.aIm,
        rhsRe: arrays.rhsRe,
        rhsIm: arrays.rhsIm,
        cReRowMajor: chiefRows.cRe,
        cImRowMajor: chiefRows.cIm,
        dRe: chiefRows.dRe,
        dIm: chiefRows.dIm,
        weight: chiefWeight,
        cNormInf: chiefRows.cNormInf,
        aNormInf: aNormInf,
        n: n,
        m: chiefPoints.count
    )
}

func solveDenseAccelerateCgesv(
    aReRowMajor: [Float],
    aImRowMajor: [Float],
    rhsRe: [Float],
    rhsIm: [Float],
    n: Int
) throws -> DenseSolveRun {
    if aReRowMajor.count != n * n || aImRowMajor.count != n * n {
        try fail("dense solve matrix size mismatch")
    }
    if rhsRe.count != n || rhsIm.count != n {
        try fail("dense solve RHS size mismatch")
    }
    let start = CFAbsoluteTimeGetCurrent()

    var matrix = Array(
        repeating: __CLPK_complex(r: 0.0, i: 0.0),
        count: n * n
    )
    for row in 0..<n {
        for col in 0..<n {
            let source = row * n + col
            let dest = col * n + row
            matrix[dest] = __CLPK_complex(
                r: aReRowMajor[source],
                i: aImRowMajor[source]
            )
        }
    }

    var rhs = Array(
        repeating: __CLPK_complex(r: 0.0, i: 0.0),
        count: n
    )
    for i in 0..<n {
        rhs[i] = __CLPK_complex(r: rhsRe[i], i: rhsIm[i])
    }

    var nClpk = __CLPK_integer(n)
    var nrhs = __CLPK_integer(1)
    var lda = __CLPK_integer(n)
    var ldb = __CLPK_integer(n)
    var info = __CLPK_integer(0)
    var pivots = Array(repeating: __CLPK_integer(0), count: n)
    let anorm = matrixOneNorm(&matrix, n: n)
    cgesv_(&nClpk, &nrhs, &matrix, &lda, &pivots, &rhs, &ldb, &info)

    if info != 0 {
        return DenseSolveRun(
            pressure: [],
            implementation: "accelerate_lapack_cgesv",
            seconds: CFAbsoluteTimeGetCurrent() - start,
            lapackInfo: Int32(info),
            rcond: nil
        )
    }
    let rcond = estimateReciprocalCondition(factored: &matrix, n: n, anorm: anorm)
    var refineIterations: Int? = nil
    var refineResidualRel: Double? = nil
    let refinePasses = try requestedDenseSolveRefineIterations()
    if refinePasses > 0 {
        let outcome = refineDenseSolveSolution(
            aReRowMajor: aReRowMajor,
            aImRowMajor: aImRowMajor,
            rhsRe: rhsRe,
            rhsIm: rhsIm,
            factored: &matrix,
            pivots: &pivots,
            solution: &rhs,
            n: n,
            maxIterations: refinePasses
        )
        refineIterations = outcome.iterations
        refineResidualRel = outcome.residualRel
    }
    return DenseSolveRun(
        pressure: rhs.map { Complex32(re: $0.r, im: $0.i) },
        implementation: "accelerate_lapack_cgesv",
        seconds: CFAbsoluteTimeGetCurrent() - start,
        lapackInfo: Int32(info),
        rcond: rcond,
        refineIterations: refineIterations,
        refineResidualRel: refineResidualRel
    )
}

func solveDenseAccelerateCgetrfCgetrs(
    aReRowMajor: [Float],
    aImRowMajor: [Float],
    rhsRe: [Float],
    rhsIm: [Float],
    n: Int
) throws -> DenseSolveRun {
    if aReRowMajor.count != n * n || aImRowMajor.count != n * n {
        try fail("dense solve matrix size mismatch")
    }
    if rhsRe.count != n || rhsIm.count != n {
        try fail("dense solve RHS size mismatch")
    }
    let start = CFAbsoluteTimeGetCurrent()

    var matrix = Array(
        repeating: __CLPK_complex(r: 0.0, i: 0.0),
        count: n * n
    )
    for row in 0..<n {
        for col in 0..<n {
            let source = row * n + col
            let dest = col * n + row
            matrix[dest] = __CLPK_complex(
                r: aReRowMajor[source],
                i: aImRowMajor[source]
            )
        }
    }

    var rhs = Array(
        repeating: __CLPK_complex(r: 0.0, i: 0.0),
        count: n
    )
    for i in 0..<n {
        rhs[i] = __CLPK_complex(r: rhsRe[i], i: rhsIm[i])
    }

    var mClpk = __CLPK_integer(n)
    var nClpk = __CLPK_integer(n)
    var nrhs = __CLPK_integer(1)
    var lda = __CLPK_integer(n)
    var ldb = __CLPK_integer(n)
    var info = __CLPK_integer(0)
    var pivots = Array(repeating: __CLPK_integer(0), count: n)
    let anorm = matrixOneNorm(&matrix, n: n)
    cgetrf_(&mClpk, &nClpk, &matrix, &lda, &pivots, &info)
    var rcond: Double? = nil
    if info == 0 {
        rcond = estimateReciprocalCondition(factored: &matrix, n: n, anorm: anorm)
        var trans = Int8(78)
        cgetrs_(&trans, &nClpk, &nrhs, &matrix, &lda, &pivots, &rhs, &ldb, &info)
    }

    if info != 0 {
        return DenseSolveRun(
            pressure: [],
            implementation: "accelerate_lapack_cgetrf_cgetrs",
            seconds: CFAbsoluteTimeGetCurrent() - start,
            lapackInfo: Int32(info),
            rcond: rcond
        )
    }
    var refineIterations: Int? = nil
    var refineResidualRel: Double? = nil
    let refinePasses = try requestedDenseSolveRefineIterations()
    if refinePasses > 0 {
        let outcome = refineDenseSolveSolution(
            aReRowMajor: aReRowMajor,
            aImRowMajor: aImRowMajor,
            rhsRe: rhsRe,
            rhsIm: rhsIm,
            factored: &matrix,
            pivots: &pivots,
            solution: &rhs,
            n: n,
            maxIterations: refinePasses
        )
        refineIterations = outcome.iterations
        refineResidualRel = outcome.residualRel
    }
    return DenseSolveRun(
        pressure: rhs.map { Complex32(re: $0.r, im: $0.i) },
        implementation: "accelerate_lapack_cgetrf_cgetrs",
        seconds: CFAbsoluteTimeGetCurrent() - start,
        lapackInfo: Int32(info),
        rcond: rcond,
        refineIterations: refineIterations,
        refineResidualRel: refineResidualRel
    )
}

func solveDenseAccelerate(
    aReRowMajor: [Float],
    aImRowMajor: [Float],
    rhsRe: [Float],
    rhsIm: [Float],
    n: Int
) throws -> DenseSolveRun {
    if try requestedDenseSolveDtype() == "float64" {
        return try solveDenseAccelerateZgesv(
            aReRowMajor: aReRowMajor,
            aImRowMajor: aImRowMajor,
            rhsRe: rhsRe,
            rhsIm: rhsIm,
            n: n
        )
    }
    let implementation = try requestedDenseSolveImplementation()
    if implementation == "cgetrf_cgetrs" {
        return try solveDenseAccelerateCgetrfCgetrs(
            aReRowMajor: aReRowMajor,
            aImRowMajor: aImRowMajor,
            rhsRe: rhsRe,
            rhsIm: rhsIm,
            n: n
        )
    }
    return try solveDenseAccelerateCgesv(
        aReRowMajor: aReRowMajor,
        aImRowMajor: aImRowMajor,
        rhsRe: rhsRe,
        rhsIm: rhsIm,
        n: n
    )
}

func assembleRegular(
    geom: Geometry,
    neumann: [Complex32],
    k: Float,
    kImag: Float = 0.0,
    robinBetas: [Complex32]? = nil,
    residentContext: ResidentMetalContext? = nil
) throws -> AssemblyRun {
    let mode = ProcessInfo.processInfo.environment[
        "HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE"
    ] ?? "optimized"
    if mode == "reference" {
        let (arrays, seconds) = timedRun {
            assembleRegularReference(
                geom: geom,
                neumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas
            )
        }
        if kImag != 0.0 || robinBetas != nil {
            let (corrected, stats) = try applyDuffyCorrectionsCPU(
                to: arrays,
                geom: geom,
                neumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas
            )
            let (nearCorrected, nearStats) = try applyNearFieldCorrectionsIfEnabled(
                to: corrected,
                geom: geom,
                neumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas
            )
            return AssemblyRun(
                arrays: nearCorrected,
                implementation: "swift_native_reference_complex_robin_quadrature_plus_cpu_duffy",
                mode: mode,
                seconds: seconds + stats.seconds + (nearStats?.seconds ?? 0.0),
                parity: nil,
                duffyStats: stats,
                nearStats: nearStats,
                metalDispatch: nil
            )
        }
        return AssemblyRun(
            arrays: arrays,
            implementation: "swift_native_reference_regular_quadrature",
            mode: mode,
            seconds: seconds,
            parity: nil,
            duffyStats: nil,
            nearStats: nil,
            metalDispatch: nil
        )
    }
    if mode == "optimized" {
        let (output, seconds) = try timedRun {
            try assembleRegularMetalSelected(
                geom: geom,
                neumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas,
                residentContext: residentContext
            )
        }
        let (nearCorrected, nearStats) = try applyNearFieldCorrectionsIfEnabled(
            to: output.arrays,
            geom: geom,
            neumann: neumann,
            k: k,
            kImag: kImag,
            robinBetas: robinBetas
        )
        return AssemblyRun(
            arrays: nearCorrected,
            implementation: regularMetalImplementationName(output),
            mode: mode,
            seconds: seconds + (nearStats?.seconds ?? 0.0),
            parity: nil,
            duffyStats: nil,
            nearStats: nearStats,
            metalDispatch: output.dispatch
        )
    }
    if mode == "corrected" {
        let (regular, regularSeconds) = try timedRun {
            try assembleRegularMetalSelected(
                geom: geom,
                neumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas,
                residentContext: residentContext
            )
        }
        let (corrected, stats) = try applyDuffyCorrections(
            to: regular.arrays,
            geom: geom,
            neumann: neumann,
            k: k,
            kImag: kImag,
            robinBetas: robinBetas,
            residentContext: residentContext
        )
        let (nearCorrected, nearStats) = try applyNearFieldCorrectionsIfEnabled(
            to: corrected,
            geom: geom,
            neumann: neumann,
            k: k,
            kImag: kImag,
            robinBetas: robinBetas
        )
        return AssemblyRun(
            arrays: nearCorrected,
            implementation: correctedMetalImplementationName(regular, stats: stats),
            mode: mode,
            seconds: regularSeconds + stats.seconds + (nearStats?.seconds ?? 0.0),
            parity: nil,
            duffyStats: stats,
            nearStats: nearStats,
            metalDispatch: regular.dispatch
        )
    }
    if mode == "parity" {
        let (reference, referenceSeconds) = timedRun {
            assembleRegularReference(
                geom: geom,
                neumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas
            )
        }
        let (optimized, optimizedSeconds) = try timedRun {
            try assembleRegularMetalSelected(
                geom: geom,
                neumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas,
                residentContext: residentContext
            )
        }
        let matrixRel = relativeL2(
            optimized.arrays.aRe,
            optimized.arrays.aIm,
            reference.aRe,
            reference.aIm
        )
        let rhsRel = relativeL2(
            optimized.arrays.rhsRe,
            optimized.arrays.rhsIm,
            reference.rhsRe,
            reference.rhsIm
        )
        if matrixRel > 1.0e-4 || rhsRel > 1.0e-4 {
            try fail("optimized native regular assembly parity failed: matrix_rel=\(matrixRel), rhs_rel=\(rhsRel)")
        }
        return AssemblyRun(
            arrays: optimized.arrays,
            implementation: regularMetalImplementationName(optimized),
            mode: mode,
            seconds: optimizedSeconds,
            parity: [
                "reference_seconds": referenceSeconds,
                "optimized_seconds": optimizedSeconds,
                "matrix_relative_l2": matrixRel,
                "rhs_relative_l2": rhsRel,
                "tolerance": 1.0e-4,
            ],
            duffyStats: nil,
            nearStats: nil,
            metalDispatch: optimized.dispatch
        )
    }
    try fail("unsupported native assembly mode: \(mode)")
}

func evaluateExterior(
    geom: Geometry,
    pressure: [Complex32],
    neumann: [Complex32],
    observationPoints: [(Float, Float, Float)],
    k: Float,
    residentContext: ResidentMetalContext? = nil,
    cachedObservationBuffer: MTLBuffer? = nil,
    cachedObservationCount: Int? = nil
) throws -> FieldRun {
    let mode = ProcessInfo.processInfo.environment[
        "HORNLAB_METAL_BEM_NATIVE_FIELD_MODE"
    ] ?? "reference"
    if mode == "reference" {
        let (values, seconds) = timedRun {
            evaluateExteriorReference(
                geom: geom,
                pressure: pressure,
                neumann: neumann,
                observationPoints: observationPoints,
                k: k
            )
        }
        return FieldRun(
            values: values,
            implementation: "swift_native_reference_regular_field",
            mode: mode,
            seconds: seconds,
            parity: nil,
            metalDispatch: nil
        )
    }
    if mode == "optimized" {
        let (output, seconds) = try timedRun {
            if let residentContext {
                return try residentContext.evaluateExteriorMetal(
                    pressure: pressure,
                    neumann: neumann,
                    observationPoints: observationPoints,
                    k: k,
                    cachedObservationBuffer: cachedObservationBuffer,
                    cachedObservationCount: cachedObservationCount
                )
            }
            return try evaluateExteriorMetal(
                    geom: geom,
                    pressure: pressure,
                    neumann: neumann,
                    observationPoints: observationPoints,
                    k: k
                )
        }
        return FieldRun(
            values: output.values,
            implementation: "swift_native_metal_regular_field",
            mode: mode,
            seconds: output.gpuSeconds ?? seconds,
            parity: nil,
            metalDispatch: output.dispatch
        )
    }
    if mode == "parity" {
        let (reference, referenceSeconds) = timedRun {
            evaluateExteriorReference(
                geom: geom,
                pressure: pressure,
                neumann: neumann,
                observationPoints: observationPoints,
                k: k
            )
        }
        let (optimized, optimizedSeconds) = try timedRun {
            if let residentContext {
                return try residentContext.evaluateExteriorMetal(
                    pressure: pressure,
                    neumann: neumann,
                    observationPoints: observationPoints,
                    k: k
                )
            }
            return try evaluateExteriorMetal(
                    geom: geom,
                    pressure: pressure,
                    neumann: neumann,
                    observationPoints: observationPoints,
                    k: k
                )
        }
        let fieldRel = relativeL2(
            optimized.values.map { $0.re },
            optimized.values.map { $0.im },
            reference.map { $0.re },
            reference.map { $0.im }
        )
        if fieldRel > 1.0e-4 {
            try fail("optimized native field parity failed: field_rel=\(fieldRel)")
        }
        return FieldRun(
            values: optimized.values,
            implementation: "swift_native_metal_regular_field",
            mode: mode,
            seconds: optimizedSeconds,
            parity: [
                "reference_seconds": referenceSeconds,
                "optimized_seconds": optimizedSeconds,
                "field_relative_l2": fieldRel,
                "tolerance": 1.0e-4,
            ],
            metalDispatch: optimized.dispatch
        )
    }
    try fail("unsupported native field mode: \(mode)")
}

func assemblyCorrectionSeconds(_ run: AssemblyRun) -> Double {
    (run.duffyStats?.seconds ?? 0.0) + (run.nearStats?.seconds ?? 0.0)
}

func attachNearQuadratureReport(_ result: inout [String: Any], run: AssemblyRun) {
    if let stats = run.nearStats {
        result["near_quadrature"] = stats.toJSON()
    }
}

func assembleStandardNeumann(
    sessionManifestPath: String,
    payloadPath: String,
    resultPath: String
) throws {
    let geom = try readGeometry(sessionManifestPath)
    let payload = try loadJSON(payloadPath)
    if try requireString(payload, "schema") != schema {
        try fail("unsupported schema")
    }
    if try requireString(payload, "op") != "assemble_standard_neumann" {
        try fail("expected assemble_standard_neumann op")
    }
    let k = Float(try requireDouble(payload, "k_real_f32"))
    let neumann = try readComplexVector(
        root: geom.root,
        descriptors: try requireObject(payload, "neumann_dp0"),
        count: geom.dp0DofCount
    )
    let outputs = try requireObject(payload, "outputs")
    let aReDesc = try requireObject(outputs, "A_real_f32")
    let aImDesc = try requireObject(outputs, "A_imag_f32")
    let rhsReDesc = try requireObject(outputs, "rhs_real_f32")
    let rhsImDesc = try requireObject(outputs, "rhs_imag_f32")
    let run = try assembleRegular(geom: geom, neumann: neumann, k: k)
    try writeF32(try descriptorPath(root: geom.root, descriptor: aReDesc), run.arrays.aRe)
    try writeF32(try descriptorPath(root: geom.root, descriptor: aImDesc), run.arrays.aIm)
    try writeF32(try descriptorPath(root: geom.root, descriptor: rhsReDesc), run.arrays.rhsRe)
    try writeF32(try descriptorPath(root: geom.root, descriptor: rhsImDesc), run.arrays.rhsIm)
    let correctionSeconds = assemblyCorrectionSeconds(run)
    let duffyReport: [String: Any]
    if let stats = run.duffyStats {
        var report = stats.toJSON()
        if geom.symmetryPlane != nil {
            report["scope"] = "real_and_image_matrix_and_rhs_duffy_delta"
            report["image_singular_correction"] = true
        }
        duffyReport = report
    } else {
        let duffyPlan = try buildDuffyPairPlan(geom)
        duffyReport = [
            "implemented": false,
            "scope": "regular_quadrature_only",
            "planned_pairs": duffyPlan.toJSON(),
            "raw_triplets_if_expanded": duffyPlan.total * 9,
        ]
    }
    var result: [String: Any] = [
        "schema": schema,
        "op": "assemble_standard_neumann_result",
        "implementation": run.implementation,
        "assembly_mode": run.mode,
        "assembly_seconds": run.seconds,
        "regular_assembly_seconds": max(0.0, run.seconds - correctionSeconds),
        "symmetry_plane": geom.symmetryPlane.map { $0 as Any } ?? NSNull(),
        "duffy_corrections": duffyReport,
        "session_id": try requireString(payload, "session_id"),
        "frequency_hz": (payload["frequency_hz"] as? NSNumber)?.doubleValue ?? 0,
        "matrix_layout": "row_major_c",
        "matrix_shape": [geom.p1DofCount, geom.p1DofCount],
        "rhs_shape": [geom.p1DofCount],
        "matrix_real_f32": try requireString(aReDesc, "path"),
        "matrix_imag_f32": try requireString(aImDesc, "path"),
        "rhs_real_f32": try requireString(rhsReDesc, "path"),
        "rhs_imag_f32": try requireString(rhsImDesc, "path"),
    ]
    if let parity = run.parity {
        result["reference_parity"] = parity
    }
    if let dispatch = run.metalDispatch {
        result["metal_dispatch"] = dispatch
    }
    attachNearQuadratureReport(&result, run: run)
    try writeJSON(resultPath, result)
}

func assemblyResultPayload(
    geom: Geometry,
    payload: [String: Any],
    run: AssemblyRun,
    outputs: [String: Any],
    residentContext: ResidentMetalContext? = nil
) throws -> [String: Any] {
    let correctionSeconds = assemblyCorrectionSeconds(run)
    let duffyReport: [String: Any]
    if let stats = run.duffyStats {
        var report = stats.toJSON()
        if geom.symmetryPlane != nil {
            report["scope"] = "real_and_image_matrix_and_rhs_duffy_delta"
            report["image_singular_correction"] = true
        }
        duffyReport = report
    } else {
        let duffyPlan: DuffyPairPlan
        if let residentContext {
            duffyPlan = residentContext.pairList.plan
        } else {
            duffyPlan = try buildDuffyPairPlan(geom)
        }
        duffyReport = [
            "implemented": false,
            "scope": "regular_quadrature_only",
            "planned_pairs": duffyPlan.toJSON(),
            "raw_triplets_if_expanded": duffyPlan.total * 9,
        ]
    }
    let aReDesc = try requireObject(outputs, "A_real_f32")
    let aImDesc = try requireObject(outputs, "A_imag_f32")
    let rhsReDesc = try requireObject(outputs, "rhs_real_f32")
    let rhsImDesc = try requireObject(outputs, "rhs_imag_f32")
    var result: [String: Any] = [
        "schema": schema,
        "op": "assemble_standard_neumann_result",
        "implementation": run.implementation,
        "assembly_mode": run.mode,
        "assembly_seconds": run.seconds,
        "regular_assembly_seconds": max(0.0, run.seconds - correctionSeconds),
        "symmetry_plane": geom.symmetryPlane.map { $0 as Any } ?? NSNull(),
        "duffy_corrections": duffyReport,
        "session_id": try requireString(payload, "session_id"),
        "frequency_hz": (payload["frequency_hz"] as? NSNumber)?.doubleValue ?? 0,
        "matrix_layout": "row_major_c",
        "matrix_shape": [geom.p1DofCount, geom.p1DofCount],
        "rhs_shape": [geom.p1DofCount],
        "matrix_real_f32": try requireString(aReDesc, "path"),
        "matrix_imag_f32": try requireString(aImDesc, "path"),
        "rhs_real_f32": try requireString(rhsReDesc, "path"),
        "rhs_imag_f32": try requireString(rhsImDesc, "path"),
    ]
    if let caseId = payload["case_id"] as? String {
        result["case_id"] = caseId
    }
    if let parity = run.parity {
        result["reference_parity"] = parity
    }
    if let dispatch = run.metalDispatch {
        result["metal_dispatch"] = dispatch
    }
    attachNearQuadratureReport(&result, run: run)
    return result
}

func assembleStandardNeumannBatch(
    sessionManifestPath: String,
    payloadPath: String,
    resultPath: String
) throws {
    let geom = try readGeometry(sessionManifestPath)
    let payload = try loadJSON(payloadPath)
    if try requireString(payload, "schema") != schema {
        try fail("unsupported schema")
    }
    if try requireString(payload, "op") != "assemble_standard_neumann_batch" {
        try fail("expected assemble_standard_neumann_batch op")
    }
    guard let cases = payload["cases"] as? [[String: Any]], !cases.isEmpty else {
        try fail("assemble_standard_neumann_batch requires non-empty cases")
    }
    let contextStart = CFAbsoluteTimeGetCurrent()
    let context = try ResidentMetalContext(geom: geom)
    let contextSeconds = CFAbsoluteTimeGetCurrent() - contextStart
    let batchStart = CFAbsoluteTimeGetCurrent()
    var caseResults: [[String: Any]] = []
    caseResults.reserveCapacity(cases.count)
    var totalAssemblySeconds = 0.0
    var totalRegularSeconds = 0.0
    for casePayload in cases {
        let k = Float(try requireDouble(casePayload, "k_real_f32"))
        let neumann = try readComplexVector(
            root: geom.root,
            descriptors: try requireObject(casePayload, "neumann_dp0"),
            count: geom.dp0DofCount
        )
        let outputs = try requireObject(casePayload, "outputs")
        let aReDesc = try requireObject(outputs, "A_real_f32")
        let aImDesc = try requireObject(outputs, "A_imag_f32")
        let rhsReDesc = try requireObject(outputs, "rhs_real_f32")
        let rhsImDesc = try requireObject(outputs, "rhs_imag_f32")
        let run = try assembleRegular(
            geom: geom,
            neumann: neumann,
            k: k,
            residentContext: context
        )
        try writeF32(try descriptorPath(root: geom.root, descriptor: aReDesc), run.arrays.aRe)
        try writeF32(try descriptorPath(root: geom.root, descriptor: aImDesc), run.arrays.aIm)
        try writeF32(try descriptorPath(root: geom.root, descriptor: rhsReDesc), run.arrays.rhsRe)
        try writeF32(try descriptorPath(root: geom.root, descriptor: rhsImDesc), run.arrays.rhsIm)
        let caseResult = try assemblyResultPayload(
            geom: geom,
            payload: [
                "session_id": try requireString(payload, "session_id"),
                "frequency_hz": (casePayload["frequency_hz"] as? NSNumber)?.doubleValue ?? 0,
                "case_id": (casePayload["case_id"] as? String) ?? "",
            ],
            run: run,
            outputs: outputs,
            residentContext: context
        )
        caseResults.append(caseResult)
        totalAssemblySeconds += run.seconds
        totalRegularSeconds += max(0.0, run.seconds - assemblyCorrectionSeconds(run))
    }
    let result: [String: Any] = [
        "schema": schema,
        "op": "assemble_standard_neumann_batch_result",
        "implementation": "swift_native_resident_metal_batch",
        "session_id": try requireString(payload, "session_id"),
        "symmetry_plane": geom.symmetryPlane.map { $0 as Any } ?? NSNull(),
        "case_count": cases.count,
        "assembly_seconds": totalAssemblySeconds,
        "regular_assembly_seconds": totalRegularSeconds,
        "resident_context_seconds": contextSeconds,
        "resident_duffy_reduction_plan_seconds": context.duffyReductionPlanBuildSeconds,
        "wall_seconds": CFAbsoluteTimeGetCurrent() - batchStart,
        "resident_reuse": [
            "geometry_buffers": true,
            "p1_incidence": true,
            "duffy_pair_list": true,
            "duffy_reduction_plan": true,
            "duffy_rules": true,
            "metal_library": true,
            "pipelines": true,
            "command_queue": true,
            "assembly_output_buffers": true,
        ],
        "cases": caseResults,
    ]
    try writeJSON(resultPath, result)
}

func assembleSolveStandardNeumannBatch(
    sessionManifestPath: String,
    payloadPath: String,
    resultPath: String
) throws {
    let geom = try readGeometry(sessionManifestPath)
    let payload = try loadJSON(payloadPath)
    if try requireString(payload, "schema") != schema {
        try fail("unsupported schema")
    }
    if try requireString(payload, "op") != "assemble_solve_standard_neumann_batch" {
        try fail("expected assemble_solve_standard_neumann_batch op")
    }
    guard let cases = payload["cases"] as? [[String: Any]], !cases.isEmpty else {
        try fail("assemble_solve_standard_neumann_batch requires non-empty cases")
    }
    let contextStart = CFAbsoluteTimeGetCurrent()
    let context = try ResidentMetalContext(geom: geom)
    let contextSeconds = CFAbsoluteTimeGetCurrent() - contextStart
    let batchStart = CFAbsoluteTimeGetCurrent()
    var caseResults: [[String: Any]] = []
    caseResults.reserveCapacity(cases.count)
    var totalAssemblySeconds = 0.0
    var totalRegularSeconds = 0.0
    var totalDenseSolveSeconds = 0.0

    for casePayload in cases {
        let k = Float(try requireDouble(casePayload, "k_real_f32"))
        let neumann = try readComplexVector(
            root: geom.root,
            descriptors: try requireObject(casePayload, "neumann_dp0"),
            count: geom.dp0DofCount
        )
        let outputs = try requireObject(casePayload, "outputs")
        let outReDesc = try requireObject(outputs, "pressure_real_f32")
        let outImDesc = try requireObject(outputs, "pressure_imag_f32")
        let run = try assembleRegular(
            geom: geom,
            neumann: neumann,
            k: k,
            residentContext: context
        )
        let solve = try solveDenseAccelerate(
            aReRowMajor: run.arrays.aRe,
            aImRowMajor: run.arrays.aIm,
            rhsRe: run.arrays.rhsRe,
            rhsIm: run.arrays.rhsIm,
            n: geom.p1DofCount
        )
        if solve.lapackInfo != 0 {
            try fail("Accelerate dense solve failed with info=\(solve.lapackInfo)")
        }
        try writeF32(
            try descriptorPath(root: geom.root, descriptor: outReDesc),
            solve.pressure.map { $0.re }
        )
        try writeF32(
            try descriptorPath(root: geom.root, descriptor: outImDesc),
            solve.pressure.map { $0.im }
        )

        let correctionSeconds = assemblyCorrectionSeconds(run)
        var caseResult: [String: Any] = [
            "schema": schema,
            "op": "assemble_solve_standard_neumann_result",
            "implementation": "swift_native_resident_metal_assembly_accelerate_solve",
            "assembly_implementation": run.implementation,
            "solve_implementation": solve.implementation,
            "assembly_mode": run.mode,
            "assembly_seconds": run.seconds,
            "regular_assembly_seconds": max(0.0, run.seconds - correctionSeconds),
            "dense_solve_seconds": solve.seconds,
            "lapack_info": solve.lapackInfo,
            "symmetry_plane": geom.symmetryPlane.map { $0 as Any } ?? NSNull(),
            "session_id": try requireString(payload, "session_id"),
            "frequency_hz": (casePayload["frequency_hz"] as? NSNumber)?.doubleValue ?? 0,
            "shape": [geom.p1DofCount],
            "pressure_real_f32": try requireString(outReDesc, "path"),
            "pressure_imag_f32": try requireString(outImDesc, "path"),
        ]
        if let rcond = solve.rcond {
            caseResult["dense_solve_rcond"] = rcond
            if rcond > 0.0 {
                caseResult["dense_solve_condition_1norm"] = 1.0 / rcond
            }
        }
        if let refineIterations = solve.refineIterations {
            caseResult["dense_solve_refine_iterations"] = refineIterations
        }
        if let refineResidualRel = solve.refineResidualRel {
            caseResult["dense_solve_refine_residual_rel"] = refineResidualRel
        }
        caseResult["dense_solve_dtype"] = solve.dtype
        if let caseId = casePayload["case_id"] as? String {
            caseResult["case_id"] = caseId
        }
        if let stats = run.duffyStats {
            var report = stats.toJSON()
            if geom.symmetryPlane != nil {
                report["scope"] = "real_and_image_matrix_and_rhs_duffy_delta"
                report["image_singular_correction"] = true
            }
            caseResult["duffy_corrections"] = report
        }
        if let dispatch = run.metalDispatch {
            caseResult["metal_dispatch"] = dispatch
        }
        attachNearQuadratureReport(&caseResult, run: run)
        caseResults.append(caseResult)
        totalAssemblySeconds += run.seconds
        totalRegularSeconds += max(0.0, run.seconds - correctionSeconds)
        totalDenseSolveSeconds += solve.seconds
    }

    let result: [String: Any] = [
        "schema": schema,
        "op": "assemble_solve_standard_neumann_batch_result",
        "implementation": "swift_native_resident_metal_assembly_accelerate_solve_batch",
        "session_id": try requireString(payload, "session_id"),
        "symmetry_plane": geom.symmetryPlane.map { $0 as Any } ?? NSNull(),
        "case_count": cases.count,
        "assembly_seconds": totalAssemblySeconds,
        "regular_assembly_seconds": totalRegularSeconds,
        "dense_solve_seconds": totalDenseSolveSeconds,
        "resident_context_seconds": contextSeconds,
        "resident_duffy_reduction_plan_seconds": context.duffyReductionPlanBuildSeconds,
        "wall_seconds": CFAbsoluteTimeGetCurrent() - batchStart,
        "resident_reuse": [
            "geometry_buffers": true,
            "assembly_output_buffers": true,
            "duffy_pair_list": true,
            "duffy_rules": true,
            "duffy_reduction_plan": true,
        ],
        "cases": caseResults,
    ]
    try writeJSON(resultPath, result)
}

struct SolvedCase {
    let assembly: AssemblyRun
    let solve: DenseSolveRun
}

final class CaseSolveResults: @unchecked Sendable {
    private let condition = NSCondition()
    private var slots: [Int: Result<SolvedCase, Error>] = [:]

    func store(_ result: Result<SolvedCase, Error>, caseIndex: Int) {
        condition.lock()
        slots[caseIndex] = result
        condition.broadcast()
        condition.unlock()
    }

    func wait(_ caseIndex: Int) throws -> SolvedCase {
        condition.lock()
        while slots[caseIndex] == nil {
            condition.wait()
        }
        let result = slots.removeValue(forKey: caseIndex)!
        condition.unlock()

        switch result {
        case .success(let solved):
            return solved
        case .failure(let error):
            throw error
        }
    }
}

func assembleSolveEvaluateStandardNeumannBatch(
    sessionManifestPath: String,
    payloadPath: String,
    resultPath: String
) throws {
    let geom = try readGeometry(sessionManifestPath)
    let payload = try loadJSON(payloadPath)
    if try requireString(payload, "schema") != schema {
        try fail("unsupported schema")
    }
    if try requireString(payload, "op") != "assemble_solve_evaluate_standard_neumann_batch" {
        try fail("expected assemble_solve_evaluate_standard_neumann_batch op")
    }
    guard let cases = payload["cases"] as? [[String: Any]], !cases.isEmpty else {
        try fail("assemble_solve_evaluate_standard_neumann_batch requires non-empty cases")
    }
    let contextStart = CFAbsoluteTimeGetCurrent()
    let context = try ResidentMetalContext(geom: geom)
    let contextSeconds = CFAbsoluteTimeGetCurrent() - contextStart
    let batchStart = CFAbsoluteTimeGetCurrent()
    var caseResults: [[String: Any]] = []
    caseResults.reserveCapacity(cases.count)
    var totalAssemblySeconds = 0.0
    var totalRegularSeconds = 0.0
    var totalDenseSolveSeconds = 0.0
    var totalFieldSeconds = 0.0
    let batchOutputs = payload["batch_outputs"] as? [String: Any]
    let batchFieldReDesc = batchOutputs?["observation_pressure_real_f32"] as? [String: Any]
    let batchFieldImDesc = batchOutputs?["observation_pressure_imag_f32"] as? [String: Any]
    if (batchFieldReDesc == nil) != (batchFieldImDesc == nil) {
        try fail("batch observation real and imag outputs must be provided together")
    }
    // Streaming contract: when case_results_dir is set, each case's result
    // manifest is written there (atomically) as soon as the case completes,
    // so the Python caller can fire per-frequency callbacks and terminate
    // this process early with all finished results already on disk.
    let caseResultsDir: String?
    if let rawCaseResultsDir = payload["case_results_dir"] {
        guard let dir = rawCaseResultsDir as? String, !dir.isEmpty else {
            try fail("case_results_dir must be a non-empty string")
        }
        if dir.hasPrefix("/") || dir.split(separator: "/").contains("..") {
            try fail("case_results_dir must be a relative path without '..'")
        }
        if batchFieldReDesc != nil {
            try fail(
                "case_results_dir requires per-case field outputs; "
                    + "batch_outputs are only written when the batch completes"
            )
        }
        caseResultsDir = URL(fileURLWithPath: geom.root)
            .appendingPathComponent(dir)
            .path
    } else {
        caseResultsDir = nil
    }
    var batchFieldReValues: [Float] = []
    var batchFieldImValues: [Float] = []
    var sharedObservationPoints: [(Float, Float, Float)]? = nil
    var sharedObservationBuffer: MTLBuffer? = nil
    var sharedObservationCount: Int? = nil
    let observationPaths = try cases.map { casePayload -> String in
        try requireString(
            try requireObject(casePayload, "observation_points"),
            "path"
        )
    }
    let allObservationPathsShared = observationPaths.allSatisfy { $0 == observationPaths[0] }
    if batchFieldReDesc != nil && !allObservationPathsShared {
        try fail("batch_outputs requires every case to use the same observation_points file")
    }
    if !observationPaths.isEmpty && allObservationPathsShared {
        let points = try readObservationPoints(
            root: geom.root,
            descriptor: try requireObject(cases[0], "observation_points")
        )
        let cached = try context.makeObservationBuffer(observationPoints: points)
        sharedObservationPoints = points
        sharedObservationBuffer = cached.buffer
        sharedObservationCount = cached.count
    }

    // Pre-read the shared CHIEF interior overdetermination points (one file
    // across all cases, like observation points). When present, every case's
    // dense solve becomes an overdetermined least-squares (zgels) solve that
    // stacks the boundary operator on the CHIEF null-field constraint rows,
    // curing the exterior-BIE fictitious-eigenvalue non-uniqueness.
    let chiefPoints: [(Float, Float, Float)]?
    let chiefWeight: Float
    if let chiefDescriptor = cases[0]["chief_points"] as? [String: Any] {
        chiefPoints = try readChiefPoints(root: geom.root, descriptor: chiefDescriptor)
        chiefWeight = Float(try optionalDouble(cases[0], "chief_weight", default: 1.0))
        if !(chiefWeight.isFinite && chiefWeight > 0) {
            try fail("chief_weight must be finite and positive")
        }
    } else {
        chiefPoints = nil
        chiefWeight = 1.0
    }

    // Pre-read per-case wavenumbers and Neumann data so case i+1's GPU
    // assembly can be committed before case i's CPU dense solve starts.
    var caseKs: [Float] = []
    var caseKImag: [Float] = []
    var caseFieldKs: [Float] = []
    var caseNeumanns: [[Complex32]] = []
    var caseRobinBetas: [[Complex32]?] = []
    caseKs.reserveCapacity(cases.count)
    caseKImag.reserveCapacity(cases.count)
    caseFieldKs.reserveCapacity(cases.count)
    caseNeumanns.reserveCapacity(cases.count)
    caseRobinBetas.reserveCapacity(cases.count)
    for casePayload in cases {
        let kReal = Float(try requireDouble(casePayload, "k_real_f32"))
        let kImag = Float(try optionalDouble(casePayload, "k_imag_f32", default: 0.0))
        let fieldK = Float(try optionalDouble(
            casePayload,
            "field_k_real_f32",
            default: Double(kReal)
        ))
        let robinBetas = try robinBetasByTriangle(geom: geom, casePayload: casePayload)
        caseKs.append(kReal)
        caseKImag.append(kImag)
        caseFieldKs.append(fieldK)
        caseNeumanns.append(
            try readComplexVector(
                root: geom.root,
                descriptors: try requireObject(casePayload, "neumann_dp0"),
                count: geom.dp0DofCount
            )
        )
        caseRobinBetas.append(robinBetas)
    }

    let assemblyMode = ProcessInfo.processInfo.environment[
        "HORNLAB_METAL_BEM_NATIVE_ASSEMBLY_MODE"
    ] ?? "optimized"
    let duffyMode = ProcessInfo.processInfo.environment[
        "HORNLAB_METAL_BEM_NATIVE_DUFFY_MODE"
    ] ?? "gpu_blocks"
    // Overlap GPU assembly of case i+1 with the CPU dense solve of case i.
    // Reference/parity assembly, block-staged assembly, and CPU Duffy
    // corrections keep the strictly sequential path: they exist to
    // cross-check the optimized kernels, not to be fast.
    let regularImplementation = try requestedRegularAssemblyImplementation()
    let pipelineAssembly = (regularImplementation == "entrywise"
            || regularImplementation == "pair_atomic")
        && (assemblyMode == "optimized"
            || (assemblyMode == "corrected" && duffyMode == "gpu_blocks"))
    let solveConcurrency = pipelineAssembly ? try requestedSolveConcurrency() : 1
    let includeDuffyBlocks = assemblyMode == "corrected"
    var pendingAssembly: ResidentMetalContext.PendingAssembly? = nil
    let solveResults = CaseSolveResults()
    let solveQueue = solveConcurrency > 1
        ? DispatchQueue(
            label: "hornlab.solve",
            qos: .userInitiated,
            attributes: .concurrent
        )
        : nil
    let solveSemaphore = solveConcurrency > 1
        ? DispatchSemaphore(value: solveConcurrency)
        : nil
    let runAhead = solveConcurrency + 1
    var nextToSubmit = 0

    func assemblyRun(
        from finished: ResidentMetalContext.FinishedAssembly,
        caseIndex: Int
    ) throws -> AssemblyRun {
        guard let blocks = finished.duffyBlocks else {
            let (nearCorrected, nearStats) = try applyNearFieldCorrectionsIfEnabled(
                to: finished.regular.arrays,
                geom: geom,
                neumann: caseNeumanns[caseIndex],
                k: caseKs[caseIndex],
                kImag: caseKImag[caseIndex],
                robinBetas: caseRobinBetas[caseIndex]
            )
            return AssemblyRun(
                arrays: nearCorrected,
                implementation: regularMetalImplementationName(finished.regular),
                mode: assemblyMode,
                seconds: finished.regularGpuSeconds + finished.readbackSeconds
                    + (nearStats?.seconds ?? 0.0),
                parity: nil,
                duffyStats: nil,
                nearStats: nearStats,
                metalDispatch: finished.regular.dispatch
            )
        }
        let (correctedArrays, reductionSeconds) = context.reduceDuffyDeltaBlocks(
            to: finished.regular.arrays,
            neumann: caseNeumanns[caseIndex],
            blocks: blocks,
            k: caseKs[caseIndex],
            kImag: caseKImag[caseIndex],
            robinBetas: caseRobinBetas[caseIndex]
        )
        let stats = DuffyCorrectionStats(
            plan: context.pairList.plan,
            rawTriplets: context.pairList.plan.total * 9,
            uniqueTriplets: context.duffyReductionPlan.matrixIndices.count,
            seconds: finished.duffyGpuSeconds + reductionSeconds,
            implementation: "metal_duffy_blocks_cpu_reduction",
            blockSeconds: finished.duffyGpuSeconds,
            reductionSeconds: reductionSeconds,
            dispatch: blocks.dispatch,
            imagePairs: context.duffyReductionPlan.imagePairs,
            reductionPrecomputed: true,
            reductionPlanBuildSeconds: context.duffyReductionPlanBuildSeconds
        )
        let (nearCorrected, nearStats) = try applyNearFieldCorrectionsIfEnabled(
            to: correctedArrays,
            geom: geom,
            neumann: caseNeumanns[caseIndex],
            k: caseKs[caseIndex],
            kImag: caseKImag[caseIndex],
            robinBetas: caseRobinBetas[caseIndex]
        )
        return AssemblyRun(
            arrays: nearCorrected,
            implementation: correctedMetalImplementationName(finished.regular, stats: stats),
            mode: assemblyMode,
            seconds: finished.regularGpuSeconds + finished.readbackSeconds
                + stats.seconds + (nearStats?.seconds ?? 0.0),
            parity: nil,
            duffyStats: stats,
            nearStats: nearStats,
            metalDispatch: finished.regular.dispatch
        )
    }

    func pipelinedAssemblyRun(_ caseIndex: Int) throws -> AssemblyRun {
        guard let pending = pendingAssembly, pending.caseIndex == caseIndex else {
            try fail("internal error: pipelined assembly is out of order")
        }
        let finished = try context.finishAssembly(pending)
        pendingAssembly = caseIndex + 1 < cases.count
            ? try context.beginAssembly(
                caseIndex: caseIndex + 1,
                neumann: caseNeumanns[caseIndex + 1],
                k: caseKs[caseIndex + 1],
                kImag: caseKImag[caseIndex + 1],
                robinBetas: caseRobinBetas[caseIndex + 1],
                includeDuffyBlocks: includeDuffyBlocks
            )
            : nil
        return try assemblyRun(from: finished, caseIndex: caseIndex)
    }

    func submitSolveJob(
        caseIndex: Int,
        finished: ResidentMetalContext.FinishedAssembly
    ) throws {
        guard let solveQueue, let solveSemaphore else {
            try fail("internal error: solve worker pool is unavailable")
        }
        solveSemaphore.wait()
        solveQueue.async {
            defer {
                solveSemaphore.signal()
            }
            do {
                let assembly = try assemblyRun(from: finished, caseIndex: caseIndex)
                let solve = try solveCaseDense(
                    arrays: assembly.arrays,
                    geom: geom,
                    chiefPoints: chiefPoints,
                    chiefWeight: chiefWeight,
                    driverNeumann: caseNeumanns[caseIndex],
                    k: caseKs[caseIndex],
                    kImag: caseKImag[caseIndex],
                    robinBetas: caseRobinBetas[caseIndex]
                )
                solveResults.store(
                    .success(SolvedCase(assembly: assembly, solve: solve)),
                    caseIndex: caseIndex
                )
            } catch {
                solveResults.store(.failure(error), caseIndex: caseIndex)
            }
        }
    }

    func pumpSubmissions(upTo caseIndex: Int) throws {
        while nextToSubmit < cases.count
            && (nextToSubmit <= caseIndex || nextToSubmit - caseIndex <= runAhead) {
            guard let pending = pendingAssembly, pending.caseIndex == nextToSubmit else {
                try fail("internal error: pipelined assembly is out of order")
            }
            let finished = try context.finishAssembly(pending)
            pendingAssembly = nextToSubmit + 1 < cases.count
                ? try context.beginAssembly(
                    caseIndex: nextToSubmit + 1,
                    neumann: caseNeumanns[nextToSubmit + 1],
                    k: caseKs[nextToSubmit + 1],
                    kImag: caseKImag[nextToSubmit + 1],
                    robinBetas: caseRobinBetas[nextToSubmit + 1],
                    includeDuffyBlocks: includeDuffyBlocks
                )
                : nil
            // The run-ahead window holds at most runAhead + 1 CPU-side
            // A/rhs/Duffy cases in flight; the semaphore bounds active solves.
            try submitSolveJob(caseIndex: nextToSubmit, finished: finished)
            nextToSubmit += 1
        }
    }

    if pipelineAssembly {
        pendingAssembly = try context.beginAssembly(
            caseIndex: 0,
            neumann: caseNeumanns[0],
            k: caseKs[0],
            kImag: caseKImag[0],
            robinBetas: caseRobinBetas[0],
            includeDuffyBlocks: includeDuffyBlocks
        )
    }

    for (caseIndex, casePayload) in cases.enumerated() {
        let k = caseKs[caseIndex]
        let kImag = caseKImag[caseIndex]
        let fieldK = caseFieldKs[caseIndex]
        let neumann = caseNeumanns[caseIndex]
        let robinBetas = caseRobinBetas[caseIndex]
        let observationPoints: [(Float, Float, Float)]
        if let sharedObservationPoints {
            observationPoints = sharedObservationPoints
        } else {
            observationPoints = try readObservationPoints(
                root: geom.root,
                descriptor: try requireObject(casePayload, "observation_points")
            )
        }
        let outputs = try requireObject(casePayload, "outputs")
        let pressureReDesc = outputs["pressure_real_f32"] as? [String: Any]
        let pressureImDesc = outputs["pressure_imag_f32"] as? [String: Any]
        if (pressureReDesc == nil) != (pressureImDesc == nil) {
            try fail("pressure_real_f32 and pressure_imag_f32 must be provided together")
        }
        let fieldReDesc = outputs["observation_pressure_real_f32"] as? [String: Any]
        let fieldImDesc = outputs["observation_pressure_imag_f32"] as? [String: Any]
        if batchFieldReDesc == nil && (fieldReDesc == nil || fieldImDesc == nil) {
            try fail("case observation pressure outputs are required without batch_outputs")
        }
        if (fieldReDesc == nil) != (fieldImDesc == nil) {
            try fail(
                "observation_pressure_real_f32 and observation_pressure_imag_f32 "
                    + "must be provided together"
            )
        }
        let sourceTags = try optionalIntArray(casePayload, "source_tags")
        let impedanceSourceTag: Int?
        if casePayload["impedance_source_tag"] == nil {
            impedanceSourceTag = nil
        } else {
            impedanceSourceTag = try requireInt(casePayload, "impedance_source_tag")
        }
        let assembly: AssemblyRun
        let solve: DenseSolveRun
        if pipelineAssembly && solveConcurrency > 1 {
            try pumpSubmissions(upTo: caseIndex)
            let solved = try solveResults.wait(caseIndex)
            assembly = solved.assembly
            solve = solved.solve
        } else if pipelineAssembly {
            assembly = try pipelinedAssemblyRun(caseIndex)
            solve = try solveCaseDense(
                arrays: assembly.arrays,
                geom: geom,
                chiefPoints: chiefPoints,
                chiefWeight: chiefWeight,
                driverNeumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas
            )
        } else {
            assembly = try assembleRegular(
                geom: geom,
                neumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas,
                residentContext: context
            )
            solve = try solveCaseDense(
                arrays: assembly.arrays,
                geom: geom,
                chiefPoints: chiefPoints,
                chiefWeight: chiefWeight,
                driverNeumann: neumann,
                k: k,
                kImag: kImag,
                robinBetas: robinBetas
            )
        }
        if solve.lapackInfo != 0 {
            try fail("Accelerate dense solve failed with info=\(solve.lapackInfo)")
        }
        let fieldNeumann = neumannWithRobin(
            geom: geom,
            driverNeumann: neumann,
            pressure: solve.pressure,
            kReal: k,
            kImag: kImag,
            robinBetas: robinBetas
        )
        let field = try evaluateExterior(
            geom: geom,
            pressure: solve.pressure,
            neumann: fieldNeumann,
            observationPoints: observationPoints,
            k: fieldK,
            residentContext: context,
            cachedObservationBuffer: sharedObservationBuffer,
            cachedObservationCount: sharedObservationCount
        )
        if let pressureReDesc, let pressureImDesc {
            try writeF32(
                try descriptorPath(root: geom.root, descriptor: pressureReDesc),
                solve.pressure.map { $0.re }
            )
            try writeF32(
                try descriptorPath(root: geom.root, descriptor: pressureImDesc),
                solve.pressure.map { $0.im }
            )
        }
        let fieldReValues = field.values.map { $0.re }
        let fieldImValues = field.values.map { $0.im }
        if batchFieldReDesc != nil {
            batchFieldReValues.append(contentsOf: fieldReValues)
            batchFieldImValues.append(contentsOf: fieldImValues)
        } else if let fieldReDesc, let fieldImDesc {
            try writeF32(
                try descriptorPath(root: geom.root, descriptor: fieldReDesc),
                fieldReValues
            )
            try writeF32(
                try descriptorPath(root: geom.root, descriptor: fieldImDesc),
                fieldImValues
            )
        }

        let correctionSeconds = assemblyCorrectionSeconds(assembly)
        var caseResult: [String: Any] = [
            "schema": schema,
            "op": "assemble_solve_evaluate_standard_neumann_result",
            "implementation": "swift_native_resident_metal_assembly_accelerate_solve_field",
            "assembly_implementation": assembly.implementation,
            "solve_implementation": solve.implementation,
            "field_implementation": field.implementation,
            "assembly_mode": assembly.mode,
            "field_mode": field.mode,
            "assembly_seconds": assembly.seconds,
            "regular_assembly_seconds": max(0.0, assembly.seconds - correctionSeconds),
            "dense_solve_seconds": solve.seconds,
            "field_seconds": field.seconds,
            "lapack_info": solve.lapackInfo,
            "symmetry_plane": geom.symmetryPlane.map { $0 as Any } ?? NSNull(),
            "session_id": try requireString(payload, "session_id"),
            "batch_id": try requireString(payload, "batch_id"),
            "frequency_hz": (casePayload["frequency_hz"] as? NSNumber)?.doubleValue ?? 0,
            "pressure_shape": [geom.p1DofCount],
            "field_shape": [observationPoints.count],
        ]
        if let rcond = solve.rcond {
            caseResult["dense_solve_rcond"] = rcond
            if rcond > 0.0 {
                caseResult["dense_solve_condition_1norm"] = 1.0 / rcond
            }
        }
        if let refineIterations = solve.refineIterations {
            caseResult["dense_solve_refine_iterations"] = refineIterations
        }
        if let refineResidualRel = solve.refineResidualRel {
            caseResult["dense_solve_refine_residual_rel"] = refineResidualRel
        }
        caseResult["dense_solve_dtype"] = solve.dtype
        if let chiefPoints {
            caseResult["chief_points"] = true
            caseResult["chief_points_count"] = chiefPoints.count
            caseResult["chief_solver"] = "accelerate_lapack_zgels"
            if let chiefResidualRel = solve.chiefResidualRel {
                caseResult["chief_residual_rel"] = chiefResidualRel
            }
        }
        if kImag != 0.0 {
            caseResult["assembly_k_imag_f32"] = kImag
            caseResult["complex_k"] = true
        }
        if robinBetas != nil {
            caseResult["robin_boundary"] = true
            caseResult["field_uses_total_neumann"] = true
        }
        if fieldK != k {
            caseResult["field_k_real_f32"] = fieldK
        }
        if let pressureReDesc, let pressureImDesc {
            caseResult["pressure_real_f32"] = try requireString(pressureReDesc, "path")
            caseResult["pressure_imag_f32"] = try requireString(pressureImDesc, "path")
        }
        if let batchFieldReDesc, let batchFieldImDesc {
            caseResult["observation_pressure_real_f32"] = try requireString(
                batchFieldReDesc,
                "path"
            )
            caseResult["observation_pressure_imag_f32"] = try requireString(
                batchFieldImDesc,
                "path"
            )
            caseResult["field_row_index"] = caseIndex
            caseResult["field_batch_shape"] = [cases.count, observationPoints.count]
            caseResult["field_output_layout"] = "batch_row_major_c"
        } else if let fieldReDesc, let fieldImDesc {
            caseResult["observation_pressure_real_f32"] = try requireString(
                fieldReDesc,
                "path"
            )
            caseResult["observation_pressure_imag_f32"] = try requireString(
                fieldImDesc,
                "path"
            )
        }
        caseResult.merge(
            nativePressureReductionPayload(
                geom: geom,
                pressure: solve.pressure,
                sourceTags: sourceTags,
                impedanceSourceTag: impedanceSourceTag
            )
        ) { _, new in new }
        if let caseId = casePayload["case_id"] as? String {
            caseResult["case_id"] = caseId
        }
        if let stats = assembly.duffyStats {
            var report = stats.toJSON()
            if geom.symmetryPlane != nil {
                report["scope"] = "real_and_image_matrix_and_rhs_duffy_delta"
                report["image_singular_correction"] = true
            }
            caseResult["duffy_corrections"] = report
        }
        if let dispatch = assembly.metalDispatch {
            caseResult["metal_dispatch"] = dispatch
        }
        if let dispatch = field.metalDispatch {
            caseResult["field_metal_dispatch"] = dispatch
        }
        attachNearQuadratureReport(&caseResult, run: assembly)
        if let caseResultsDir {
            var streamedResult = caseResult
            streamedResult["case_index"] = caseIndex
            try writeJSON(
                URL(fileURLWithPath: caseResultsDir)
                    .appendingPathComponent(String(format: "case-%04d.json", caseIndex))
                    .path,
                streamedResult
            )
        }
        caseResults.append(caseResult)
        totalAssemblySeconds += assembly.seconds
        totalRegularSeconds += max(0.0, assembly.seconds - correctionSeconds)
        totalDenseSolveSeconds += solve.seconds
        totalFieldSeconds += field.seconds
    }
    if let batchFieldReDesc, let batchFieldImDesc {
        try writeF32(
            try descriptorPath(root: geom.root, descriptor: batchFieldReDesc),
            batchFieldReValues
        )
        try writeF32(
            try descriptorPath(root: geom.root, descriptor: batchFieldImDesc),
            batchFieldImValues
        )
    }

    let result: [String: Any] = [
        "schema": schema,
        "op": "assemble_solve_evaluate_standard_neumann_batch_result",
        "implementation": "swift_native_resident_metal_assembly_accelerate_solve_field_batch",
        "session_id": try requireString(payload, "session_id"),
        "batch_id": try requireString(payload, "batch_id"),
        "symmetry_plane": geom.symmetryPlane.map { $0 as Any } ?? NSNull(),
        "case_count": cases.count,
        "assembly_seconds": totalAssemblySeconds,
        "regular_assembly_seconds": totalRegularSeconds,
        "dense_solve_seconds": totalDenseSolveSeconds,
        "dense_solve_concurrency": solveConcurrency,
        "field_seconds": totalFieldSeconds,
        "resident_context_seconds": contextSeconds,
        "resident_duffy_reduction_plan_seconds": context.duffyReductionPlanBuildSeconds,
        "wall_seconds": CFAbsoluteTimeGetCurrent() - batchStart,
        "assembly_solve_overlap": pipelineAssembly,
        "streamed_case_results": caseResultsDir != nil,
        "resident_reuse": [
            "geometry_buffers": true,
            "assembly_output_buffers": true,
            "duffy_pair_list": true,
            "duffy_rules": true,
            "duffy_reduction_plan": true,
            "field_output_buffers": true,
            "batched_field_output_files": batchFieldReDesc != nil,
            "observation_points_buffer": sharedObservationBuffer != nil,
        ],
        "cases": caseResults,
    ]
    try writeJSON(resultPath, result)
}

func evaluateStandardExterior(
    sessionManifestPath: String,
    payloadPath: String,
    resultPath: String
) throws {
    let geom = try readGeometry(sessionManifestPath)
    let payload = try loadJSON(payloadPath)
    if try requireString(payload, "schema") != schema {
        try fail("unsupported schema")
    }
    if try requireString(payload, "op") != "evaluate_standard_exterior" {
        try fail("expected evaluate_standard_exterior op")
    }
    let k = Float(try requireDouble(payload, "k_real_f32"))
    let pressure = try readComplexVector(
        root: geom.root,
        descriptors: try requireObject(payload, "pressure_p1"),
        count: geom.p1DofCount
    )
    let neumann = try readComplexVector(
        root: geom.root,
        descriptors: try requireObject(payload, "neumann_dp0"),
        count: geom.dp0DofCount
    )
    let observationPoints = try readObservationPoints(
        root: geom.root,
        descriptor: try requireObject(payload, "observation_points")
    )
    let run = try evaluateExterior(
        geom: geom,
        pressure: pressure,
        neumann: neumann,
        observationPoints: observationPoints,
        k: k
    )
    let output = try requireObject(payload, "output")
    let outReDesc = try requireObject(output, "pressure_real_f32")
    let outImDesc = try requireObject(output, "pressure_imag_f32")
    try writeF32(
        try descriptorPath(root: geom.root, descriptor: outReDesc),
        run.values.map { $0.re }
    )
    try writeF32(
        try descriptorPath(root: geom.root, descriptor: outImDesc),
        run.values.map { $0.im }
    )
    var result: [String: Any] = [
        "schema": schema,
        "op": "evaluate_standard_exterior_result",
        "implementation": run.implementation,
        "field_mode": run.mode,
        "field_seconds": run.seconds,
        "symmetry_plane": geom.symmetryPlane.map { $0 as Any } ?? NSNull(),
        "session_id": try requireString(payload, "session_id"),
        "batch_id": try requireString(payload, "batch_id"),
        "frequency_hz": (payload["frequency_hz"] as? NSNumber)?.doubleValue ?? 0,
        "shape": [observationPoints.count],
        "pressure_real_f32": try requireString(outReDesc, "path"),
        "pressure_imag_f32": try requireString(outImDesc, "path"),
    ]
    if let parity = run.parity {
        result["reference_parity"] = parity
    }
    if let dispatch = run.metalDispatch {
        result["metal_dispatch"] = dispatch
    }
    try writeJSON(resultPath, result)
}

func fieldResultPayload(
    payload: [String: Any],
    run: FieldRun,
    output: [String: Any],
    observationCount: Int,
    symmetryPlane: String?
) throws -> [String: Any] {
    let outReDesc = try requireObject(output, "pressure_real_f32")
    let outImDesc = try requireObject(output, "pressure_imag_f32")
    var result: [String: Any] = [
        "schema": schema,
        "op": "evaluate_standard_exterior_result",
        "implementation": run.implementation,
        "field_mode": run.mode,
        "field_seconds": run.seconds,
        "symmetry_plane": symmetryPlane.map { $0 as Any } ?? NSNull(),
        "session_id": try requireString(payload, "session_id"),
        "batch_id": try requireString(payload, "batch_id"),
        "frequency_hz": (payload["frequency_hz"] as? NSNumber)?.doubleValue ?? 0,
        "shape": [observationCount],
        "pressure_real_f32": try requireString(outReDesc, "path"),
        "pressure_imag_f32": try requireString(outImDesc, "path"),
    ]
    if let caseId = payload["case_id"] as? String {
        result["case_id"] = caseId
    }
    if let parity = run.parity {
        result["reference_parity"] = parity
    }
    if let dispatch = run.metalDispatch {
        result["metal_dispatch"] = dispatch
    }
    return result
}

func evaluateStandardExteriorBatch(
    sessionManifestPath: String,
    payloadPath: String,
    resultPath: String
) throws {
    let geom = try readGeometry(sessionManifestPath)
    let payload = try loadJSON(payloadPath)
    if try requireString(payload, "schema") != schema {
        try fail("unsupported schema")
    }
    if try requireString(payload, "op") != "evaluate_standard_exterior_batch" {
        try fail("expected evaluate_standard_exterior_batch op")
    }
    guard let cases = payload["cases"] as? [[String: Any]], !cases.isEmpty else {
        try fail("evaluate_standard_exterior_batch requires non-empty cases")
    }
    let sharedObservationPoints: [(Float, Float, Float)]?
    if let sharedObservationDescriptor = payload["observation_points"] as? [String: Any] {
        sharedObservationPoints = try readObservationPoints(
            root: geom.root,
            descriptor: sharedObservationDescriptor
        )
    } else {
        sharedObservationPoints = nil
    }
    let context = try ResidentMetalContext(geom: geom)
    let batchStart = CFAbsoluteTimeGetCurrent()
    var caseResults: [[String: Any]] = []
    caseResults.reserveCapacity(cases.count)
    var totalFieldSeconds = 0.0
    for casePayload in cases {
        let k = Float(try requireDouble(casePayload, "k_real_f32"))
        let pressure = try readComplexVector(
            root: geom.root,
            descriptors: try requireObject(casePayload, "pressure_p1"),
            count: geom.p1DofCount
        )
        let neumann = try readComplexVector(
            root: geom.root,
            descriptors: try requireObject(casePayload, "neumann_dp0"),
            count: geom.dp0DofCount
        )
        let observationPoints: [(Float, Float, Float)]
        if let sharedObservationPoints {
            observationPoints = sharedObservationPoints
        } else {
            observationPoints = try readObservationPoints(
                root: geom.root,
                descriptor: try requireObject(casePayload, "observation_points")
            )
        }
        let run = try evaluateExterior(
            geom: geom,
            pressure: pressure,
            neumann: neumann,
            observationPoints: observationPoints,
            k: k,
            residentContext: context
        )
        let output = try requireObject(casePayload, "output")
        let outReDesc = try requireObject(output, "pressure_real_f32")
        let outImDesc = try requireObject(output, "pressure_imag_f32")
        try writeF32(
            try descriptorPath(root: geom.root, descriptor: outReDesc),
            run.values.map { $0.re }
        )
        try writeF32(
            try descriptorPath(root: geom.root, descriptor: outImDesc),
            run.values.map { $0.im }
        )
        let caseResult = try fieldResultPayload(
            payload: [
                "session_id": try requireString(payload, "session_id"),
                "batch_id": try requireString(payload, "batch_id"),
                "frequency_hz": (casePayload["frequency_hz"] as? NSNumber)?.doubleValue ?? 0,
                "case_id": (casePayload["case_id"] as? String) ?? "",
            ],
            run: run,
            output: output,
            observationCount: observationPoints.count,
            symmetryPlane: geom.symmetryPlane
        )
        caseResults.append(caseResult)
        totalFieldSeconds += run.seconds
    }
    let result: [String: Any] = [
        "schema": schema,
        "op": "evaluate_standard_exterior_batch_result",
        "implementation": "swift_native_resident_metal_field_batch",
        "session_id": try requireString(payload, "session_id"),
        "batch_id": try requireString(payload, "batch_id"),
        "symmetry_plane": geom.symmetryPlane.map { $0 as Any } ?? NSNull(),
        "case_count": cases.count,
        "field_seconds": totalFieldSeconds,
        "wall_seconds": CFAbsoluteTimeGetCurrent() - batchStart,
        "resident_reuse": [
            "geometry_buffers": true,
            "metal_library": true,
            "pipelines": true,
            "command_queue": true,
            "field_output_buffers": true,
        ],
        "cases": caseResults,
    ]
    try writeJSON(resultPath, result)
}

func smoke() throws {
    let device = try MetalWarmup.shared.device()
    print("hornlab-metal-bem native Metal helper smoke ok: \(device.name)")
}

func main(_ args: [String]) throws {
    if args.count == 1 && args[0] == "--smoke" {
        MetalWarmup.shared.begin()
        try smoke()
        return
    }
    guard let op = args.first else {
        try fail("usage: HornlabMetalBemNative <operation> <session.json> [<payload.json>] <result.json>")
    }
    if op != "validate_session" {
        MetalWarmup.shared.begin()
    }
    if op == "validate_session" {
        guard args.count == 3 else {
            try fail("usage: HornlabMetalBemNative.swift validate_session <session.json> <result.json>")
        }
        let sessionPath = args[1]
        let resultPath = args[2]
        let result = try validateSession(loadJSON(sessionPath))
        try writeJSON(resultPath, result)
    } else if op == "assemble_standard_neumann" {
        guard args.count == 4 else {
            try fail("usage: HornlabMetalBemNative.swift assemble_standard_neumann <session.json> <payload.json> <result.json>")
        }
        try assembleStandardNeumann(
            sessionManifestPath: args[1],
            payloadPath: args[2],
            resultPath: args[3]
        )
    } else if op == "assemble_standard_neumann_batch" {
        guard args.count == 4 else {
            try fail("usage: HornlabMetalBemNative.swift assemble_standard_neumann_batch <session.json> <payload.json> <result.json>")
        }
        try assembleStandardNeumannBatch(
            sessionManifestPath: args[1],
            payloadPath: args[2],
            resultPath: args[3]
        )
    } else if op == "assemble_solve_standard_neumann_batch" {
        guard args.count == 4 else {
            try fail("usage: HornlabMetalBemNative.swift assemble_solve_standard_neumann_batch <session.json> <payload.json> <result.json>")
        }
        try assembleSolveStandardNeumannBatch(
            sessionManifestPath: args[1],
            payloadPath: args[2],
            resultPath: args[3]
        )
    } else if op == "assemble_solve_evaluate_standard_neumann_batch" {
        guard args.count == 4 else {
            try fail("usage: HornlabMetalBemNative.swift assemble_solve_evaluate_standard_neumann_batch <session.json> <payload.json> <result.json>")
        }
        try assembleSolveEvaluateStandardNeumannBatch(
            sessionManifestPath: args[1],
            payloadPath: args[2],
            resultPath: args[3]
        )
    } else if op == "evaluate_standard_exterior" {
        guard args.count == 4 else {
            try fail("usage: HornlabMetalBemNative.swift evaluate_standard_exterior <session.json> <payload.json> <result.json>")
        }
        try evaluateStandardExterior(
            sessionManifestPath: args[1],
            payloadPath: args[2],
            resultPath: args[3]
        )
    } else if op == "evaluate_standard_exterior_batch" {
        guard args.count == 4 else {
            try fail("usage: HornlabMetalBemNative.swift evaluate_standard_exterior_batch <session.json> <payload.json> <result.json>")
        }
        try evaluateStandardExteriorBatch(
            sessionManifestPath: args[1],
            payloadPath: args[2],
            resultPath: args[3]
        )
    } else {
        try fail("unsupported operation: \(op)")
    }
}

do {
    try main(Array(CommandLine.arguments.dropFirst()))
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
