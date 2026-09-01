import Accelerate
import Foundation

// Block-Jacobi preconditioned GMRES for the dense standard-Neumann system.
//
// The dense LU is O(N^3) and is what caps this solver near 20,000 DOF. Measured
// on this codebase's own operators (2026-09-01), block-Jacobi GMRES converges in
// 20-35 iterations across kD 5-202 and stays flat as the mesh refines, so the
// solve becomes O(iterations * N^2) -- the same order as the assembly that
// already dominates the wall clock. It agreed with the LU solution to 1e-8
// relative, which is below this path's own float32 assembly noise.
//
// Opt in with SolveConfig(dense_solve_implementation="gmres"). The helper reads
// HORNLAB_METAL_BEM_NATIVE_DENSE_SOLVE_IMPL, but the Python sweep sets that
// variable from the config field on every solve, so a value placed in the
// process environment never reaches a Python-driven solve.
//
// One caveat is load-bearing. On a CLOSED body the real-k `standard` formulation
// carries uncured interior resonances whose modal density above ~6 kHz is high
// enough that no frequency placement avoids them: measured 344 iterations at
// 8 kHz against 55 with a complex-k shift, on the same mesh. The formulation is
// a Python-side concern, so the guard against that combination lives there.

/// P1 dof coordinates, derived from the triangle->dof map rather than assumed to
/// equal vertex order.
func p1DofCoordinates(_ geom: Geometry) -> (x: [Float], y: [Float], z: [Float]) {
    var x = [Float](repeating: 0, count: geom.p1DofCount)
    var y = x
    var z = x
    for t in 0..<geom.nTriangles {
        for l in 0..<3 {
            let dof = geom.p1Dof(t, l)
            let vertex = geom.triangleVertex(t, l)
            x[dof] = geom.px[vertex]
            y[dof] = geom.py[vertex]
            z[dof] = geom.pz[vertex]
        }
    }
    return (x, y, z)
}

/// Median split on the longest bounding-box axis until every leaf fits `leafSize`.
/// Spatial locality is the whole point: a block-diagonal preconditioner only helps
/// if its blocks capture the near-field coupling, which is what dominates the
/// operator.
func clusterLeaves(x: [Float], y: [Float], z: [Float], leafSize: Int) -> [[Int]] {
    var leaves: [[Int]] = []
    var stack: [[Int]] = [Array(0..<x.count)]
    while let indices = stack.popLast() {
        if indices.count <= leafSize {
            leaves.append(indices)
            continue
        }
        var lo = [Float](repeating: .greatestFiniteMagnitude, count: 3)
        var hi = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        for i in indices {
            let point = [x[i], y[i], z[i]]
            for c in 0..<3 {
                lo[c] = min(lo[c], point[c])
                hi[c] = max(hi[c], point[c])
            }
        }
        var axis = 0
        for c in 1..<3 where hi[c] - lo[c] > hi[axis] - lo[axis] {
            axis = c
        }
        let coordinate: [Float] = axis == 0 ? x : (axis == 1 ? y : z)
        let sorted = indices.sorted { coordinate[$0] < coordinate[$1] }
        let half = sorted.count / 2
        stack.append(Array(sorted[0..<half]))
        stack.append(Array(sorted[half...]))
    }
    return leaves
}

/// LU factors of the diagonal blocks, applied as an approximate inverse.
struct BlockJacobiPreconditioner {
    var blocks: [[Int]]
    var factors: [[__CLPK_complex]]
    var pivots: [[__CLPK_integer]]
    /// Blocks LAPACK reported singular. Those act as identity rather than
    /// producing NaN, so a degenerate block degrades convergence instead of
    /// destroying the solve.
    var singularBlocks: Int

    func apply(re: [Float], im: [Float]) -> (re: [Float], im: [Float]) {
        var outRe = re
        var outIm = im
        for (b, block) in blocks.enumerated() {
            let m = block.count
            if pivots[b].isEmpty { continue }
            var rhs = [__CLPK_complex](repeating: __CLPK_complex(r: 0, i: 0), count: m)
            for (local, global) in block.enumerated() {
                rhs[local] = __CLPK_complex(r: re[global], i: im[global])
            }
            var factored = factors[b]
            var piv = pivots[b]
            var trans = Int8(78)  // 'N'
            var mm = __CLPK_integer(m)
            var nrhs = __CLPK_integer(1)
            var lda = __CLPK_integer(m)
            var ldb = __CLPK_integer(m)
            var info = __CLPK_integer(0)
            cgetrs_(&trans, &mm, &nrhs, &factored, &lda, &piv, &rhs, &ldb, &info)
            if info != 0 { continue }
            for (local, global) in block.enumerated() {
                outRe[global] = rhs[local].r
                outIm[global] = rhs[local].i
            }
        }
        return (outRe, outIm)
    }
}

func buildBlockJacobi(
    aReRowMajor: [Float],
    aImRowMajor: [Float],
    n: Int,
    blocks: [[Int]]
) -> BlockJacobiPreconditioner {
    var factors: [[__CLPK_complex]] = []
    var pivots: [[__CLPK_integer]] = []
    var singular = 0
    factors.reserveCapacity(blocks.count)
    pivots.reserveCapacity(blocks.count)
    for block in blocks {
        let m = block.count
        var sub = [__CLPK_complex](repeating: __CLPK_complex(r: 0, i: 0), count: m * m)
        for (cj, j) in block.enumerated() {
            for (ci, i) in block.enumerated() {
                let source = i * n + j          // row-major A[i][j]
                sub[cj * m + ci] = __CLPK_complex(   // column-major for LAPACK
                    r: aReRowMajor[source],
                    i: aImRowMajor[source]
                )
            }
        }
        var rows = __CLPK_integer(m)
        var cols = __CLPK_integer(m)
        var lda = __CLPK_integer(m)
        var info = __CLPK_integer(0)
        var piv = [__CLPK_integer](repeating: 0, count: m)
        cgetrf_(&rows, &cols, &sub, &lda, &piv, &info)
        if info != 0 {
            singular += 1
            factors.append([])
            pivots.append([])
        } else {
            factors.append(sub)
            pivots.append(piv)
        }
    }
    return BlockJacobiPreconditioner(
        blocks: blocks, factors: factors, pivots: pivots, singularBlocks: singular
    )
}

/// Complex matvec over the split real/imaginary row-major arrays the assembler
/// already produces.
///
/// Deliberately does NOT build an interleaved copy: at the mesh sizes this path
/// exists to reach, an extra N^2 allocation is the binding constraint. Stacking
/// the real and imaginary parts of x as two columns lets one `sgemm` per matrix
/// half produce both products, so the matrix is read exactly twice per matvec --
/// the same traffic an interleaved `cgemv` would pay, with none of the memory.
final class DenseComplexOperator {
    let aRe: [Float]
    let aIm: [Float]
    let n: Int
    private var stacked: [Float]
    private var productRe: [Float]
    private var productIm: [Float]

    init(aRe: [Float], aIm: [Float], n: Int) {
        self.aRe = aRe
        self.aIm = aIm
        self.n = n
        stacked = [Float](repeating: 0, count: n * 2)
        productRe = stacked
        productIm = stacked
    }

    private static func gemm(_ a: [Float], _ b: [Float], _ c: inout [Float], _ n: Int) {
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(n), 2, Int32(n), 1.0,
                        ap.baseAddress, Int32(n),
                        bp.baseAddress, 2,
                        0.0, cp.baseAddress, 2
                    )
                }
            }
        }
    }

    func apply(xRe: [Float], xIm: [Float]) -> (re: [Float], im: [Float]) {
        for i in 0..<n {
            stacked[2 * i] = xRe[i]
            stacked[2 * i + 1] = xIm[i]
        }
        DenseComplexOperator.gemm(aRe, stacked, &productRe, n)
        DenseComplexOperator.gemm(aIm, stacked, &productIm, n)
        var yRe = [Float](repeating: 0, count: n)
        var yIm = yRe
        for i in 0..<n {
            yRe[i] = productRe[2 * i] - productIm[2 * i + 1]
            yIm[i] = productRe[2 * i + 1] + productIm[2 * i]
        }
        return (yRe, yIm)
    }
}

struct GmresRun {
    var xRe: [Float]
    var xIm: [Float]
    var iterations: Int
    var relativeResidual: Double
    var converged: Bool
}

private struct C2 {
    var re: Double
    var im: Double
    static let zero = C2(re: 0, im: 0)
    var abs: Double { (re * re + im * im).squareRoot() }
    static func * (a: C2, b: C2) -> C2 {
        C2(re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re)
    }
    static func + (a: C2, b: C2) -> C2 { C2(re: a.re + b.re, im: a.im + b.im) }
    static func - (a: C2, b: C2) -> C2 { C2(re: a.re - b.re, im: a.im - b.im) }
    var conj: C2 { C2(re: re, im: -im) }
}

private func dotConj(_ aRe: [Float], _ aIm: [Float], _ bRe: [Float], _ bIm: [Float]) -> C2 {
    // <a, b> = sum conj(a) * b, accumulated in double: the Krylov vectors are
    // float32 and orthogonality is what GMRES is most sensitive to losing.
    var re = 0.0
    var im = 0.0
    for i in 0..<aRe.count {
        let ar = Double(aRe[i]), ai = Double(aIm[i])
        let br = Double(bRe[i]), bi = Double(bIm[i])
        re += ar * br + ai * bi
        im += ar * bi - ai * br
    }
    return C2(re: re, im: im)
}

private func norm2(_ re: [Float], _ im: [Float]) -> Double {
    var sum = 0.0
    for i in 0..<re.count {
        sum += Double(re[i]) * Double(re[i]) + Double(im[i]) * Double(im[i])
    }
    return sum.squareRoot()
}

/// Left-preconditioned restarted GMRES.
func gmresSolve(
    operatorA: DenseComplexOperator,
    preconditioner: BlockJacobiPreconditioner,
    bRe: [Float],
    bIm: [Float],
    restart: Int,
    maxIterations: Int,
    tolerance: Double
) -> GmresRun {
    let n = operatorA.n
    var xRe = [Float](repeating: 0, count: n)
    var xIm = xRe

    let preconditionedRhs = preconditioner.apply(re: bRe, im: bIm)
    let referenceNorm = norm2(preconditionedRhs.re, preconditionedRhs.im)
    if referenceNorm == 0 || !referenceNorm.isFinite {
        return GmresRun(xRe: xRe, xIm: xIm, iterations: 0,
                        relativeResidual: 0, converged: true)
    }

    var total = 0
    var relative = 1.0
    var firstOuter = true

    while total < maxIterations {
        var rRe: [Float]
        var rIm: [Float]
        if firstOuter {
            rRe = bRe
            rIm = bIm
        } else {
            let ax = operatorA.apply(xRe: xRe, xIm: xIm)
            rRe = [Float](repeating: 0, count: n)
            rIm = rRe
            for i in 0..<n {
                rRe[i] = bRe[i] - ax.re[i]
                rIm[i] = bIm[i] - ax.im[i]
            }
        }
        firstOuter = false
        let z = preconditioner.apply(re: rRe, im: rIm)
        let beta = norm2(z.re, z.im)
        relative = beta / referenceNorm
        if relative <= tolerance || !beta.isFinite { break }

        let m = min(restart, maxIterations - total)
        if m < 1 { break }
        var basisRe: [[Float]] = []
        var basisIm: [[Float]] = []
        basisRe.reserveCapacity(m + 1)
        basisIm.reserveCapacity(m + 1)
        let inv = Float(1.0 / beta)
        basisRe.append(z.re.map { $0 * inv })
        basisIm.append(z.im.map { $0 * inv })

        var h = [[C2]](repeating: [C2](repeating: .zero, count: m), count: m + 1)
        var cs = [Double](repeating: 0, count: m)
        var sn = [C2](repeating: .zero, count: m)
        var g = [C2](repeating: .zero, count: m + 1)
        g[0] = C2(re: beta, im: 0)

        var used = 0
        for j in 0..<m {
            let av = operatorA.apply(xRe: basisRe[j], xIm: basisIm[j])
            var w = preconditioner.apply(re: av.re, im: av.im)
            // Modified Gram-Schmidt, then one reorthogonalisation pass. The
            // basis is float32; a single pass loses orthogonality well before
            // the iteration counts this solver needs.
            for pass in 0..<2 {
                for i in 0...j {
                    let coeff = dotConj(basisRe[i], basisIm[i], w.re, w.im)
                    for t in 0..<n {
                        let vr = Double(basisRe[i][t]), vi = Double(basisIm[i][t])
                        w.re[t] = Float(Double(w.re[t]) - (coeff.re * vr - coeff.im * vi))
                        w.im[t] = Float(Double(w.im[t]) - (coeff.re * vi + coeff.im * vr))
                    }
                    h[i][j] = pass == 0 ? coeff : h[i][j] + coeff
                }
            }
            let hNext = norm2(w.re, w.im)
            h[j + 1][j] = C2(re: hNext, im: 0)
            if hNext > 0 {
                let scale = Float(1.0 / hNext)
                basisRe.append(w.re.map { $0 * scale })
                basisIm.append(w.im.map { $0 * scale })
            } else {
                basisRe.append([Float](repeating: 0, count: n))
                basisIm.append([Float](repeating: 0, count: n))
            }

            for i in 0..<j {
                let t1 = h[i][j]
                let t2 = h[i + 1][j]
                h[i][j] = C2(re: cs[i] * t1.re, im: cs[i] * t1.im) + sn[i] * t2
                h[i + 1][j] = C2(re: cs[i] * t2.re, im: cs[i] * t2.im)
                    - (sn[i].conj * t1)
            }

            // Complex Givens: c real, s complex, [c s; -conj(s) c] zeroes h[j+1][j].
            let a = h[j][j]
            let bb = h[j + 1][j]
            let absA = a.abs
            let absB = bb.abs
            let d = (absA * absA + absB * absB).squareRoot()
            if d == 0 {
                cs[j] = 1
                sn[j] = .zero
            } else if absA == 0 {
                cs[j] = 0
                sn[j] = C2(re: bb.conj.re / absB, im: bb.conj.im / absB)
                h[j][j] = C2(re: absB, im: 0)
            } else {
                cs[j] = absA / d
                let unit = C2(re: a.re / absA, im: a.im / absA)
                let sc = unit * bb.conj
                sn[j] = C2(re: sc.re / d, im: sc.im / d)
                h[j][j] = C2(re: a.re * d / absA, im: a.im * d / absA)
            }
            h[j + 1][j] = .zero
            let gj = g[j]
            g[j] = C2(re: cs[j] * gj.re, im: cs[j] * gj.im)
            g[j + 1] = C2(re: -1, im: 0) * (sn[j].conj * gj)

            total += 1
            used = j + 1
            relative = g[j + 1].abs / referenceNorm
            if relative <= tolerance { break }
        }

        if used > 0 {
            var y = [C2](repeating: .zero, count: used)
            for i in stride(from: used - 1, through: 0, by: -1) {
                var acc = g[i]
                for k in (i + 1)..<used {
                    acc = acc - (h[i][k] * y[k])
                }
                let denom = h[i][i]
                let scale = denom.re * denom.re + denom.im * denom.im
                if scale == 0 { y[i] = .zero; continue }
                let num = denom.conj * acc
                y[i] = C2(re: num.re / scale, im: num.im / scale)
            }
            for i in 0..<used {
                let yr = y[i].re, yi = y[i].im
                for t in 0..<n {
                    let vr = Double(basisRe[i][t]), vi = Double(basisIm[i][t])
                    xRe[t] = Float(Double(xRe[t]) + (yr * vr - yi * vi))
                    xIm[t] = Float(Double(xIm[t]) + (yr * vi + yi * vr))
                }
            }
        }
        if relative <= tolerance { break }
    }

    // Report the TRUE unpreconditioned residual, not the preconditioned estimate
    // GMRES minimises. They differ, and the honest number is the one a caller
    // would compute for themselves.
    let ax = operatorA.apply(xRe: xRe, xIm: xIm)
    var resRe = [Float](repeating: 0, count: n)
    var resIm = resRe
    for i in 0..<n {
        resRe[i] = bRe[i] - ax.re[i]
        resIm[i] = bIm[i] - ax.im[i]
    }
    let trueRelative = norm2(resRe, resIm) / max(norm2(bRe, bIm), Double.leastNormalMagnitude)
    return GmresRun(
        xRe: xRe, xIm: xIm, iterations: total,
        relativeResidual: trueRelative,
        converged: trueRelative <= max(tolerance * 10.0, 1e-5)
    )
}
