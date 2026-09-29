import Foundation
import Accelerate

// usage: schur_race N M THREADS ITERS [serialize-stage]
// Replays the helper's coupled-IB Schur solve (zgesv M x M with N RHS, zgecon,
// cgemm N x N x M, cgesv N x 1, cgecon, cgemm M x 1 x N) on fixed random data,
// concurrently on THREADS GCD threads, and compares every stage bit-for-bit
// with a serial reference.
let a = CommandLine.arguments
let n = Int(a[1])!, m = Int(a[2])!, threads = Int(a[3])!, iters = Int(a[4])!
let lockStage = a.count > 5 ? a[5] : ""
let cold = ProcessInfo.processInfo.environment["COLD"] != nil
var g = SystemRandomNumberGenerator()
func rc() -> __CLPK_complex { .init(r: Float.random(in: -1...1, using: &g), i: Float.random(in: -1...1, using: &g)) }
var saa = [__CLPK_doublecomplex](repeating: .init(r: 0, i: 0), count: m * m)
for i in 0..<(m * m) { saa[i] = .init(r: Double.random(in: -1...1), i: Double.random(in: -1...1)) }
for i in 0..<m { saa[i * m + i].r += Double(m) * 0.05 }
var pavg = [__CLPK_doublecomplex](repeating: .init(r: 0, i: 0), count: m * n)
for i in 0..<(m * n) where i % 7 == 0 { pavg[i] = .init(r: 1.0 / 3.0, i: 0) }
var sia0 = (0..<(n * m)).map { _ in rc() }
for i in 0..<(n * m) { sia0[i].r *= 0.01; sia0[i].i *= 0.01 }
var A0 = (0..<(n * n)).map { _ in rc() }
for i in 0..<n { A0[i * n + i].r += Float(n) * 0.05 }
let b0 = (0..<n).map { _ in rc() }
let serial = NSLock()
func h64<T>(_ x: [T]) -> UInt64 { x.withUnsafeBytes { raw in var h: UInt64 = 1469598103934665603; for b in raw { h = (h ^ UInt64(b)) &* 1099511628211 }; return h } }
func run() -> [UInt64] {
    var hs: [UInt64] = []
    var am = saa, rhs = pavg
    var mm = __CLPK_integer(m), nr = __CLPK_integer(n), lda = __CLPK_integer(m), ldb = __CLPK_integer(m), info = __CLPK_integer(0)
    var piv = [__CLPK_integer](repeating: 0, count: m)
    if lockStage.split(separator: "+").contains("zgesv") { serial.lock() }
    nzgesv(&mm, &nr, &am, &lda, &piv, &rhs, &ldb, &info)
    if lockStage.split(separator: "+").contains("zgesv") { serial.unlock() }
    hs.append(h64(rhs))
    var t = [__CLPK_complex](repeating: .init(r: 0, i: 0), count: m * n)
    for r in 0..<m { for c in 0..<n { t[r * n + c] = .init(r: Float(rhs[c * m + r].r), i: Float(rhs[c * m + r].i)) } }
    var sia = sia0, schur = A0
    var al = __CLPK_complex(r: -1, i: 0), be = __CLPK_complex(r: 1, i: 0)
    if lockStage.split(separator: "+").contains("cgemm") { serial.lock() }
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(n), Int32(n), Int32(m), &al, &sia, Int32(m), &t, Int32(n), &be, &schur, Int32(n))
    if lockStage.split(separator: "+").contains("cgemm") { serial.unlock() }
    hs.append(h64(schur))
    var mat = [__CLPK_complex](repeating: .init(r: 0, i: 0), count: n * n)
    for r in 0..<n { for c in 0..<n { mat[c * n + r] = schur[r * n + c] } }
    var b = b0
    var nn = __CLPK_integer(n), one = __CLPK_integer(1), ld1 = __CLPK_integer(n), ld2 = __CLPK_integer(n), info2 = __CLPK_integer(0)
    var piv2 = [__CLPK_integer](repeating: 0, count: n)
    if lockStage.split(separator: "+").contains("cgesv") { serial.lock() }
    ncgesv(&nn, &one, &mat, &ld1, &piv2, &b, &ld2, &info2)
    if lockStage.split(separator: "+").contains("cgesv") { serial.unlock() }
    hs.append(h64(b))
    hs.append(b.allSatisfy { $0.r.isFinite && $0.i.isFinite } ? 0 : 1)
    return hs
}
var ref: [UInt64] = cold ? [] : run()
var results: [[UInt64]] = []
let lk = NSLock(); var total = 0; var bad = [0, 0, 0]; var nonfinite = 0
let mode = ProcessInfo.processInfo.environment["MODE"] ?? "gcd"
let whole = NSLock()
func body() {
    for _ in 0..<iters {
        if lockStage == "whole" { whole.lock() }
        let h = run()
        if lockStage == "whole" { whole.unlock() }
        lk.lock(); results.append(h); lk.unlock()
    }
}
if mode == "gcd" {
    DispatchQueue.concurrentPerform(iterations: threads) { _ in body() }
} else if mode == "gcdasync" {
    let grp = DispatchGroup()
    let q = DispatchQueue(label: "w", attributes: .concurrent)
    for _ in 0..<threads { q.async(group: grp) { body() } }
    grp.wait()
} else {
    let stackMB = Int(mode.dropFirst(6))!  // "thread8" etc
    let grp = DispatchGroup()
    for _ in 0..<threads {
        grp.enter()
        let t = Thread { body(); grp.leave() }
        t.stackSize = stackMB * 1024 * 1024
        t.start()
    }
    grp.wait()
}
if cold { ref = run() }
for h in results { total += 1; for s in 0..<3 where h[s] != ref[s] { bad[s] += 1 }; if h[3] != 0 { nonfinite += 1 } }
let ref2 = run()
print("refNaN \(ref[3]) ref2NaN \(ref2[3]) ref2==ref \(ref2 == ref) perThreadNaN \(results.map { $0[3] })")
print("n \(n) m \(m) threads \(threads) lock=\(lockStage) runs \(total) mismatch zgesv/cgemm/cgesv \(bad) nonfinite \(nonfinite)")
