import Accelerate

// Accelerate's current LAPACK interface.
//
// Package.swift defines ACCELERATE_NEW_LAPACK for this target, so the
// `<name>_` functions imported below bind to the `$NEWLAPACK` entry points.
// The legacy CLAPACK entry points (`__CLPK_*` types) return NaN when they run
// concurrently with other Accelerate calls, which the coupled infinite-baffle
// solve pipeline does. Do not remove the define, and do not call the legacy
// `clapack.h` interface: a test checks the built helper links no legacy symbol.
//
// Swift cannot import C99 `_Complex` types, so the complex LAPACK arguments are
// declared here as layout-compatible structs and passed through OpaquePointer.

typealias LapackInt = __LAPACK_int

/// Layout-compatible with C `float _Complex`.
struct LapackComplex {
    var r: Float
    var i: Float
}

/// Layout-compatible with C `double _Complex`.
struct LapackDoubleComplex {
    var r: Double
    var i: Double
}

typealias LapackIntPtr = UnsafeMutablePointer<LapackInt>
typealias LapackCPtr = UnsafeMutablePointer<LapackComplex>
typealias LapackZPtr = UnsafeMutablePointer<LapackDoubleComplex>

func lapack_cgesv(_ n: LapackIntPtr, _ nrhs: LapackIntPtr, _ a: LapackCPtr, _ lda: LapackIntPtr, _ ipiv: LapackIntPtr, _ b: LapackCPtr, _ ldb: LapackIntPtr, _ info: LapackIntPtr) {
    cgesv_(n, nrhs, OpaquePointer(a), lda, ipiv, OpaquePointer(b), ldb, info)
}

func lapack_zgesv(_ n: LapackIntPtr, _ nrhs: LapackIntPtr, _ a: LapackZPtr, _ lda: LapackIntPtr, _ ipiv: LapackIntPtr, _ b: LapackZPtr, _ ldb: LapackIntPtr, _ info: LapackIntPtr) {
    zgesv_(n, nrhs, OpaquePointer(a), lda, ipiv, OpaquePointer(b), ldb, info)
}

func lapack_cgetrf(_ m: LapackIntPtr, _ n: LapackIntPtr, _ a: LapackCPtr, _ lda: LapackIntPtr, _ ipiv: LapackIntPtr, _ info: LapackIntPtr) {
    cgetrf_(m, n, OpaquePointer(a), lda, ipiv, info)
}

func lapack_cgetrs(_ trans: UnsafeMutablePointer<Int8>, _ n: LapackIntPtr, _ nrhs: LapackIntPtr, _ a: LapackCPtr, _ lda: LapackIntPtr, _ ipiv: LapackIntPtr, _ b: LapackCPtr, _ ldb: LapackIntPtr, _ info: LapackIntPtr) {
    cgetrs_(trans, n, nrhs, OpaquePointer(a), lda, ipiv, OpaquePointer(b), ldb, info)
}

func lapack_zgels(_ trans: UnsafeMutablePointer<Int8>, _ m: LapackIntPtr, _ n: LapackIntPtr, _ nrhs: LapackIntPtr, _ a: LapackZPtr, _ lda: LapackIntPtr, _ b: LapackZPtr, _ ldb: LapackIntPtr, _ work: LapackZPtr, _ lwork: LapackIntPtr, _ info: LapackIntPtr) {
    zgels_(trans, m, n, nrhs, OpaquePointer(a), lda, OpaquePointer(b), ldb, OpaquePointer(work), lwork, info)
}

func lapack_cgecon(_ norm: UnsafeMutablePointer<Int8>, _ n: LapackIntPtr, _ a: LapackCPtr, _ lda: LapackIntPtr, _ anorm: UnsafeMutablePointer<Float>, _ rcond: UnsafeMutablePointer<Float>, _ work: LapackCPtr, _ rwork: UnsafeMutablePointer<Float>, _ info: LapackIntPtr) {
    cgecon_(norm, n, OpaquePointer(a), lda, anorm, rcond, OpaquePointer(work), rwork, info)
}

func lapack_zgecon(_ norm: UnsafeMutablePointer<Int8>, _ n: LapackIntPtr, _ a: LapackZPtr, _ lda: LapackIntPtr, _ anorm: UnsafeMutablePointer<Double>, _ rcond: UnsafeMutablePointer<Double>, _ work: LapackZPtr, _ rwork: UnsafeMutablePointer<Double>, _ info: LapackIntPtr) {
    zgecon_(norm, n, OpaquePointer(a), lda, anorm, rcond, OpaquePointer(work), rwork, info)
}

func lapack_clange(_ norm: UnsafeMutablePointer<Int8>, _ m: LapackIntPtr, _ n: LapackIntPtr, _ a: LapackCPtr, _ lda: LapackIntPtr, _ work: UnsafeMutablePointer<Float>) -> Float {
    clange_(norm, m, n, OpaquePointer(a), lda, work)
}

func lapack_zlange(_ norm: UnsafeMutablePointer<Int8>, _ m: LapackIntPtr, _ n: LapackIntPtr, _ a: LapackZPtr, _ lda: LapackIntPtr, _ work: UnsafeMutablePointer<Double>) -> Double {
    Double(zlange_(norm, m, n, OpaquePointer(a), lda, work))
}

/// Row-major, no-transpose complex64 GEMM: C = alpha * A * B + beta * C
/// (A is m x k, B is k x n, C is m x n).
func lapack_cgemm(
    m: Int, n: Int, k: Int,
    alpha: UnsafeMutablePointer<LapackComplex>,
    a: UnsafeMutablePointer<LapackComplex>, lda: Int,
    b: UnsafeMutablePointer<LapackComplex>, ldb: Int,
    beta: UnsafeMutablePointer<LapackComplex>,
    c: UnsafeMutablePointer<LapackComplex>, ldc: Int
) {
    cblas_cgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        LapackInt(m), LapackInt(n), LapackInt(k),
        OpaquePointer(alpha), OpaquePointer(a), LapackInt(lda),
        OpaquePointer(b), LapackInt(ldb),
        OpaquePointer(beta), OpaquePointer(c), LapackInt(ldc)
    )
}
