import Foundation
import Metal

// Native Burton–Miller assembly for exterior prescribed-Neumann solves.
//
// The GPU assembles every regular (test, source, image) triangle pair in
// float32 with the six-point rule and skips the pairs listed here. Those
// singular and near pairs are integrated completely in float64 on the CPU and
// added to the float32 system afterwards, so there is no regular-minus-
// correction cancellation however thin a wall is.
//
// Singular pairs (a shared vertex, edge or face) use Duffy rules whose order
// follows the pair's closeness; slivers and close caps use the adaptive path.
// The adaptive path (near pairs and those singular pairs) takes the source
// integral at each test point in polar coordinates about the projection of
// the point, with sinh-transformed Gauss rules along each fan edge and each
// ray, and grades the test face anisotropically towards the source edges and
// the line where the test plane cuts the source. Cost grows with the log of
// size over gap, so thin walls and long slivers stay cheap. The only refused
// geometry is faces that touch, intersect or coincide without sharing a
// vertex.

private typealias V3 = SIMD3<Double>
private typealias V4 = SIMD4<Double>

@inline(__always) private func dot3(_ a: V3, _ b: V3) -> Double {
    a.x*b.x + a.y*b.y + a.z*b.z
}

@inline(__always) private func cross3(_ a: V3, _ b: V3) -> V3 {
    V3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)
}

@inline(__always) private func norm3(_ a: V3) -> Double {
    dot3(a, a).squareRoot()
}

// MARK: - Quadrature settings

private struct BMQuadratureSettings {
    // ln(1/tolerance) for the Gauss order estimate below.
    var logTolerance: Double
    // An outer test piece is cut until, for every feature, its extent
    // across and along is at most this multiple of the distance.
    let splitRatio: Double
    // A test point this many source diameters away uses a direct source rule.
    let directRatio: Double
    let maxDepth: Int
    let minOrder: Int
    let maxOrder: Int

    // Calibrated on the Speaker2 LF mesh against a 1e-6 run (operator change
    // 6e-6) and on the thin-wall and lip references.
    static let standard = BMQuadratureSettings(
        logTolerance: log(1/1.0e-5), splitRatio: 2.0, directRatio: 0.5,
        maxDepth: 200, minOrder: 3, maxOrder: 64
    )
}

// Faces closer than this fraction of their size without sharing a vertex
// touch, intersect or coincide. Float32 vertex coordinates cannot resolve such
// a gap, so the solve is refused instead of integrated.
private let bmDegenerateSeparationRatio = 1.0e-6

// Gauss order for a panel whose nearest integrand singularity lies `ratio`
// panel lengths away (Bernstein-ellipse bound rho^(-2n) <= tolerance), raised
// for the oscillation of exp(ikr) over the panel.
@inline(__always) private func bmOrder(
    _ ratio: Double, phase: Double, _ settings: BMQuadratureSettings
) -> Int {
    var order = settings.minOrder
    if ratio < 1.0e6 {
        let rho = ratio + (1 + ratio*ratio).squareRoot()
        order = max(order, Int((settings.logTolerance/(2*log(rho))).rounded(.up)))
    }
    order = max(order, Int((0.5*phase).rounded(.up)) + 2)
    return min(order, settings.maxOrder)
}

// MARK: - Gauss–Legendre on [0, 1]

private struct BMGauss: Sendable {
    let nodes: [Double]
    let weights: [Double]
}

private func bmGaussRule(_ order: Int) -> BMGauss {
    var nodes: [Double] = [], weights: [Double] = []
    for i in 0..<order {
        var z = cos(Double.pi*(Double(i)+0.75)/(Double(order)+0.5))
        var derivative = 0.0
        for _ in 0..<100 {
            var p0 = 1.0, p1 = z
            if order > 1 {
                for j in 2...order {
                    let next = ((2*Double(j)-1)*z*p1-(Double(j)-1)*p0)/Double(j)
                    p0 = p1; p1 = next
                }
            } else {
                p0 = 1.0; p1 = z
            }
            derivative = Double(order)*(z*p1-p0)/(z*z-1)
            let step = p1/derivative
            z -= step
            if abs(step) < 1e-16 { break }
        }
        nodes.append((z+1)/2)
        weights.append(1/((1-z*z)*derivative*derivative))
    }
    return BMGauss(nodes: nodes, weights: weights)
}

private let bmGaussTable: [BMGauss] = (0...64).map { order in
    order == 0 ? BMGauss(nodes: [], weights: []) : bmGaussRule(order)
}

// MARK: - Faces and distances

fileprivate struct BMFace {
    let v0: V3, v1: V3, v2: V3
    // Outward normal used by the kernels (mirrored for an image).
    let normal: V3
    // Unit normal of the vertex order; signed fan areas are measured with it.
    let plane: V3
    let curl0: V3, curl1: V3, curl2: V3
    let jacobian: Double
    let diameter: Double
    // Barycentric coordinates of an in-plane point p: l1 = (p-v0).dual1,
    // l2 = (p-v0).dual2.
    let dual1: V3, dual2: V3
    // Barycentric coordinates of v0, v1, v2 in the mesh face this piece was
    // cut from (the unit vectors for a whole face). Basis values are always
    // reported in the mesh face's P1 basis.
    let map0: V4, map1: V4, map2: V4

    func point(_ l1: Double, _ l2: Double) -> V3 {
        v0 + l1*(v1-v0) + l2*(v2-v0)
    }

    @inline(__always) func parentBasis(_ l: V4) -> V4 {
        l.x*map0 + l.y*map1 + l.z*map2
    }

    // Longest edge squared over twice the area; 1.15 for an equilateral face.
    var aspect: Double { diameter*diameter/jacobian }
    // Smallest altitude.
    var height: Double { jacobian/diameter }

    func vertex(_ index: Int) -> V3 { index == 0 ? v0 : (index == 1 ? v1 : v2) }
    func map(_ index: Int) -> V4 { index == 0 ? map0 : (index == 1 ? map1 : map2) }
}

// A piece of `parent` with the given corners (in the parent's orientation).
// Normal and basis curls are the parent's: they are constant on a flat face.
fileprivate func bmPiece(_ parent: BMFace, _ a: V3, _ b: V3, _ c: V3,
                         _ ma: V4, _ mb: V4, _ mc: V4) -> BMFace {
    let e0 = b-a, e1 = c-a
    let crossProduct = cross3(e0, e1)
    let jac = norm3(crossProduct)
    let g00 = dot3(e0, e0), g01 = dot3(e0, e1), g11 = dot3(e1, e1)
    let det = g00*g11-g01*g01
    return BMFace(
        v0: a, v1: b, v2: c, normal: parent.normal, plane: crossProduct/jac,
        curl0: parent.curl0, curl1: parent.curl1, curl2: parent.curl2,
        jacobian: jac, diameter: max(norm3(e0), norm3(e1), norm3(c-b)),
        dual1: (g11*e0-g01*e1)/det, dual2: (g00*e1-g01*e0)/det,
        map0: ma, map1: mb, map2: mc
    )
}

fileprivate func bmFace(_ geom: Geometry, _ tri: Int, _ mask: Int) -> BMFace {
    func vertex(_ local: Int) -> V3 {
        let id = geom.triangleVertex(tri, local)
        let p = mirrorPoint((geom.px[id], geom.py[id], geom.pz[id]), mask: mask)
        return V3(Double(p.0), Double(p.1), Double(p.2))
    }
    let v0 = vertex(0), v1 = vertex(1), v2 = vertex(2)
    let mirrored = mirrorNormal(
        (geom.normal(tri, 0), geom.normal(tri, 1), geom.normal(tri, 2)), mask: mask
    )
    let meshNormal = V3(Double(mirrored.0), Double(mirrored.1), Double(mirrored.2))
    let e0 = v1-v0, e1 = v2-v0
    let crossProduct = cross3(e0, e1)
    let jac = norm3(crossProduct)
    // The float64 plane normal, oriented like the mesh normal. The float32
    // mesh normal is off-plane by ~1e-7, which gives coincident and coplanar
    // pairs a spurious log-singular dG/dn term.
    let n = (dot3(crossProduct, meshNormal) >= 0 ? 1.0 : -1.0)*crossProduct/jac
    // curl(phi_i) = n x grad(phi_i), grad(phi_i) = n x opposite_edge / jac.
    // An odd reflection reverses the surface orientation.
    let orientation: Double = mask.nonzeroBitCount.isMultiple(of: 2) ? 1 : -1
    func curl(_ edge: V3) -> V3 {
        orientation*cross3(n, cross3(n, edge)/jac)
    }
    let a = dot3(e0, e0), b = dot3(e0, e1), c = dot3(e1, e1)
    let det = a*c-b*b
    return BMFace(
        v0: v0, v1: v1, v2: v2, normal: n, plane: crossProduct/jac,
        curl0: curl(v2-v1), curl1: curl(v0-v2), curl2: curl(v1-v0),
        jacobian: jac,
        diameter: max(norm3(e0), norm3(e1), norm3(v2-v1)),
        dual1: (c*e0-b*e1)/det, dual2: (a*e1-b*e0)/det,
        map0: V4(1, 0, 0, 0), map1: V4(0, 1, 0, 0), map2: V4(0, 0, 1, 0)
    )
}

private func bmClosestPointOnTriangle(_ p: V3, _ a: V3, _ b: V3, _ c: V3) -> V3 {
    let ab = b-a, ac = c-a, ap = p-a
    let d1 = dot3(ab, ap), d2 = dot3(ac, ap)
    if d1 <= 0 && d2 <= 0 { return a }
    let bp = p-b
    let d3 = dot3(ab, bp), d4 = dot3(ac, bp)
    if d3 >= 0 && d4 <= d3 { return b }
    let vc = d1*d4-d3*d2
    if vc <= 0 && d1 >= 0 && d3 <= 0 { return a+(d1/(d1-d3))*ab }
    let cp = p-c
    let d5 = dot3(ab, cp), d6 = dot3(ac, cp)
    if d6 >= 0 && d5 <= d6 { return c }
    let vb = d5*d2-d1*d6
    if vb <= 0 && d2 >= 0 && d6 <= 0 { return a+(d2/(d2-d6))*ac }
    let va = d3*d6-d5*d4
    if va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0 {
        return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b)
    }
    let denominator = 1/(va+vb+vc)
    return a+ab*(vb*denominator)+ac*(vc*denominator)
}

private func bmPointSegmentDistance(_ p: V3, _ a: V3, _ b: V3) -> Double {
    let ab = b-a
    let length2 = dot3(ab, ab)
    if length2 <= 0 { return norm3(p-a) }
    let t = min(max(dot3(p-a, ab)/length2, 0), 1)
    return norm3(p-(a+t*ab))
}

private func bmSegmentSegmentDistance(_ p1: V3, _ q1: V3, _ p2: V3, _ q2: V3) -> Double {
    let d1 = q1-p1, d2 = q2-p2, r = p1-p2
    let a = dot3(d1, d1), e = dot3(d2, d2), f = dot3(d2, r)
    var s = 0.0, t = 0.0
    if a <= 0 && e <= 0 { return norm3(p1-p2) }
    if a <= 0 {
        t = min(max(f/e, 0), 1)
    } else {
        let c = dot3(d1, r)
        if e <= 0 {
            s = min(max(-c/a, 0), 1)
        } else {
            let b = dot3(d1, d2)
            let denominator = a*e-b*b
            s = denominator > 0 ? min(max((b*f-c*e)/denominator, 0), 1) : 0
            t = (b*s+f)/e
            if t < 0 {
                t = 0; s = min(max(-c/a, 0), 1)
            } else if t > 1 {
                t = 1; s = min(max((b-c)/a, 0), 1)
            }
        }
    }
    return norm3((p1+s*d1)-(p2+t*d2))
}

private func bmSegmentTriangleDistance(_ a: V3, _ b: V3,
                                       _ t0: V3, _ t1: V3, _ t2: V3) -> Double {
    let normal = cross3(t1-t0, t2-t0)
    let ha = dot3(a-t0, normal), hb = dot3(b-t0, normal)
    if (ha < 0 && hb > 0) || (ha > 0 && hb < 0) {
        let c = a+(ha/(ha-hb))*(b-a)
        let e0 = t1-t0, e1 = t2-t0, rel = c-t0
        let d00 = dot3(e0, e0), d01 = dot3(e0, e1), d11 = dot3(e1, e1)
        let d20 = dot3(rel, e0), d21 = dot3(rel, e1)
        let det = d00*d11-d01*d01
        let l1 = (d11*d20-d01*d21)/det, l2 = (d00*d21-d01*d20)/det
        if l1 >= 0 && l2 >= 0 && l1+l2 <= 1 { return 0 }
    }
    var distance = min(norm3(a-bmClosestPointOnTriangle(a, t0, t1, t2)),
                       norm3(b-bmClosestPointOnTriangle(b, t0, t1, t2)))
    distance = min(distance, bmSegmentSegmentDistance(a, b, t0, t1))
    distance = min(distance, bmSegmentSegmentDistance(a, b, t1, t2))
    distance = min(distance, bmSegmentSegmentDistance(a, b, t2, t0))
    return distance
}

private func bmTriangleDistance(_ t: BMFace, _ s: BMFace) -> Double {
    var distance = bmSegmentTriangleDistance(s.v0, s.v1, t.v0, t.v1, t.v2)
    distance = min(distance, bmSegmentTriangleDistance(s.v1, s.v2, t.v0, t.v1, t.v2))
    distance = min(distance, bmSegmentTriangleDistance(s.v2, s.v0, t.v0, t.v1, t.v2))
    distance = min(distance, bmSegmentTriangleDistance(t.v0, t.v1, s.v0, s.v1, s.v2))
    distance = min(distance, bmSegmentTriangleDistance(t.v1, t.v2, s.v0, s.v1, s.v2))
    distance = min(distance, bmSegmentTriangleDistance(t.v2, t.v0, s.v0, s.v1, s.v2))
    return distance
}

// MARK: - Kernels

private struct BMPairConstants {
    let k: Double
    let inverseK: Double
    let kNormalProduct: Double
    let testNormal: V3
    let sourceNormal: V3
    // curl(phi_a) . curl(psi_b), rows a.
    let curl0: V4, curl1: V4, curl2: V4

    init(_ test: BMFace, _ source: BMFace, k: Double) {
        self.k = k
        inverseK = 1/k
        kNormalProduct = k*dot3(test.normal, source.normal)
        testNormal = test.normal
        sourceNormal = source.normal
        func row(_ c: V3) -> V4 {
            V4(dot3(c, source.curl0), dot3(c, source.curl1), dot3(c, source.curl2), 0)
        }
        curl0 = row(test.curl0); curl1 = row(test.curl1); curl2 = row(test.curl2)
    }
}

// Source integral at one test point: sum of c(x,y) psi(y), G and the RHS
// kernel G + (i/k) dG/dn_x. c = dG/dn_y + i k (n_x.n_y) G is the part of
// D - (i/k) H multiplying phi psi; the curl part is added once per pair.
private struct BMInner {
    var cRe = V4(), cIm = V4()
    var gRe = 0.0, gIm = 0.0
    var rhsRe = 0.0, rhsIm = 0.0
    var evaluations = 0
}

@inline(__always) private func bmKernel(
    _ acc: inout BMInner, delta: V3, sourceBasis: V4, weight: Double,
    _ pc: BMPairConstants
) {
    let r2 = dot3(delta, delta)
    if r2 <= 0 { return }
    let r = r2.squareRoot()
    var sine = 0.0, cosine = 0.0
    __sincos(pc.k*r, &sine, &cosine)
    let scale = weight*0.07957747154594767/r
    let gRe = cosine*scale, gIm = sine*scale
    let a = -1/r2, b = pc.k/r
    let dRe = gRe*a-gIm*b, dIm = gRe*b+gIm*a
    let sourceProjection = dot3(delta, pc.sourceNormal)
    let testProjection = -dot3(delta, pc.testNormal)
    acc.cRe += (dRe*sourceProjection-pc.kNormalProduct*gIm)*sourceBasis
    acc.cIm += (dIm*sourceProjection+pc.kNormalProduct*gRe)*sourceBasis
    acc.gRe += gRe
    acc.gIm += gIm
    acc.rhsRe += gRe-dIm*testProjection*pc.inverseK
    acc.rhsIm += gIm+dRe*testProjection*pc.inverseK
    acc.evaluations += 1
}

// A finished 3x3 matrix block (rows a) and 3-entry RHS block, float64.
fileprivate struct BMBlock {
    var mRe0 = V4(), mRe1 = V4(), mRe2 = V4()
    var mIm0 = V4(), mIm1 = V4(), mIm2 = V4()
    var rhsRe = V4(), rhsIm = V4()
    var gRe = 0.0, gIm = 0.0
    var evaluations = 0
    var graded = 0
    var leaves = 0
    var depthLimited = 0

    @inline(__always) mutating func add(_ inner: BMInner, testBasis tb: V4, weight: Double) {
        let t0 = weight*tb.x, t1 = weight*tb.y, t2 = weight*tb.z
        mRe0 += t0*inner.cRe; mRe1 += t1*inner.cRe; mRe2 += t2*inner.cRe
        mIm0 += t0*inner.cIm; mIm1 += t1*inner.cIm; mIm2 += t2*inner.cIm
        rhsRe += (weight*inner.rhsRe)*tb
        rhsIm += (weight*inner.rhsIm)*tb
        gRe += weight*inner.gRe
        gIm += weight*inner.gIm
        evaluations += inner.evaluations
    }

    mutating func add(_ other: BMBlock) {
        mRe0 += other.mRe0; mRe1 += other.mRe1; mRe2 += other.mRe2
        mIm0 += other.mIm0; mIm1 += other.mIm1; mIm2 += other.mIm2
        rhsRe += other.rhsRe; rhsIm += other.rhsIm
        evaluations += other.evaluations
        leaves += other.leaves
        depthLimited += other.depthLimited
    }

    // -(i/k) G curl.curl completes D - (i/k) H.
    mutating func finish(_ pc: BMPairConstants) {
        let re = gIm*pc.inverseK, im = -gRe*pc.inverseK
        mRe0 += re*pc.curl0; mRe1 += re*pc.curl1; mRe2 += re*pc.curl2
        mIm0 += im*pc.curl0; mIm1 += im*pc.curl1; mIm2 += im*pc.curl2
        gRe = 0; gIm = 0
    }
}

// MARK: - Duffy rules

private struct BMDuffyRule: Sendable {
    let tx: [Double], ty: [Double], sx: [Double], sy: [Double], w: [Double]
}

// The same Sauter–Schwab/Duffy maps as the float32 rule in main.swift, in
// float64: kind 1 coincident, 2 shared edge, 3 shared vertex.
private func bmDuffyRule(kind: Int, order: Int) -> BMDuffyRule {
    let rule = bmGaussTable[order]
    var tx: [Double] = [], ty: [Double] = [], sx: [Double] = [], sy: [Double] = []
    var w: [Double] = []
    func append(_ a: Double, _ b: Double, _ c: Double, _ d: Double, _ weight: Double) {
        tx.append(a-b); ty.append(b); sx.append(c-d); sy.append(d); w.append(weight)
    }
    let xs = rule.nodes, ws = rule.weights
    for a in xs.indices {
        for b in xs.indices {
            for c in xs.indices {
                for d in xs.indices {
                    let xi = xs[b], eta1 = xs[a], eta2 = xs[c], eta3 = xs[d]
                    let eta12 = eta1*eta2, eta123 = eta12*eta3
                    let base = ws[a]*ws[b]*ws[c]*ws[d]
                    if kind == 1 {
                        let weight = base*xi*xi*xi*eta1*eta1*eta2
                        append(xi, xi*(1-eta1+eta12), xi*(1-eta123), xi*(1-eta1), weight)
                        append(xi*(1-eta123), xi*(1-eta1), xi, xi*(1-eta1+eta12), weight)
                        append(xi, xi*(eta1-eta12+eta123), xi*(1-eta12), xi*(eta1-eta12), weight)
                        append(xi*(1-eta12), xi*(eta1-eta12), xi, xi*(eta1-eta12+eta123), weight)
                        append(xi*(1-eta123), xi*(eta1-eta123), xi, xi*(eta1-eta12), weight)
                        append(xi, xi*(eta1-eta12), xi*(1-eta123), xi*(eta1-eta123), weight)
                    } else if kind == 2 {
                        let weight = base*xi*xi*xi*eta1*eta1
                        append(xi, xi*eta1*eta3, xi*(1-eta12), xi*eta1*(1-eta2), weight)
                        append(xi, xi*eta1, xi*(1-eta123), xi*eta1*eta2*(1-eta3), weight*eta2)
                        append(xi*(1-eta12), xi*eta1*(1-eta2), xi, xi*eta123, weight*eta2)
                        append(xi*(1-eta123), xi*eta12*(1-eta3), xi, xi*eta1, weight*eta2)
                        append(xi*(1-eta123), xi*eta1*(1-eta2*eta3), xi, xi*eta12, weight*eta2)
                    } else {
                        let weight = base*xi*xi*xi*eta2
                        append(xi, xi*eta1, xi*eta2, xi*eta2*eta3, weight)
                        append(xi*eta2, xi*eta2*eta3, xi, xi*eta1, weight)
                    }
                }
            }
        }
    }
    return BMDuffyRule(tx: tx, ty: ty, sx: sx, sy: sy, w: w)
}

// Duffy orders by pair closeness (see bmDuffyOrder).
private let bmDuffyOrders = [5, 6, 8, 10, 16]
private let bmDuffyRuleTable: [Int: [BMDuffyRule]] = Dictionary(
    uniqueKeysWithValues: bmDuffyOrders.map { order in
        (order, [1, 2, 3].map { bmDuffyRule(kind: $0, order: order) })
    }
)

@inline(__always) private func bmRemapSingular(
    _ x: Double, _ y: Double, kind: Int, local1: Int, local2: Int
) -> (Double, Double) {
    if kind == 1 { return (x, y) }
    func ref(_ local: Int) -> (Double, Double) {
        (local == 1 ? 1 : 0, local == 2 ? 1 : 0)
    }
    if kind == 2 {
        let a = ref(local1), b = ref(local2), c = ref(3-local1-local2)
        return (a.0+x*(b.0-a.0)+y*(c.0-a.0), a.1+x*(b.1-a.1)+y*(c.1-a.1))
    }
    if local1 == 0 { return (x, y) }
    if local1 == 1 { return (1-x-y, y) }
    return (x, 1-x-y)
}

// MARK: - Near pairs

// Direct collapsed Gauss rule over the source for a well separated point.
@inline(__always) private func bmDirectSource(
    _ acc: inout BMInner, x: V3, source s: BMFace, order: Int, _ pc: BMPairConstants
) {
    let rule = bmGaussTable[order]
    let e0 = s.v1-s.v0, e1 = s.v2-s.v0
    for i in 0..<order {
        let u = rule.nodes[i]
        let wu = rule.weights[i]*u*s.jacobian
        for j in 0..<order {
            let v = rule.nodes[j]
            let l1 = u*(1-v), l2 = u*v
            bmKernel(&acc, delta: (s.v0-x)+l1*e0+l2*e1,
                     sourceBasis: s.parentBasis(V4(1-u, l1, l2, 0)),
                     weight: wu*rule.weights[j], pc)
        }
    }
}

// Nodes and weights on [0, 1] for an integrand that is nearly singular at
// `center` (possibly outside the interval), at complex distance `width`:
// the sinh substitution x = center + width*sinh(t) makes it analytic in a
// strip of half-width pi/2 about the real t axis, so one Gauss rule whose
// order grows with log(1/width) replaces geometric grading.
@inline(__always) private func bmNearRule(
    center: Double, width: Double, phase: Double, _ settings: BMQuadratureSettings,
    _ body: (Double, Double) -> Void
) {
    if width >= 0.5 || width <= 0 {
        let dx = center < 0 ? -center : (center > 1 ? center-1 : 0)
        let ratio = width > 0 ? (dx*dx+width*width).squareRoot() : 1.0e9
        let order = bmOrder(ratio, phase: phase, settings)
        let rule = bmGaussTable[order]
        for i in 0..<order { body(rule.nodes[i], rule.weights[i]) }
        return
    }
    let t0 = asinh(-center/width), t1 = asinh((1-center)/width)
    let span = t1-t0
    let order = bmOrder(Double.pi/span, phase: phase, settings)
    let rule = bmGaussTable[order]
    for i in 0..<order {
        let t = t0+span*rule.nodes[i]
        body(center+width*sinh(t), span*rule.weights[i]*width*cosh(t))
    }
}

// Source integral in polar coordinates about the projection p of x onto the
// source plane. The three fan triangles (p, v_i, v_j) carry signed areas, so
// the decomposition is exact for p inside or outside the face. Along each
// edge the ray end point is placed by a sinh rule about the foot of p (width:
// the distance of x from the edge line), and along each ray by a sinh rule
// about p (width: the height of x), so neither a thin gap nor a point close
// to an edge needs extra panels.
private func bmPolarSource(
    _ acc: inout BMInner, x: V3, projection p: V3, height h: Double,
    projectionBasis lp: V4, source s: BMFace, _ pc: BMPairConstants,
    _ settings: BMQuadratureSettings
) {
    let absH = abs(h)
    for edge in 0..<3 {
        let vi = edge == 0 ? s.v0 : (edge == 1 ? s.v1 : s.v2)
        let vj = edge == 0 ? s.v1 : (edge == 1 ? s.v2 : s.v0)
        let bi = s.map(edge), bj = s.map((edge+1)%3)
        let signedArea = dot3(cross3(vi-p, vj-p), s.plane)
        if abs(signedArea) <= 1e-14*s.jacobian { continue }
        let e = vj-vi
        let length2 = dot3(e, e)
        let length = length2.squareRoot()
        let alpha0 = dot3(p-vi, e)/length2
        let lineDistance = norm3(p-(vi+alpha0*e))
        let width = (lineDistance*lineDistance+absH*absH).squareRoot()/length
        bmNearRule(center: alpha0, width: width, phase: pc.k*length, settings) { alpha, wAlpha in
            let q = vi+alpha*e
            let ray = q-p
            let rayLength = norm3(ray)
            if rayLength <= 0 { return }
            let qBasis = (1-alpha)*bi+alpha*bj
            let scale = wAlpha*signedArea
            bmNearRule(center: 0, width: absH/rayLength, phase: pc.k*rayLength,
                       settings) { rho, wRho in
                // y - x without cancellation: the offset of p from x is
                // exactly -h n.
                bmKernel(&acc, delta: rho*ray-h*s.plane,
                         sourceBasis: (1-rho)*lp+rho*qBasis,
                         weight: scale*wRho*rho, pc)
            }
        }
    }
}

private func bmSourceIntegral(
    x: V3, source s: BMFace, _ pc: BMPairConstants, _ settings: BMQuadratureSettings
) -> BMInner {
    var acc = BMInner()
    var h = dot3(x-s.v0, s.plane)
    // Coplanar to rounding: a nonzero height here only adds cancellation.
    if abs(h) <= 1e-12*s.diameter { h = 0 }
    let p = x-h*s.plane
    let rel = p-s.v0
    let l1 = dot3(rel, s.dual1), l2 = dot3(rel, s.dual2), l0 = 1-l1-l2
    var planeDistance = 0.0
    if l0 < 0 || l1 < 0 || l2 < 0 {
        planeDistance = min(bmPointSegmentDistance(p, s.v0, s.v1),
                            bmPointSegmentDistance(p, s.v1, s.v2),
                            bmPointSegmentDistance(p, s.v2, s.v0))
    }
    let distance = (h*h+planeDistance*planeDistance).squareRoot()
    let ratio = distance/s.diameter
    if ratio >= settings.directRatio {
        let order = bmOrder(ratio, phase: pc.k*s.diameter, settings)
        bmDirectSource(&acc, x: x, source: s, order: order, pc)
    } else {
        bmPolarSource(&acc, x: x, projection: p, height: h,
                      projectionBasis: s.parentBasis(V4(l0, l1, l2, 0)),
                      source: s, pc, settings)
    }
    return acc
}

// MARK: - Outer (test face) integration

// Leaves with less than 1/bmLeafShareFactor of the test face area get a
// relative tolerance loosened by that share, never beyond this.
private let bmLeafShareFactor = 100.0
private let bmLeafMaximumTolerance = 1.0e-2

// As a function of the test point x, the source integral is analytic except
// near the source: it varies on the scale of the distance from x to the
// source edges, and to the segment where the test plane cuts the source.
// Those segments are the features the outer quadrature is graded towards.
private struct BMFeature {
    let a: V3, b: V3
}

private struct BMPairGeometry {
    let features: [BMFeature]
    // Distances are clamped to this (1e-4 of the smaller face height).
    // Touching pairs, and walls thinner than it, reach it: the outer
    // integrand stays bounded there, so a strip of this width next to the
    // feature contributes at most its area fraction to the error.
    let floor: Double
}

private func bmPairGeometry(test t: BMFace, source s: BMFace, floor: Double) -> BMPairGeometry {
    var features = [BMFeature(a: s.v0, b: s.v1), BMFeature(a: s.v1, b: s.v2),
                    BMFeature(a: s.v2, b: s.v0)]
    let heights = [dot3(s.v0-t.v0, t.plane), dot3(s.v1-t.v0, t.plane), dot3(s.v2-t.v0, t.plane)]
    let tolerance = 1e-12*max(s.diameter, t.diameter)
    func side(_ h: Double) -> Int { h > tolerance ? 1 : (h < -tolerance ? -1 : 0) }
    var crossing: [V3] = []
    for i in 0..<3 {
        let j = (i+1)%3
        if side(heights[i])*side(heights[j]) < 0 {
            let f = heights[i]/(heights[i]-heights[j])
            crossing.append(s.vertex(i)+f*(s.vertex(j)-s.vertex(i)))
        } else if side(heights[i]) == 0
                    && side(heights[j])*side(heights[(i+2)%3]) < 0 {
            crossing.append(s.vertex(i))
        }
    }
    if crossing.count == 2 && norm3(crossing[1]-crossing[0]) > tolerance {
        features.append(BMFeature(a: crossing[0], b: crossing[1]))
    }
    return BMPairGeometry(features: features, floor: floor)
}

private func bmPolygonSegmentDistance(_ p: [V3], _ a: V3, _ b: V3) -> Double {
    var distance = Double.infinity
    for i in 1..<(p.count-1) {
        distance = min(distance, bmSegmentTriangleDistance(a, b, p[0], p[i], p[i+1]))
    }
    return distance
}

private func bmPolygonPointDistance(_ p: [V3], _ q: V3) -> Double {
    var distance = Double.infinity
    for i in 1..<(p.count-1) {
        distance = min(distance, norm3(q-bmClosestPointOnTriangle(q, p[0], p[i], p[i+1])))
    }
    return distance
}

// Split a convex polygon (corners as test-face barycentric coordinates, with
// a scalar coordinate per corner) at `level` into the parts below and above.
private func bmSplitPolygon(_ polygon: [V4], _ values: [Double], at level: Double)
    -> (below: [V4], above: [V4]) {
    var below: [V4] = [], above: [V4] = []
    below.reserveCapacity(polygon.count+1); above.reserveCapacity(polygon.count+1)
    for i in polygon.indices {
        let j = (i+1)%polygon.count
        let si = values[i], sj = values[j]
        if si <= level { below.append(polygon[i]) }
        if si >= level { above.append(polygon[i]) }
        if (si < level && sj > level) || (si > level && sj < level) {
            let cut = polygon[i]+((level-si)/(sj-si))*(polygon[j]-polygon[i])
            below.append(cut); above.append(cut)
        }
    }
    return (below, above)
}

// Integrate the source integral over one convex test piece with collapsed
// Gauss rules on its fan triangles.
private func bmIntegrateTestPolygon(
    _ block: inout BMBlock, test t: BMFace, source s: BMFace, polygon: [V4],
    order: Int, _ pc: BMPairConstants, _ settings: BMQuadratureSettings
) {
    let rule = bmGaussTable[order]
    let b0 = polygon[0]
    let p0 = t.point(b0.y, b0.z)
    for corner in 1..<(polygon.count-1) {
        let b1 = polygon[corner], b2 = polygon[corner+1]
        let area = norm3(cross3(t.point(b1.y, b1.z)-p0, t.point(b2.y, b2.z)-p0))
        if area <= 0 { continue }
        for i in 0..<order {
            let u = rule.nodes[i]
            let wu = rule.weights[i]*u*area
            for j in 0..<order {
                let v = rule.nodes[j]
                let basis = b0+u*((1-v)*(b1-b0)+v*(b2-b0))
                let inner = bmSourceIntegral(x: t.point(basis.y, basis.z), source: s, pc, settings)
                block.add(inner, testBasis: t.parentBasis(basis), weight: wu*rule.weights[j])
            }
        }
    }
}

// Anisotropic adaptive outer quadrature. For each feature segment a piece
// must be small against the distance to the segment across it, and against
// the distance to where the singularity changes along it (the segment's
// end points, or where it leaves the test plane). A piece violating either is
// cut parallel or perpendicular to the segment, so a strip next to a long
// edge is graded towards the edge and, separately, towards the edge's ends.
// Cost grows with log(size/distance) in each direction, not size/distance,
// so thin walls and slivers stay cheap. Gauss orders follow the final ratio.
private func bmOuter(
    _ block: inout BMBlock, test t: BMFace, source s: BMFace, geometry g: BMPairGeometry,
    polygon: [V4], depth: Int, _ pc: BMPairConstants, _ settings: BMQuadratureSettings
) {
    let points = polygon.map { t.point($0.y, $0.z) }
    var diameter = 0.0
    for i in points.indices {
        for j in (i+1)..<points.count { diameter = max(diameter, norm3(points[j]-points[i])) }
    }
    if diameter <= 0 { return }
    let n = t.plane
    let c = settings.splitRatio
    var worst = 1.0
    var ratio = Double.infinity
    var cutDirection = V3(), cutOrigin = V3(), cutCenter = 0.0, cutScale = 0.0
    for feature in g.features {
        let d = max(bmPolygonSegmentDistance(points, feature.a, feature.b), g.floor)
        let segment = feature.b-feature.a
        let length = norm3(segment)
        let rise = dot3(segment, n)
        let inPlane = segment-rise*n
        let inPlaneLength = norm3(inPlane)
        var tau: V3
        if inPlaneLength > 1e-9*length {
            tau = inPlane/inPlaneLength
        } else {
            // Normal to the test plane: a point singularity; any direction.
            let e = points[1]-points[0]
            tau = (e-dot3(e, n)*n)/norm3(e-dot3(e, n)*n)
        }
        let perp = cross3(n, tau)
        // Along the segment the singularity changes at its end points and,
        // if it rises out of the test plane, over d / sin(rise angle).
        let aDistance = bmPolygonPointDistance(points, feature.a)
        let bDistance = bmPolygonPointDistance(points, feature.b)
        var along = max(min(aDistance, bDistance), g.floor)
        var alongCenter = aDistance <= bDistance ? 0.0 : dot3(segment, tau)
        let sinRise = length > 0 ? abs(rise)/length : 1
        if sinRise*along > d {
            along = d/sinRise
            // Closest approach of the segment's line to the test plane.
            let ha = dot3(feature.a-points[0], n)
            let fraction = rise != 0 ? min(max(-ha/rise, 0), 1) : 0
            alongCenter = fraction*dot3(segment, tau)
        }
        along = max(along, d)
        var uLo = Double.infinity, uHi = -Double.infinity
        var vLo = Double.infinity, vHi = -Double.infinity
        for p in points {
            let u = dot3(p-feature.a, perp), v = dot3(p-feature.a, tau)
            uLo = min(uLo, u); uHi = max(uHi, u); vLo = min(vLo, v); vHi = max(vHi, v)
        }
        let across = uHi-uLo, lengthAlong = vHi-vLo
        ratio = min(ratio, d/max(across, 1e-300), along/max(lengthAlong, 1e-300))
        if across > c*d*worst {
            worst = across/(c*d)
            cutDirection = perp; cutOrigin = feature.a; cutCenter = 0; cutScale = c*d
        }
        if lengthAlong > c*along*worst {
            worst = lengthAlong/(c*along)
            cutDirection = tau; cutOrigin = feature.a; cutCenter = alongCenter; cutScale = c*along
        }
    }
    if worst <= 1 || depth >= settings.maxDepth {
        // A leaf holding a small share of the test face needs a
        // proportionally smaller relative accuracy: its integrand is bounded
        // like the rest, so the absolute error budget is shared by area.
        var leaf = settings
        var area = 0.0
        for i in 1..<(points.count-1) {
            area += norm3(cross3(points[i]-points[0], points[i+1]-points[0]))
        }
        let share = bmLeafShareFactor*area/t.jacobian
        if share < 1 {
            leaf.logTolerance = max(log(1/bmLeafMaximumTolerance),
                                    settings.logTolerance+log(share))
        }
        let order = bmOrder(ratio, phase: pc.k*diameter, leaf)
        bmIntegrateTestPolygon(&block, test: t, source: s, polygon: polygon,
                               order: order, pc, leaf)
        block.leaves += 1
        if worst > 1 { block.depthLimited += 1 }
        return
    }
    let values = points.map { dot3($0-cutOrigin, cutDirection) }
    let lo = values.min()!, hi = values.max()!
    let extent = hi-lo
    var level: Double
    if cutCenter > lo+1e-3*extent && cutCenter < hi-1e-3*extent {
        // Put the singular line or point on a piece boundary first.
        level = cutCenter
    } else if cutCenter <= lo+1e-3*extent {
        level = lo+cutScale
    } else {
        level = hi-cutScale
    }
    // A cut near the far side only shaves a sliver; halve instead.
    if level <= lo+1e-3*extent || level >= hi-1e-3*extent || cutScale >= 0.5*extent {
        level = 0.5*(lo+hi)
    }
    let (below, above) = bmSplitPolygon(polygon, values, at: level)
    for piece in [below, above] where piece.count >= 3 {
        bmOuter(&block, test: t, source: s, geometry: g, polygon: piece,
                depth: depth+1, pc, settings)
    }
}

// One outer piece of the test face for parallel work: a sub-triangle of the
// test face of a near pair, as barycentric corners.
fileprivate struct BMNearItem {
    let pair: Int
    let b0: V4, b1: V4, b2: V4
}

// MARK: - Singular pairs

// A Duffy (Sauter-Schwab) rule removes the singularity at the shared vertex,
// edge or face, but converges slowly when anything else is close: a sliver's
// own narrow height, or a vertex of one face close to the other face away from
// the contact (a thin cap next to its neighbour). The pair's closeness is the
// smaller of 1/aspect of either face and the distance of each non-shared vertex
// from the other face over the larger diameter. Measured per pair against the
// adaptive path on the Speaker2 LF mesh, these orders keep the pair error at or
// below about 1e-4 (typically 1e-6); closer pairs use the adaptive path, which
// grades towards every contact and near feature.
private func bmDuffyOrder(closeness: Double) -> Int? {
    if closeness >= 0.6 { return 5 }
    if closeness >= 0.4 { return 6 }
    if closeness >= 0.3 { return 8 }
    if closeness >= 0.15 { return 10 }
    if closeness >= 0.08 { return 16 }
    return nil
}

private func bmCloseness(_ a: BMFace, _ b: BMFace, shared: [(Int, Int)]) -> Double {
    var closeness = min(1/a.aspect, 1/b.aspect)
    let scale = max(a.diameter, b.diameter)
    for i in 0..<3 where !shared.contains(where: { $0.0 == i }) {
        let p = a.vertex(i)
        closeness = min(closeness, norm3(p-bmClosestPointOnTriangle(p, b.v0, b.v1, b.v2))/scale)
    }
    for j in 0..<3 where !shared.contains(where: { $0.1 == j }) {
        let p = b.vertex(j)
        closeness = min(closeness, norm3(p-bmClosestPointOnTriangle(p, a.v0, a.v1, a.v2))/scale)
    }
    return closeness
}

// The Duffy order for a singular pair, or 0 for the adaptive path (also when
// the vertices shared by index do not coincide exactly, e.g. a seam vertex a
// little off the symmetry plane).
private func bmSingularOrder(_ a: BMFace, _ b: BMFace) -> Int {
    let shared = bmSharedVertices(a, b)
    if shared.isEmpty { return 0 }
    return bmDuffyOrder(closeness: bmCloseness(a, b, shared: shared)) ?? 0
}
// Grading towards a contact stops at this fraction of the smaller face
// height (see BMPairGeometry.floor).
private let bmSingularFloor = 1.0e-4

@inline(__always) private func bmSame(_ a: V3, _ b: V3) -> Bool {
    a.x == b.x && a.y == b.y && a.z == b.z
}

private func bmSharedVertices(_ a: BMFace, _ b: BMFace) -> [(Int, Int)] {
    var shared: [(Int, Int)] = []
    for i in 0..<3 {
        for j in 0..<3 where bmSame(a.vertex(i), b.vertex(j)) {
            shared.append((i, j))
        }
    }
    return shared
}

private func bmDuffyPair(_ block: inout BMBlock, _ a: BMFace, _ b: BMFace,
                         shared: [(Int, Int)], order: Int, _ pc: BMPairConstants) {
    let kind = shared.count >= 3 ? 1 : (shared.count == 2 ? 2 : 3)
    var source = b
    if kind == 1 {
        // The coincident rule needs the same vertex order on both sides.
        let order = (0..<3).map { i in shared.first { $0.0 == i }!.1 }
        source = bmPiece(b, b.vertex(order[0]), b.vertex(order[1]), b.vertex(order[2]),
                         b.map(order[0]), b.map(order[1]), b.map(order[2]))
    }
    let rule = bmDuffyRuleTable[order]![kind-1]
    let testLocal1 = kind == 1 ? 0 : shared[0].0
    let trialLocal1 = kind == 1 ? 0 : shared[0].1
    let testLocal2 = kind == 2 ? shared[1].0 : 0
    let trialLocal2 = kind == 2 ? shared[1].1 : 0
    let jac = a.jacobian*source.jacobian
    for index in rule.w.indices {
        let t = bmRemapSingular(rule.tx[index], rule.ty[index], kind: kind,
                                local1: testLocal1, local2: testLocal2)
        let u = bmRemapSingular(rule.sx[index], rule.sy[index], kind: kind,
                                local1: trialLocal1, local2: trialLocal2)
        var inner = BMInner()
        bmKernel(&inner, delta: source.point(u.0, u.1)-a.point(t.0, t.1),
                 sourceBasis: source.parentBasis(V4(1-u.0-u.1, u.0, u.1, 0)),
                 weight: jac*rule.w[index], pc)
        block.add(inner, testBasis: a.parentBasis(V4(1-t.0-t.1, t.0, t.1, 0)), weight: 1)
    }
}

// Returns true when the pair used the adaptive path.
private func bmSingularPair(
    _ block: inout BMBlock, _ a: BMFace, _ b: BMFace, order: Int,
    _ pc: BMPairConstants, _ settings: BMQuadratureSettings
) -> Bool {
    if order > 0 {
        bmDuffyPair(&block, a, b, shared: bmSharedVertices(a, b), order: order, pc)
        return false
    }
    let geometry = bmPairGeometry(test: a, source: b,
                                  floor: bmSingularFloor*min(a.height, b.height))
    bmOuter(&block, test: a, source: b, geometry: geometry,
            polygon: [V4(1, 0, 0, 0), V4(0, 1, 0, 0), V4(0, 0, 1, 0)], depth: 0, pc, settings)
    return true
}

// MARK: - Frequency-independent cache

fileprivate struct BMNearPair {
    let test: Int
    let trial: Int
    let slot: Int
    // Face separation over the test face diameter.
    let ratio: Double
}

final class BurtonMillerGeometryCache {
    let masks: [Int]
    fileprivate let faces: [[BMFace]]
    let singularPairs: [DuffyPair]
    let singularSlots: [Int]
    let singularOrders: [Int]
    fileprivate let nearPairs: [BMNearPair]
    fileprivate let nearItems: [BMNearItem]
    let excluded: MTLBuffer
    let maskSlots: MTLBuffer
    let buildSeconds: Double
    let nearListSeconds: Double
    let minimumNearRatio: Double

    fileprivate init(masks: [Int], faces: [[BMFace]], singularPairs: [DuffyPair],
                     singularSlots: [Int], singularOrders: [Int], nearPairs: [BMNearPair],
                     nearItems: [BMNearItem], excluded: MTLBuffer,
                     maskSlots: MTLBuffer, buildSeconds: Double,
                     nearListSeconds: Double, minimumNearRatio: Double) {
        self.masks = masks; self.faces = faces
        self.singularPairs = singularPairs; self.singularSlots = singularSlots
        self.singularOrders = singularOrders
        self.nearPairs = nearPairs; self.nearItems = nearItems
        self.excluded = excluded; self.maskSlots = maskSlots
        self.buildSeconds = buildSeconds; self.nearListSeconds = nearListSeconds
        self.minimumNearRatio = minimumNearRatio
    }
}

private final class BMFailure: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var message: String?
    func record(_ text: String) {
        lock.lock()
        if message == nil { message = text }
        lock.unlock()
    }
}

func buildBurtonMillerGeometryCache(
    geom: Geometry, context: ResidentMetalContext
) throws -> BurtonMillerGeometryCache {
    let start = CFAbsoluteTimeGetCurrent()
    let masks = [0]+symmetryImageMasks(geom.symmetryPlane)
    var slotOfMask = [Int](repeating: -1, count: 8)
    for (slot, mask) in masks.enumerated() { slotOfMask[mask] = slot }
    let m = geom.nTriangles
    let faces = masks.map { mask in (0..<m).map { bmFace(geom, $0, mask) } }

    // Reflections are isometries: (a, b) and (0, a^b) give the same block,
    // so only original test faces are integrated and weighted by the image
    // count, which is also the GPU row weight.
    let singularPairs = context.pairList.pairs.filter { $0.testImageMask == 0 }
    if singularPairs.count*masks.count != context.pairList.pairs.count {
        try fail("burton_miller singular image pairs are not closed under reflection")
    }
    let singularSlots = singularPairs.map { slotOfMask[$0.trialImageMask] }
    var singularOrders = [Int](repeating: 0, count: singularPairs.count)
    singularOrders.withUnsafeMutableBufferPointer { output in
        bmParallel(count: singularPairs.count, chunk: 256) { index in
            let pair = singularPairs[index]
            output[index] = bmSingularOrder(faces[0][pair.test],
                                            faces[singularSlots[index]][pair.trial])
        }
    }

    let nearStart = CFAbsoluteTimeGetCurrent()
    let nearList = try buildNearPairList(geom: geom, threshold: 1.5, originalTestOnly: true)
    let nearListSeconds = CFAbsoluteTimeGetCurrent()-nearStart

    let failure = BMFailure()
    var nearPairs = [BMNearPair](
        repeating: BMNearPair(test: 0, trial: 0, slot: 0, ratio: 0),
        count: nearList.pairs.count
    )
    let settings = BMQuadratureSettings.standard
    nearPairs.withUnsafeMutableBufferPointer { output in
        let count = nearList.pairs.count
        let chunk = 512
        DispatchQueue.concurrentPerform(iterations: (count+chunk-1)/chunk) { part in
            for index in (part*chunk)..<min(count, (part+1)*chunk) {
                let pair = nearList.pairs[index]
                let slot = slotOfMask[pair.trialImageMask]
                let t = faces[0][pair.test]
                let s = faces[slot][pair.trial]
                let scale = max(t.diameter, s.diameter)
                let separation = bmTriangleDistance(t, s)
                if separation <= bmDegenerateSeparationRatio*scale {
                    failure.record(
                        "burton_miller refuses faces \(pair.test) and \(pair.trial)"
                            + (pair.trialImageMask == 0 ? "" : " (image \(pair.trialImageMask))")
                            + ": they share no vertex but are \(separation) m apart, below "
                            + "\(bmDegenerateSeparationRatio) of their size; the surface "
                            + "touches, intersects or coincides with itself"
                    )
                }
                output[index] = BMNearPair(
                    test: pair.test, trial: pair.trial, slot: slot,
                    ratio: separation/t.diameter
                )
            }
        }
    }
    if let message = failure.message { try fail(message) }

    // Split close pairs into pieces up front so one thin-wall pair is shared
    // by several threads. This only refines: the adaptive recursion goes on
    // from each piece with the same criterion.
    var nearItems: [BMNearItem] = []
    nearItems.reserveCapacity(nearPairs.count+nearPairs.count/8)
    let e0 = V4(1, 0, 0, 0), e1 = V4(0, 1, 0, 0), e2 = V4(0, 0, 1, 0)
    var minimumRatio = Double.infinity
    for (index, pair) in nearPairs.enumerated() {
        minimumRatio = min(minimumRatio, pair.ratio)
        let depth = pair.ratio < 0.02 ? 2 : (pair.ratio < 0.1 ? 1 : 0)
        var pieces = [(e0, e1, e2)]
        for _ in 0..<depth {
            pieces = pieces.flatMap { (b0, b1, b2) -> [(V4, V4, V4)] in
                let m01 = 0.5*(b0+b1), m12 = 0.5*(b1+b2), m20 = 0.5*(b2+b0)
                return [(b0, m01, m20), (m01, b1, m12), (m20, m12, b2), (m01, m12, m20)]
            }
        }
        for (b0, b1, b2) in pieces {
            nearItems.append(BMNearItem(pair: index, b0: b0, b1: b1, b2: b2))
        }
    }

    let wordCount = max(1, (masks.count*m*m+31)/32)
    var bits = [UInt32](repeating: 0, count: wordCount)
    func exclude(_ slot: Int, _ test: Int, _ trial: Int) {
        let index = (slot*m+test)*m+trial
        bits[index >> 5] |= UInt32(1) << UInt32(index & 31)
    }
    for (pair, slot) in zip(singularPairs, singularSlots) { exclude(slot, pair.test, pair.trial) }
    for pair in nearPairs { exclude(pair.slot, pair.test, pair.trial) }
    let excluded = try makeBuffer(context.device, bits, label: "bm_excluded_pairs")
    let maskSlots = try makeBuffer(context.device, slotOfMask.map { Int32($0) },
                                   label: "bm_mask_slots")
    return BurtonMillerGeometryCache(
        masks: masks, faces: faces, singularPairs: singularPairs,
        singularSlots: singularSlots, singularOrders: singularOrders,
        nearPairs: nearPairs, nearItems: nearItems,
        excluded: excluded, maskSlots: maskSlots,
        buildSeconds: CFAbsoluteTimeGetCurrent()-start,
        nearListSeconds: nearListSeconds,
        minimumNearRatio: nearPairs.isEmpty ? 0 : minimumRatio
    )
}

// MARK: - Assembly


private func bmParallel(count: Int, chunk: Int, _ body: (Int) -> Void) {
    if count == 0 { return }
    DispatchQueue.concurrentPerform(iterations: (count+chunk-1)/chunk) { part in
        for index in (part*chunk)..<min(count, (part+1)*chunk) { body(index) }
    }
}

func assembleBurtonMillerMetal(
    geom: Geometry, neumann: [Complex32], k: Float,
    residentContext: ResidentMetalContext? = nil
) throws -> AssemblyRun {
    if !k.isFinite || k <= 0 { try fail("burton_miller requires positive real k") }
    if geom.apertureTag != nil {
        try fail("burton_miller does not support coupled infinite-baffle solves")
    }
    let start = CFAbsoluteTimeGetCurrent()
    let context: ResidentMetalContext
    if let residentContext {
        context = residentContext
    } else {
        context = try ResidentMetalContext(geom: geom)
    }
    let cache: BurtonMillerGeometryCache
    let cacheReused: Bool
    if let existing = context.burtonMillerCache {
        cache = existing
        cacheReused = true
    } else {
        cache = try buildBurtonMillerGeometryCache(geom: geom, context: context)
        context.burtonMillerCache = cache
        cacheReused = false
    }
    let gpu = try context.beginBurtonMillerFarField(
        neumann: neumann, k: k, excluded: cache.excluded, maskSlots: cache.maskSlots
    )
    let kd = Double(k)
    let settings = BMQuadratureSettings.standard
    let faces = cache.faces

    let singularStart = CFAbsoluteTimeGetCurrent()
    var singularBlocks = [BMBlock](repeating: BMBlock(), count: cache.singularPairs.count)
    singularBlocks.withUnsafeMutableBufferPointer { output in
        bmParallel(count: cache.singularPairs.count, chunk: 16) { index in
            let pair = cache.singularPairs[index]
            let test = faces[0][pair.test]
            let source = faces[cache.singularSlots[index]][pair.trial]
            let pc = BMPairConstants(test, source, k: kd)
            var block = BMBlock()
            if bmSingularPair(&block, test, source, order: cache.singularOrders[index], pc, settings) {
                block.graded = 1
            }
            block.finish(pc)
            output[index] = block
        }
    }
    let singularSeconds = CFAbsoluteTimeGetCurrent()-singularStart

    let nearStart = CFAbsoluteTimeGetCurrent()
    var nearBlocks = [BMBlock](repeating: BMBlock(), count: cache.nearItems.count)
    nearBlocks.withUnsafeMutableBufferPointer { output in
        bmParallel(count: cache.nearItems.count, chunk: 16) { index in
            let item = cache.nearItems[index]
            let pair = cache.nearPairs[item.pair]
            let test = faces[0][pair.test]
            let source = faces[pair.slot][pair.trial]
            let pc = BMPairConstants(test, source, k: kd)
            var block = BMBlock()
            let geometry = bmPairGeometry(test: test, source: source,
                                          floor: bmSingularFloor*min(test.height, source.height))
            bmOuter(&block, test: test, source: source, geometry: geometry,
                    polygon: [item.b0, item.b1, item.b2], depth: 0, pc, settings)
            block.finish(pc)
            output[index] = block
        }
    }
    let nearSeconds = CFAbsoluteTimeGetCurrent()-nearStart

    let far = try context.finishBurtonMillerFarField(
        commandBuffer: gpu.commandBuffer, slot: gpu.slot
    )
    let applyStart = CFAbsoluteTimeGetCurrent()
    let n = geom.p1DofCount
    var aRe = far.arrays.aRe
    var aIm = far.arrays.aIm
    var rhsRe = far.arrays.rhsRe
    var rhsIm = far.arrays.rhsIm
    let weight = Double(cache.masks.count)
    var evaluations = 0
    var gradedPairs = 0
    var leaves = 0
    var depthLimited = 0
    func apply(_ block: BMBlock, test: Int, trial: Int) {
        let q = neumann[trial]
        let qRe = Double(q.re), qIm = Double(q.im)
        for i in 0..<3 {
            let row = geom.p1Dof(test, i)
            let rowRe = i == 0 ? block.mRe0 : (i == 1 ? block.mRe1 : block.mRe2)
            let rowIm = i == 0 ? block.mIm0 : (i == 1 ? block.mIm1 : block.mIm2)
            let bRe = block.rhsRe[i], bIm = block.rhsIm[i]
            rhsRe[row] = Float(Double(rhsRe[row])+weight*(bRe*qRe-bIm*qIm))
            rhsIm[row] = Float(Double(rhsIm[row])+weight*(bRe*qIm+bIm*qRe))
            for j in 0..<3 {
                let index = row*n+geom.p1Dof(trial, j)
                aRe[index] = Float(Double(aRe[index])+weight*rowRe[j])
                aIm[index] = Float(Double(aIm[index])+weight*rowIm[j])
            }
        }
        evaluations += block.evaluations
        gradedPairs += block.graded
        leaves += block.leaves
        depthLimited += block.depthLimited
    }
    for (index, pair) in cache.singularPairs.enumerated() {
        apply(singularBlocks[index], test: pair.test, trial: pair.trial)
    }
    var itemIndex = 0
    for (pairIndex, pair) in cache.nearPairs.enumerated() {
        var block = BMBlock()
        while itemIndex < cache.nearItems.count && cache.nearItems[itemIndex].pair == pairIndex {
            block.add(nearBlocks[itemIndex])
            itemIndex += 1
        }
        apply(block, test: pair.test, trial: pair.trial)
    }
    let applySeconds = CFAbsoluteTimeGetCurrent()-applyStart

    let arrays = AssemblyArrays(aRe: aRe, aIm: aIm, rhsRe: rhsRe, rhsIm: rhsIm)
    return AssemblyRun(
        arrays: arrays, implementation: "swift_native_metal_fused_burton_miller",
        mode: "burton_miller", seconds: CFAbsoluteTimeGetCurrent()-start,
        parity: nil, duffyStats: nil, nearStats: nil,
        metalDispatch: [
            "regular": ["bm_pairs": gpu.dispatch],
            "bm_singular_pairs": cache.singularPairs.count,
            "bm_near_pairs": cache.nearPairs.count,
            "bm_near_items": cache.nearItems.count,
            "bm_image_weight": cache.masks.count,
            "bm_singular_graded_pairs": gradedPairs,
            "bm_near_min_separation_ratio": cache.minimumNearRatio,
            "bm_cpu_kernel_evaluations": evaluations,
            "bm_outer_leaves": leaves,
            "bm_outer_depth_limited": depthLimited,
            "bm_geometry_cache_reused": cacheReused,
            "bm_seconds": [
                "geometry_cache": cacheReused ? 0.0 : cache.buildSeconds,
                "near_pair_list": cacheReused ? 0.0 : cache.nearListSeconds,
                "gpu_far_field": far.gpuSeconds,
                "cpu_singular": singularSeconds,
                "cpu_near": nearSeconds,
                "apply": applySeconds,
            ],
        ]
    )
}
