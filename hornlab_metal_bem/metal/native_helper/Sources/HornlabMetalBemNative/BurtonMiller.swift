import Foundation

// The Burton–Miller path keeps only the fused matrix and one RHS per drive.
// All four kernels in a triangle pair use the same quadrature nodes.
private struct BMFace {
    let vertices: [(Float, Float, Float)]
    let center: (Float, Float, Float)
    let normal: (Float, Float, Float)
    let curls: [(Float, Float, Float)]
    let jacobian: Float
}

private struct BMPairBlock {
    var matrix = Array(repeating: Complex32.zero, count: 9)
    var rhs = Array(repeating: Complex32.zero, count: 3)
}

private func bmDot(_ a: (Float, Float, Float), _ b: (Float, Float, Float)) -> Float {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

private func bmSubtract(_ a: (Float, Float, Float), _ b: (Float, Float, Float)) -> (Float, Float, Float) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

private func bmCross(_ a: (Float, Float, Float), _ b: (Float, Float, Float)) -> (Float, Float, Float) {
    (a.1*b.2 - a.2*b.1, a.2*b.0 - a.0*b.2, a.0*b.1 - a.1*b.0)
}

private func bmPoint(_ face: BMFace, _ xi: Float, _ eta: Float) -> (Float, Float, Float) {
    let b0 = 1 - xi - eta
    return (
        b0*face.vertices[0].0 + xi*face.vertices[1].0 + eta*face.vertices[2].0,
        b0*face.vertices[0].1 + xi*face.vertices[1].1 + eta*face.vertices[2].1,
        b0*face.vertices[0].2 + xi*face.vertices[1].2 + eta*face.vertices[2].2
    )
}

private func bmFace(_ geom: Geometry, _ tri: Int, _ mask: Int) -> BMFace {
    let vertices = (0..<3).map { local -> (Float, Float, Float) in
        let id = geom.triangleVertex(tri, local)
        return mirrorPoint((geom.px[id], geom.py[id], geom.pz[id]), mask: mask)
    }
    let n = mirrorNormal((geom.normal(tri, 0), geom.normal(tri, 1), geom.normal(tri, 2)), mask: mask)
    let jac = 2 * geom.areas[tri]
    let edge0 = bmSubtract(vertices[2], vertices[1])
    let edge1 = bmSubtract(vertices[0], vertices[2])
    let edge2 = bmSubtract(vertices[1], vertices[0])
    // curl(phi) = n × grad(phi), grad(phi_i) = n × opposite_edge / jac.
    let orientation: Float = mask.nonzeroBitCount.isMultiple(of: 2) ? 1 : -1
    let curls = [edge0, edge1, edge2].map { edge -> (Float, Float, Float) in
        let grad = bmCross(n, edge)
        let curl = bmCross(n, (grad.0/jac, grad.1/jac, grad.2/jac))
        return (orientation*curl.0, orientation*curl.1, orientation*curl.2)
    }
    return BMFace(
        vertices: vertices,
        center: ((vertices[0].0 + vertices[1].0 + vertices[2].0)/3,
                 (vertices[0].1 + vertices[1].1 + vertices[2].1)/3,
                 (vertices[0].2 + vertices[1].2 + vertices[2].2)/3),
        normal: n, curls: curls, jacobian: jac
    )
}

private func bmAccumulate(
    _ block: inout BMPairBlock, test: BMFace, source: BMFace,
    tx: Float, ty: Float, sx: Float, sy: Float, weight: Float, k: Float
) {
    let x = bmPoint(test, tx, ty)
    let y = bmPoint(source, sx, sy)
    let delta = bmSubtract(y, x)
    let r2 = bmDot(delta, delta)
    if r2 <= 0 { return }
    let r = sqrt(r2)
    let g = helmholtzG(delta.0, delta.1, delta.2, k) * weight
    let derivative = g * Complex32(re: -1/r2, im: k/r)
    let d = derivative * bmDot(delta, source.normal)
    let kp = derivative * (-bmDot(delta, test.normal))
    let normalProduct = -k*k*bmDot(test.normal, source.normal)
    let eta = Complex32(re: 0, im: 1/k)
    let tb = [1-tx-ty, tx, ty]
    let sb = [1-sx-sy, sx, sy]
    for a in 0..<3 {
        block.rhs[a] = block.rhs[a] + (g + eta*kp) * tb[a]
        for b in 0..<3 {
            let h = g * (bmDot(test.curls[a], source.curls[b])
                + normalProduct*tb[a]*sb[b])
            block.matrix[a*3+b] = block.matrix[a*3+b]
                + d*(tb[a]*sb[b]) - eta*h
        }
    }
}

private func bmRegularPair(_ test: BMFace, _ source: BMFace, k: Float,
                           level: Int) -> BMPairBlock {
    var block = BMPairBlock()
    let (qx, qy, qw) = triangleRule6()
    let subtriangles = level == 0
        ? [ReferenceSubtriangle(a: (0, 0), b: (1, 0), c: (0, 1))]
        : referenceSubtriangles(level: level)
    for testSub in subtriangles {
        for sourceSub in subtriangles {
            let jac = test.jacobian*source.jacobian*testSub.det*sourceSub.det
            for a in qx.indices {
                let ta = pointInSubtriangle(testSub, qx[a], qy[a])
                for b in qx.indices {
                    let sb = pointInSubtriangle(sourceSub, qx[b], qy[b])
                    bmAccumulate(&block, test: test, source: source,
                                 tx: ta.0, ty: ta.1, sx: sb.0, sy: sb.1,
                                 weight: jac*qw[a]*qw[b], k: k)
                }
            }
        }
    }
    return block
}

// A Duffy fan about the projection of each test point removes the 1/r
// singularity. It also handles a reflected triangle touching a symmetry seam.
private func bmFanPair(_ test: BMFace, _ source: BMFace, k: Float,
                       testLevel: Int = 0) -> BMPairBlock {
    var block = BMPairBlock()
    let (nodes, weights) = gaussRule1D4()
    let testSubs = testLevel == 0
        ? [ReferenceSubtriangle(a: (0, 0), b: (1, 0), c: (0, 1))]
        : referenceSubtriangles(level: testLevel)
    let a = source.vertices[0]
    let e0 = bmSubtract(source.vertices[1], a)
    let e1 = bmSubtract(source.vertices[2], a)
    let dot00 = bmDot(e0, e0)
    let dot01 = bmDot(e0, e1)
    let dot11 = bmDot(e1, e1)
    let denominator = dot00*dot11-dot01*dot01
    for sub in testSubs {
        for iu in nodes.indices {
            for iv in nodes.indices {
                let u = nodes[iu]
                let v = nodes[iv]
                let t = pointInSubtriangle(sub, u*(1-v), u*v)
                let x = bmPoint(test, t.0, t.1)
                let away = bmSubtract(x, a)
                let height = bmDot(away, source.normal)
                let projected = (x.0-height*source.normal.0,
                                 x.1-height*source.normal.1,
                                 x.2-height*source.normal.2)
                let relative = bmSubtract(projected, a)
                let dot20 = bmDot(relative, e0)
                let dot21 = bmDot(relative, e1)
                let cx = (dot11*dot20-dot01*dot21)/denominator
                let cy = (dot00*dot21-dot01*dot20)/denominator
                let tw = weights[iu]*weights[iv]*u*test.jacobian*sub.det
                if min(cx, cy, 1-cx-cy) < -1e-5 {
                    // The radial fan only applies when the projected point is
                    // inside the source face, as in the M1 reference.
                    for su in nodes.indices {
                        for sv in nodes.indices {
                            let radial = nodes[su]
                            let along = nodes[sv]
                            let sourceWeight = tw*source.jacobian*weights[su]*weights[sv]*radial
                            bmAccumulate(&block, test: test, source: source,
                                         tx: t.0, ty: t.1,
                                         sx: radial*(1-along), sy: radial*along,
                                         weight: sourceWeight, k: k)
                        }
                    }
                    continue
                }
                let center = [1-cx-cy, cx, cy]
                // K' develops a narrow peak when the source face nearly
                // touches the test point. Grade the Duffy radial coordinate
                // around the normal gap; the same nodes integrate S, D and H.
                var radialBounds: [Float] = [0]
                let ratio = abs(height)/max(sqrt(source.jacobian), Float.leastNormalMagnitude)
                if ratio > 0 && ratio < 0.1 {
                    var edge = min(ratio, 1)
                    radialBounds.append(edge)
                    while edge < 1 && radialBounds.count < 32 {
                        edge = min(1, 2*edge)
                        radialBounds.append(edge)
                    }
                }
                if radialBounds.last! < 1 { radialBounds.append(1) }
                for (i,j) in [(0,1), (1,2), (2,0)] {
                    let f0 = bmSubtract(source.vertices[i], projected)
                    let f1 = bmSubtract(source.vertices[j], projected)
                    let cross = bmCross(f0, f1)
                    let fanJac = sqrt(bmDot(cross, cross))
                    if fanJac < source.jacobian*1e-14 { continue }
                    for panel in 0..<(radialBounds.count-1) {
                        let lower = radialBounds[panel]
                        let width = radialBounds[panel+1]-lower
                        for su in nodes.indices {
                            for sv in nodes.indices {
                                let radial = lower+width*nodes[su]
                                let along = nodes[sv]
                                var basis = center.map { $0*(1-radial) }
                                basis[i] += radial*(1-along)
                                basis[j] += radial*along
                                let sourceWeight = tw*width*weights[su]*weights[sv]*radial*fanJac
                                bmAccumulate(&block, test: test, source: source,
                                             tx: t.0, ty: t.1,
                                             sx: basis[1], sy: basis[2],
                                             weight: sourceWeight, k: k)
                            }
                        }
                    }
                }
            }
        }
    }
    return block
}

private func bmSingularPair(_ test: BMFace, _ source: BMFace,
                            pair: DuffyPair, rule: DuffyRule, k: Float) -> BMPairBlock {
    var block = BMPairBlock()
    let jac = test.jacobian*source.jacobian
    for index in rule.weights.indices {
        let t = remapSingular(rule.testPoints[index], kind: pair.kind,
                              local1: pair.testLocal1, local2: pair.testLocal2)
        let s = remapSingular(rule.trialPoints[index], kind: pair.kind,
                              local1: pair.trialLocal1, local2: pair.trialLocal2)
        let originalT = imageRefToOriginalRef(t.0, t.1, mask: pair.testImageMask)
        let originalS = imageRefToOriginalRef(s.0, s.1, mask: pair.trialImageMask)
        // The reflected triangle is geometrically reversed for an odd image.
        // BMFace is stored in original local order, so the original basis
        // coordinates are used for physical points and curls.
        bmAccumulate(&block, test: test, source: source,
                     tx: originalT.0, ty: originalT.1,
                     sx: originalS.0, sy: originalS.1,
                     weight: jac*rule.weights[index], k: k)
    }
    return block
}

func assembleBurtonMillerMetal(
    geom: Geometry, neumann: [Complex32], k: Float,
    residentContext: ResidentMetalContext? = nil
) throws -> AssemblyRun {
    if !k.isFinite || k <= 0 { try fail("burton_miller requires positive real k") }
    let start = CFAbsoluteTimeGetCurrent()
    let context: ResidentMetalContext
    if let residentContext {
        context = residentContext
    } else {
        context = try ResidentMetalContext(geom: geom)
    }
    let metal = try context.assembleBurtonMillerRegularMetal(neumann: neumann, k: k)
    let n = geom.p1DofCount
    let m = geom.nTriangles
    let masks = [0] + symmetryImageMasks(geom.symmetryPlane)
    let faces = Dictionary(uniqueKeysWithValues: masks.map { mask in
        (mask, (0..<m).map { bmFace(geom, $0, mask) })
    })
    var aRe = metal.arrays.aRe
    var aIm = metal.arrays.aIm
    var rhsRe = metal.arrays.rhsRe
    var rhsIm = metal.arrays.rhsIm
    func applyDelta(_ regular: BMPairBlock, _ accurate: BMPairBlock,
                    test: Int, trial: Int) {
        for i in 0..<3 {
            let row = geom.p1Dof(test, i)
            let rhsDelta = (accurate.rhs[i]-regular.rhs[i])*neumann[trial]
            rhsRe[row] += rhsDelta.re
            rhsIm[row] += rhsDelta.im
            for j in 0..<3 {
                let col = geom.p1Dof(trial, j)
                let delta = accurate.matrix[i*3+j]-regular.matrix[i*3+j]
                aRe[row*n+col] += delta.re
                aIm[row*n+col] += delta.im
            }
        }
    }
    var singularCount = 0
    for pair in context.pairList.pairs {
        guard let rule = context.rules[pair.kind],
              let tests = faces[pair.testImageMask],
              let sources = faces[pair.trialImageMask] else {
            try fail("invalid Burton-Miller singular pair")
        }
        let test = tests[pair.test]
        let source = sources[pair.trial]
        let regular = bmRegularPair(test, source, k: k, level: 0)
        let accurate = bmSingularPair(test, source, pair: pair, rule: rule, k: k)
        applyDelta(regular, accurate, test: pair.test, trial: pair.trial)
        singularCount += 1
    }
    let nearList = try buildNearPairList(geom: geom, threshold: 1.5)
    var refinedCount = 0
    for pair in nearList.pairs {
        guard let tests = faces[pair.testImageMask],
              let sources = faces[pair.trialImageMask] else {
            try fail("invalid Burton-Miller near pair")
        }
        let test = tests[pair.test]
        let source = sources[pair.trial]
        let regular = bmRegularPair(test, source, k: k, level: 0)
        let distance = sqrt(distanceSquared(test.center, source.center))
        let scale = max(sqrt(test.jacobian), sqrt(source.jacobian))
        // The base Metal quadrature has a 1/gap^2 K' peak for almost
        // coincident disjoint faces. Its float32 atomic sum cannot recover
        // that peak by a later delta, even with a graded Duffy fan. Fail
        // explicitly at the narrow-gap limit instead of returning a plausible
        // but wrong solve. This does not affect adjacent singular pairs,
        // which are handled above by their paired Duffy rule.
        if distance < 0.001*scale {
            try fail("burton_miller requires separated disjoint faces (gap below float32 near limit)")
        }
        let level = distance < 0.25*scale ? 1 : 0
        let accurate = bmFanPair(test, source, k: k, testLevel: level)
        applyDelta(regular, accurate, test: pair.test, trial: pair.trial)
        if level > 0 { refinedCount += 1 }
    }
    let arrays = AssemblyArrays(aRe: aRe, aIm: aIm, rhsRe: rhsRe, rhsIm: rhsIm)
    return AssemblyRun(
        arrays: arrays, implementation: "swift_native_metal_fused_burton_miller",
        mode: "burton_miller", seconds: CFAbsoluteTimeGetCurrent()-start,
        parity: nil, duffyStats: nil, nearStats: nil,
        metalDispatch: [
            "regular": metal.dispatch,
            "bm_singular_pairs": singularCount,
            "bm_near_pairs": nearList.pairs.count,
            "bm_refined_pairs": refinedCount,
            "bm_max_near_level": 1,
        ]
    )
}
