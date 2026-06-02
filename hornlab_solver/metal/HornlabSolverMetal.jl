#!/usr/bin/env julia

using JSON
using LinearAlgebra
using Metal

const SCHEMA = "hornlab.metal.standard.v1"
const C32 = ComplexF32

function read_f32(path::String, shape::Vector{Int})
    data = Vector{Float32}(undef, prod(shape))
    open(path, "r") do io
        read!(io, data)
    end
    return data
end

function read_i32(path::String, shape::Vector{Int})
    data = Vector{Int32}(undef, prod(shape))
    open(path, "r") do io
        read!(io, data)
    end
    return data
end

function write_f32(path::String, data)
    mkpath(dirname(path))
    output = ndims(data) == 2 ? permutedims(data) : data
    open(path, "w") do io
        write(io, Float32.(vec(output)))
    end
end

function descriptor_path(root::String, descriptor)
    return joinpath(root, String(descriptor["path"]))
end

function descriptor_shape(descriptor)
    return Int.(descriptor["shape"])
end

function read_geometry(session_manifest_path::String)
    root = dirname(session_manifest_path)
    manifest = JSON.parsefile(session_manifest_path)
    manifest["schema"] == SCHEMA || error("unsupported schema")
    Int(manifest["index_base"]) == 0 || error("expected index_base=0")
    mesh = manifest["mesh"]
    space = manifest["space"]

    vdesc = mesh["vertices_f32"]
    vertices = read_f32(descriptor_path(root, vdesc), descriptor_shape(vdesc))
    nverts = Int(vdesc["shape"][2])
    px = vertices[1:nverts]
    py = vertices[(nverts + 1):(2 * nverts)]
    pz = vertices[(2 * nverts + 1):(3 * nverts)]

    tdesc = mesh["triangles_i32"]
    ntri = Int(tdesc["shape"][2])
    tri_raw = read_i32(descriptor_path(root, tdesc), descriptor_shape(tdesc))
    triangles = reshape(tri_raw, ntri, 3) .+ Int32(1)

    ldesc = mesh["p1_local2global_i32"]
    local_raw = read_i32(descriptor_path(root, ldesc), descriptor_shape(ldesc))
    p1_local2global = reshape(local_raw, 3, ntri)' .+ Int32(1)

    adesc = mesh["triangle_areas_f32"]
    areas = read_f32(descriptor_path(root, adesc), descriptor_shape(adesc))

    ndesc = mesh["triangle_normals_3xm_f32"]
    normal_raw = read_f32(descriptor_path(root, ndesc), descriptor_shape(ndesc))
    normals = reshape(normal_raw, ntri, 3)

    return (
        root=root,
        manifest=manifest,
        px=px,
        py=py,
        pz=pz,
        triangles=triangles,
        p1_local2global=p1_local2global,
        areas=areas,
        normals=normals,
        p1_dof_count=Int(space["p1_dof_count"]),
        dp0_dof_count=Int(space["dp0_dof_count"]),
    )
end

function triangle_rule_6()
    qx = Float32[
        0.4459484909159651, 0.0915762135097710,
        0.1081030181680700, 0.4459484909159651,
        0.8168475729804590, 0.0915762135097710,
    ]
    qy = Float32[
        0.4459484909159651, 0.0915762135097700,
        0.4459484909159651, 0.1081030181680700,
        0.0915762135097700, 0.8168475729804580,
    ]
    qw = Float32[
        0.5 * 0.2233815896780110, 0.5 * 0.1099517436553220,
        0.5 * 0.2233815896780110, 0.5 * 0.2233815896780110,
        0.5 * 0.1099517436553220, 0.5 * 0.1099517436553220,
    ]
    return qx, qy, qw
end

function local_basis(xi::Float32, eta::Float32)
    return (1.0f0 - xi - eta, xi, eta)
end

function point_on_triangle(px, py, pz, triangles, tri::Int, xi::Float32, eta::Float32)
    b1, b2, b3 = local_basis(xi, eta)
    i1 = Int(triangles[tri, 1])
    i2 = Int(triangles[tri, 2])
    i3 = Int(triangles[tri, 3])
    return (
        b1 * px[i1] + b2 * px[i2] + b3 * px[i3],
        b1 * py[i1] + b2 * py[i2] + b3 * py[i3],
        b1 * pz[i1] + b2 * pz[i2] + b3 * pz[i3],
    )
end

function helmholtz_g(dx::Float32, dy::Float32, dz::Float32, k::Float32)
    r2 = dx * dx + dy * dy + dz * dz
    r2 <= 0.0f0 && return C32(0)
    r = sqrt(r2)
    phase = k * r
    scale = Float32(0.07957747154594767) / r
    return C32(cos(phase) * scale, sin(phase) * scale)
end

function helmholtz_dlp(
    dx::Float32,
    dy::Float32,
    dz::Float32,
    nx::Float32,
    ny::Float32,
    nz::Float32,
    k::Float32,
)
    r2 = dx * dx + dy * dy + dz * dz
    r2 <= 0.0f0 && return C32(0)
    r = sqrt(r2)
    phase = k * r
    scale = Float32(0.07957747154594767) / r
    gre = cos(phase) * scale
    gim = sin(phase) * scale
    projection = (dx * nx + dy * ny + dz * nz) / r
    fre = -1.0f0 / r
    fim = k
    return C32(
        (gre * fre - gim * fim) * projection,
        (gre * fim + gim * fre) * projection,
    )
end

function assemble_regular!(A, rhs, geom, neumann, k::Float32)
    qx, qy, qw = triangle_rule_6()
    ntri = size(geom.triangles, 1)
    for trial in 1:ntri
        nnx = geom.normals[trial, 1]
        nny = geom.normals[trial, 2]
        nnz = geom.normals[trial, 3]
        trial_area = geom.areas[trial]
        g_trial = neumann[trial]
        for test in 1:ntri
            jac = (2.0f0 * geom.areas[test]) * (2.0f0 * trial_area)
            block = zeros(C32, 3, 3)
            slp = zeros(C32, 3)
            for a in eachindex(qw)
                tx, ty, tz = point_on_triangle(
                    geom.px, geom.py, geom.pz, geom.triangles, test, qx[a], qy[a],
                )
                tb = local_basis(qx[a], qy[a])
                for b in eachindex(qw)
                    sx, sy, sz = point_on_triangle(
                        geom.px, geom.py, geom.pz, geom.triangles, trial, qx[b], qy[b],
                    )
                    sb = local_basis(qx[b], qy[b])
                    dx = sx - tx
                    dy = sy - ty
                    dz = sz - tz
                    g = helmholtz_g(dx, dy, dz, k)
                    d = helmholtz_dlp(dx, dy, dz, nnx, nny, nnz, k)
                    w = qw[a] * qw[b] * jac
                    for i in 1:3
                        slp[i] += tb[i] * g * w
                        for j in 1:3
                            block[i, j] += tb[i] * sb[j] * d * w
                        end
                    end
                end
            end
            tdofs = geom.p1_local2global[test, :]
            sdofs = geom.p1_local2global[trial, :]
            for i in 1:3
                rhs[Int(tdofs[i])] += slp[i] * g_trial
                for j in 1:3
                    A[Int(tdofs[i]), Int(sdofs[j])] += block[i, j]
                end
            end
            if test == trial
                for i in 1:3, j in 1:3
                    mass = geom.areas[test] * (i == j ? Float32(1 / 6) : Float32(1 / 12))
                    A[Int(tdofs[i]), Int(sdofs[j])] -= 0.5f0 * mass
                end
            end
        end
    end
end

function read_complex_vector(root::String, descriptors, n::Int)
    real_desc = descriptors["real_f32"]
    imag_desc = descriptors["imag_f32"]
    re = read_f32(descriptor_path(root, real_desc), descriptor_shape(real_desc))
    im = read_f32(descriptor_path(root, imag_desc), descriptor_shape(imag_desc))
    length(re) == n || error("real vector length mismatch")
    length(im) == n || error("imag vector length mismatch")
    return C32.(re, im)
end

function write_result_manifest(path::String, data)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, data, 2)
        write(io, "\n")
    end
end

function assemble_standard_neumann(session_manifest_path::String, payload_path::String, result_path::String)
    geom = read_geometry(session_manifest_path)
    payload = JSON.parsefile(payload_path)
    payload["schema"] == SCHEMA || error("unsupported schema")
    k = Float32(payload["k_real_f32"])
    neumann = read_complex_vector(geom.root, payload["neumann_dp0"], geom.dp0_dof_count)
    A = zeros(C32, geom.p1_dof_count, geom.p1_dof_count)
    rhs = zeros(C32, geom.p1_dof_count)
    elapsed = @elapsed assemble_regular!(A, rhs, geom, neumann, k)
    outputs = payload["outputs"]
    write_f32(descriptor_path(geom.root, outputs["A_real_f32"]), real.(A))
    write_f32(descriptor_path(geom.root, outputs["A_imag_f32"]), imag.(A))
    write_f32(descriptor_path(geom.root, outputs["rhs_real_f32"]), real.(rhs))
    write_f32(descriptor_path(geom.root, outputs["rhs_imag_f32"]), imag.(rhs))
    write_result_manifest(result_path, Dict(
        "schema" => SCHEMA,
        "op" => "assemble_standard_neumann_result",
        "session_id" => payload["session_id"],
        "frequency_hz" => payload["frequency_hz"],
        "matrix_layout" => "row_major_c",
        "matrix_shape" => [geom.p1_dof_count, geom.p1_dof_count],
        "rhs_shape" => [geom.p1_dof_count],
        "matrix_real_f32" => outputs["A_real_f32"]["path"],
        "matrix_imag_f32" => outputs["A_imag_f32"]["path"],
        "rhs_real_f32" => outputs["rhs_real_f32"]["path"],
        "rhs_imag_f32" => outputs["rhs_imag_f32"]["path"],
        "timing_s" => elapsed,
        "implementation" => "julia_reference_regular_quadrature",
    ))
end

function evaluate_standard_exterior(session_manifest_path::String, payload_path::String, result_path::String)
    geom = read_geometry(session_manifest_path)
    payload = JSON.parsefile(payload_path)
    payload["schema"] == SCHEMA || error("unsupported schema")
    k = Float32(payload["k_real_f32"])
    pressure = read_complex_vector(geom.root, payload["pressure_p1"], geom.p1_dof_count)
    neumann = read_complex_vector(geom.root, payload["neumann_dp0"], geom.dp0_dof_count)
    obs_desc = payload["observation_points"]
    obs_raw = read_f32(descriptor_path(geom.root, obs_desc), descriptor_shape(obs_desc))
    nobs = Int(obs_desc["shape"][2])
    obs = reshape(obs_raw, nobs, 3)
    qx, qy, qw = triangle_rule_6()
    out = zeros(C32, nobs)
    elapsed = @elapsed begin
        for oi in 1:nobs
            ox = obs[oi, 1]
            oy = obs[oi, 2]
            oz = obs[oi, 3]
            acc = C32(0)
            for tri in 1:size(geom.triangles, 1)
                normal = (
                    geom.normals[tri, 1],
                    geom.normals[tri, 2],
                    geom.normals[tri, 3],
                )
                jac = 2.0f0 * geom.areas[tri]
                dofs = geom.p1_local2global[tri, :]
                for a in eachindex(qw)
                    sx, sy, sz = point_on_triangle(
                        geom.px, geom.py, geom.pz, geom.triangles, tri, qx[a], qy[a],
                    )
                    b = local_basis(qx[a], qy[a])
                    surface_p = b[1] * pressure[Int(dofs[1])] +
                                b[2] * pressure[Int(dofs[2])] +
                                b[3] * pressure[Int(dofs[3])]
                    dx = sx - ox
                    dy = sy - oy
                    dz = sz - oz
                    d = helmholtz_dlp(dx, dy, dz, normal[1], normal[2], normal[3], k)
                    g = helmholtz_g(dx, dy, dz, k)
                    acc += (d * surface_p - g * neumann[tri]) * qw[a] * jac
                end
            end
            out[oi] = acc
        end
    end
    output = payload["output"]
    write_f32(descriptor_path(geom.root, output["pressure_real_f32"]), real.(out))
    write_f32(descriptor_path(geom.root, output["pressure_imag_f32"]), imag.(out))
    write_result_manifest(result_path, Dict(
        "schema" => SCHEMA,
        "op" => "evaluate_standard_exterior_result",
        "session_id" => payload["session_id"],
        "batch_id" => payload["batch_id"],
        "frequency_hz" => payload["frequency_hz"],
        "shape" => [nobs],
        "pressure_real_f32" => output["pressure_real_f32"]["path"],
        "pressure_imag_f32" => output["pressure_imag_f32"]["path"],
        "timing_s" => elapsed,
        "implementation" => "julia_reference_regular_quadrature",
    ))
end

function smoke()
    data = mtl(Float32[1, 2, 3, 4])
    out = Metal.zeros(Float32, 4)
    Metal.@sync @metal threads=4 groups=1 smoke_kernel!(out, data, Int32(4))
    Metal.synchronize()
    Array(out) == Float32[2, 3, 4, 5] || error("Metal smoke kernel failed")
    println("hornlab-solver Metal backend smoke ok")
end

function smoke_kernel!(out, data, n)
    idx = thread_position_in_grid().x
    if idx <= n
        out[idx] = data[idx] + 1.0f0
    end
    return
end

function main(args)
    if length(args) == 1 && args[1] == "--smoke"
        smoke()
        return
    end
    length(args) == 4 || error("usage: HornlabSolverMetal.jl <op> <session.json> <payload.json> <result.json>")
    op, session_manifest_path, payload_path, result_path = args
    if op == "assemble_standard_neumann"
        assemble_standard_neumann(session_manifest_path, payload_path, result_path)
    elseif op == "evaluate_standard_exterior"
        evaluate_standard_exterior(session_manifest_path, payload_path, result_path)
    else
        error("unsupported operation: $op")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
