using JuliaGMG
using LinearAlgebra
using Random
using Test

# Manufactured solution u = sin(πx) sin(πy) on the cell-centred grid of [0,1]².
cc(i, h) = (i - 0.5) * h

# Padded (nx+2)×(ny+2) array with g(x, y) at interior cell centres, zero ghost ring.
function padded(g, nx, ny)
    a = zeros(nx+2, ny+2)
    for j in 1:ny, i in 1:nx
        a[i+1, j+1] = g(cc(i, 1/nx), cc(j, 1/ny))
    end
    a
end

u_exact(x, y) = sin(π*x) * sin(π*y)

# Coarsening of the coefficient field: arithmetic (as helmholtz/) or harmonic (as README) 2×2 mean.
coarsen_arith(fa, cnx, cny) = begin
    ca = zeros(eltype(fa), cnx+2, cny+2)
    @views ca[2:cnx+1, 2:cny+1] .= 0.25 .* (fa[2:2:2cnx+1, 2:2:2cny+1] .+ fa[3:2:2cnx+2, 2:2:2cny+1] .+
                                            fa[2:2:2cnx+1, 3:2:2cny+2] .+ fa[3:2:2cnx+2, 3:2:2cny+2])
    ca
end
coarsen_harm(fa, cnx, cny) = begin
    ca = zeros(eltype(fa), cnx+2, cny+2)
    @views ca[2:cnx+1, 2:cny+1] .= 4 ./ (1 ./ fa[2:2:2cnx+1, 2:2:2cny+1] .+ 1 ./ fa[3:2:2cnx+2, 2:2:2cny+1] .+
                                         1 ./ fa[2:2:2cnx+1, 3:2:2cny+2] .+ 1 ./ fa[3:2:2cnx+2, 3:2:2cny+2])
    ca
end

# (α − ∇²) u
helmholtz_stencil(i, j, c, ihx2, ihy2) = (-ihx2, -ihx2, -ihy2, -ihy2, c[i,j] + 2ihx2 + 2ihy2)

# −∇·(α∇u), harmonic face means; on boundary faces (zero ghost coefficient) the cell value is used.
face(a, b) = b > 0 ? 2a*b / (a + b) : a
function diffusion_stencil(i, j, c, ihx2, ihy2)
    a = c[i,j]
    w = -face(a, c[i-1,j]) * ihx2; e = -face(a, c[i+1,j]) * ihx2
    s = -face(a, c[i,j-1]) * ihy2; n = -face(a, c[i,j+1]) * ihy2
    (w, e, s, n, -(w + e + s + n))
end

# Solve on an nx×nx grid; return (discrete L2 error vs u_exact, iterations).
function solve_mms(nx, alpha, f, stencil_fn, coarsen_fn, solver)
    h = 1 / nx
    op = FDOp(padded(alpha, nx, nx), nx, nx, h, h, STENCIL_5PT, stencil_fn, coarsen_fn)
    b = zeros(nx*nx); padded_to_vec!(b, padded(f, nx, nx), nx, nx)
    H = build_hierarchy(op)
    x, stats, _, _ = solve(op, b, H, solver; tol_rel=1e-12)
    u = zeros(nx*nx); padded_to_vec!(u, padded(u_exact, nx, nx), nx, nx)
    h * norm(x - u), stats.niter
end

const NS = (32, 64, 128)

function check_mms(label, alpha, f, stencil_fn, coarsen_fn, solver)
    res   = [solve_mms(n, alpha, f, stencil_fn, coarsen_fn, solver) for n in NS]
    errs  = first.(res); iters = last.(res)
    ratios = errs[1:end-1] ./ errs[2:end]
    println(rpad(label, 34), "error ratios ", round.(ratios; digits=3), "  iterations ", iters)
    @test all(r -> 3.6 < r < 4.4, ratios)                 # 2nd order
    @test maximum(iters) - minimum(iters) <= 3             # h-independent MG preconditioning
end

@testset "JuliaGMG" begin
    @testset "apply! == assembled 5-point Laplacian" begin
        nx, ny, hx, hy = 7, 5, 1/7, 1/5
        op = FDOp(zeros(nx+2, ny+2), nx, ny, hx, hy, STENCIL_5PT,
                  (i, j, _, ihx2, ihy2) -> (-ihx2, -ihx2, -ihy2, -ihy2, 2ihx2 + 2ihy2),
                  (fa, cnx, cny) -> zeros(cnx+2, cny+2))
        # 1D Dirichlet matrix with ghost u_0 = −u_1 (cell-centred): corner entries 3/h²
        lap1d(n, h) = (T = SymTridiagonal(fill(2.0, n), fill(-1.0, n-1)) |> Matrix;
                       T[1,1] = T[n,n] = 3; T / h^2)
        A = kron(I(ny), lap1d(nx, hx)) + kron(lap1d(ny, hy), I(nx))
        v = rand(MersenneTwister(1), nx*ny)
        xp = zeros(nx+2, ny+2); yp = zeros(nx+2, ny+2); y = zeros(nx*ny)
        vec_to_padded!(xp, v, nx, ny)
        apply!(yp, op, xp)
        padded_to_vec!(y, yp, nx, ny)
        @test y ≈ A * v
    end

    for solver in (GMRES(), CG())
        name = "Helmholtz, constant α, $(nameof(typeof(solver)))"
        @testset "$name" begin
            check_mms(name, (x, y) -> 1.0, (x, y) -> (1 + 2π^2) * u_exact(x, y),
                      helmholtz_stencil, coarsen_arith, solver)
        end
        name = "Helmholtz, variable α, $(nameof(typeof(solver)))"
        @testset "$name" begin
            α(x, y) = 1 + 100x*y
            check_mms(name, α, (x, y) -> (α(x, y) + 2π^2) * u_exact(x, y),
                      helmholtz_stencil, coarsen_arith, solver)
        end
    end

    @testset "Diffusion −∇·(α∇u), harmonic means" begin
        α(x, y) = 1 + x + 2y
        # −∇·(α∇u) = −α∇²u − ∂xα ∂xu − ∂yα ∂yu
        f(x, y) = 2π^2 * α(x, y) * u_exact(x, y) -
                  π * cos(π*x) * sin(π*y) - 2π * sin(π*x) * cos(π*y)
        check_mms("Diffusion, harmonic means, CG", α, f, diffusion_stencil, coarsen_harm, CG())
    end
end
