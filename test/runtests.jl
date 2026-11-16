using JuliaGMG
using LinearAlgebra
using Random
using Test

# Manufactured solutions on the cell-centred grid of [0,Lx]×[0,Ly] (default [0,1]²).
cc(i, h) = (i - 0.5) * h

# Padded (nx+2)×(ny+2) array with g(x, y) at interior cell centres, zero ghost ring.
function padded(g, nx, ny, hx=1/nx, hy=1/ny)
    a = zeros(nx+2, ny+2)
    for j in 1:ny, i in 1:nx
        a[i+1, j+1] = g(cc(i, hx), cc(j, hy))
    end
    a
end

u_exact(x, y) = sin(π*x) * sin(π*y)

# Coarsening of the coefficient field: mean of g(α) over the fx×fy fine children (factor 1 or 2
# per direction, inferred from the sizes); arithmetic (as helmholtz/) or harmonic (as README).
function coarsen_mean(fa, cnx, cny, g)
    fx, fy = (size(fa, 1) - 2) ÷ cnx, (size(fa, 2) - 2) ÷ cny
    ca = zeros(eltype(fa), cnx+2, cny+2)
    for J in 1:cny, I in 1:cnx
        s = zero(eltype(fa))
        for dj in 1:fy, di in 1:fx
            s += g(fa[fx*(I-1)+di+1, fy*(J-1)+dj+1])
        end
        ca[I+1, J+1] = g(s / (fx * fy))
    end
    ca
end
coarsen_arith(fa, cnx, cny) = coarsen_mean(fa, cnx, cny, identity)
coarsen_harm(fa, cnx, cny)  = coarsen_mean(fa, cnx, cny, inv)

# (α − ∇²) u
helmholtz_stencil(i, j, c, ihx2, ihy2) = (-ihx2, -ihx2, -ihy2, -ihy2, c[i,j] + 2ihx2 + 2ihy2)

# 1D (α − ∂x²) u: zero y-weights, used with ny = 1
helmholtz1d_stencil(i, j, c, ihx2, ihy2) = (-ihx2, -ihx2, 0.0, 0.0, c[i,j] + 2ihx2)

# −∇·(α∇u), harmonic face means; on boundary faces (zero ghost coefficient) the cell value is used.
face(a, b) = b > 0 ? 2a*b / (a + b) : a
function diffusion_stencil(i, j, c, ihx2, ihy2)
    a = c[i,j]
    w = -face(a, c[i-1,j]) * ihx2; e = -face(a, c[i+1,j]) * ihx2
    s = -face(a, c[i,j-1]) * ihy2; n = -face(a, c[i,j+1]) * ihy2
    (w, e, s, n, -(w + e + s + n))
end

# Solve on an nx×ny grid of [0,Lx]×[0,Ly]; return (discrete L2 error vs u, iterations).
# singular: compare mean-free parts (the solver returns the mean-free solution).
function solve_mms((nx, ny), alpha, f, stencil_fn, coarsen_fn, solver; u=u_exact,
                   bc=ntuple(_ -> Dirichlet(), 4), singular=false, L=(1.0, 1.0))
    hx, hy = L[1] / nx, L[2] / ny
    op = FDOp(padded(alpha, nx, ny, hx, hy), nx, ny, hx, hy, STENCIL_5PT, stencil_fn, coarsen_fn, bc)
    b = zeros(nx*ny); padded_to_vec!(b, padded(f, nx, ny, hx, hy), nx, ny)
    H = build_hierarchy(op; singular)
    x, stats, _, _ = solve(op, b, H, solver; tol_rel=1e-12)
    ue = zeros(nx*ny); padded_to_vec!(ue, padded(u, nx, ny, hx, hy), nx, ny)
    singular && (ue .-= sum(ue) / length(ue); @test abs(sum(x)) < 1e-10 * norm(x))
    sqrt(hx * hy) * norm(x - ue), stats.niter
end

const NS = ((32, 32), (64, 64), (128, 128))

function check_mms(label, alpha, f, stencil_fn, coarsen_fn, solver; sizes=NS, kw...)
    res   = [solve_mms(n, alpha, f, stencil_fn, coarsen_fn, solver; kw...) for n in sizes]
    errs  = first.(res); iters = last.(res)
    ratios = errs[1:end-1] ./ errs[2:end]
    println(rpad(label, 52), "error ratios ", round.(ratios; digits=3), "  iterations ", iters)
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

    P, N = Periodic(), Neumann()
    for solver in (GMRES(), CG())
        sname = nameof(typeof(solver))

        name = "Periodic Helmholtz, variable α, $sname"
        @testset "$name" begin
            α(x, y) = 1 + 100x*y
            u(x, y) = sin(2π*x) * cos(2π*y)
            check_mms(name, α, (x, y) -> (α(x, y) + 8π^2) * u(x, y),
                      helmholtz_stencil, coarsen_arith, solver; u, bc=(P, P, P, P))
        end

        name = "Periodic Poisson (singular), $sname"
        @testset "$name" begin
            u(x, y) = sin(2π*x) * cos(2π*y) + cos(4π*y)
            f(x, y) = 8π^2 * sin(2π*x) * cos(2π*y) + 16π^2 * cos(4π*y)
            check_mms(name, (x, y) -> 0.0, f, helmholtz_stencil, coarsen_arith, solver;
                      u, bc=(P, P, P, P), singular=true)
        end

        name = "Neumann Helmholtz, variable α, $sname"
        @testset "$name" begin
            α(x, y) = 1 + 100x*y
            u(x, y) = cos(π*x) * cos(π*y)
            check_mms(name, α, (x, y) -> (α(x, y) + 2π^2) * u(x, y),
                      helmholtz_stencil, coarsen_arith, solver; u, bc=(N, N, N, N))
        end

        name = "Neumann x / Periodic y diffusion (singular), $sname"
        @testset "$name" begin
            α(x, y) = 1 + x + 0.5sin(2π*y)
            u(x, y) = cos(π*x) * sin(2π*y)
            # −∇·(α∇u) = 5π² α u − ∂xα ∂xu − ∂yα ∂yu
            f(x, y) = 5π^2 * α(x, y) * u(x, y) + π * sin(π*x) * sin(2π*y) -
                      2π^2 * cos(2π*y) * cos(π*x) * cos(2π*y)
            check_mms(name, α, f, diffusion_stencil, coarsen_harm, solver;
                      u, bc=(N, N, P, P), singular=true)
        end
    end

    @testset "Singular: constant in the RHS is projected out" begin
        nx = 32; h = 1 / nx
        op = FDOp(zeros(nx+2, nx+2), nx, nx, h, h, STENCIL_5PT, helmholtz_stencil, coarsen_arith,
                  (P, P, P, P))
        H = build_hierarchy(op; singular=true)
        b = rand(MersenneTwister(2), nx*nx); b .-= sum(b) / length(b)
        for solver in (GMRES(), CG())
            x0 = solve(op, b, H, solver; tol_rel=1e-12)[1]
            x1 = solve(op, b .+ 3.0, H, solver; tol_rel=1e-12)[1]
            @test norm(x1 - x0) < 1e-8 * norm(x0)
        end
    end

    @testset "Unpaired Periodic throws" begin
        mk(bc) = FDOp(zeros(6, 6), 4, 4, 0.25, 0.25, STENCIL_5PT, helmholtz_stencil, coarsen_arith, bc)
        @test_throws ArgumentError mk((Periodic(), Dirichlet(), Dirichlet(), Dirichlet()))
        @test_throws ArgumentError mk((Dirichlet(), Dirichlet(), Neumann(), Periodic()))
        @test mk((Neumann(), Neumann(), Periodic(), Periodic())).bc[3] isa Periodic
    end

    # ---- J2: grid shapes ------------------------------------------------------------------
    S1 = ((64, 1), (128, 1), (256, 1))         # 1D: ny = 1, y never coarsened, Periodic in y
    for solver in (GMRES(), CG())
        sname = nameof(typeof(solver))

        name = "1D periodic Helmholtz, variable α, $sname"
        @testset "$name" begin
            α(x, y) = 1 + 10x
            u(x, y) = sin(2π*x)
            check_mms(name, α, (x, y) -> (α(x, y) + 4π^2) * u(x, y), helmholtz1d_stencil,
                      coarsen_arith, solver; u, bc=(P, P, P, P), sizes=S1)
        end

        for (bx, u, f) in ((N, (x, y) -> cos(π*x), (x, y) -> π^2 * cos(π*x)),
                           (P, (x, y) -> sin(2π*x) + cos(4π*x),
                               (x, y) -> 4π^2 * sin(2π*x) + 16π^2 * cos(4π*x)))
            name = "1D $(nameof(typeof(bx))) Poisson (singular), $sname"
            @testset "$name" begin
                check_mms(name, (x, y) -> 0.0, f, helmholtz1d_stencil, coarsen_arith, solver;
                          u, bc=(bx, bx, P, P), singular=true, sizes=S1)
            end
        end

        name = "Non-power-of-two 48/96/192, variable α, $sname"
        @testset "$name" begin
            α(x, y) = 1 + 100x*y
            check_mms(name, α, (x, y) -> (α(x, y) + 2π^2) * u_exact(x, y),
                      helmholtz_stencil, coarsen_arith, solver; sizes=((48, 48), (96, 96), (192, 192)))
        end

        # Strip [0,1]×[0,1/16] (and transposed), hx = hy: the short direction stops at 4 cells,
        # the long one keeps being halved (semi-coarsening).
        for (L, sizes) in (((1.0, 1/16), ((128, 8), (256, 16), (512, 32))),
                           ((1/16, 1.0), ((8, 128), (16, 256), (32, 512))))
            name = "Strip $(sizes[1][1])×$(sizes[1][2]), semi-coarsening, $sname"
            @testset "$name" begin
                u(x, y) = sin(π*x/L[1]) * sin(π*y/L[2])
                k2 = (π/L[1])^2 + (π/L[2])^2
                check_mms(name, (x, y) -> 1.0, (x, y) -> (1 + k2) * u(x, y),
                          helmholtz_stencil, coarsen_arith, solver; u, L, sizes)
            end
        end

        name = "hx = 2hy, [0,2]×[0,1], $sname"
        @testset "$name" begin
            u(x, y) = sin(π*x/2) * sin(π*y)
            α(x, y) = 1 + 10x*y
            check_mms(name, α, (x, y) -> (α(x, y) + 1.25π^2) * u(x, y),
                      helmholtz_stencil, coarsen_arith, solver; u, L=(2.0, 1.0))
        end
    end

    @testset "Odd size 45×64: x not coarsened" begin
        α(x, y) = 1 + 100x*y
        f(x, y) = (α(x, y) + 2π^2) * u_exact(x, y)
        nx, ny = 45, 64
        op = FDOp(padded(α, nx, ny), nx, ny, 1/nx, 1/ny, STENCIL_5PT, helmholtz_stencil, coarsen_arith)
        H = build_hierarchy(op)
        @test [(l.op.nx, l.op.ny) for l in H.levels] == [(45, 64), (45, 32), (45, 16), (45, 8), (45, 4)]
        for solver in (GMRES(), CG())
            err, it = solve_mms((nx, ny), α, f, helmholtz_stencil, coarsen_arith, solver)
            println(rpad("Odd size 45×64, $(nameof(typeof(solver)))", 52), "error ", round(err; sigdigits=3),
                    "  iterations ", it)
            @test err < 1e-4 && it < 30
        end
    end

    @testset "Coarse system > 4096 unknowns warns" begin
        op = FDOp(zeros(67, 67), 65, 65, 1/65, 1/65, STENCIL_5PT, helmholtz_stencil, coarsen_arith)
        @test_logs (:warn, r"coarse direct solve") build_hierarchy(op)
    end

    # Batch: nb = 3 independent planes with their own coefficients == three separate solves.
    @testset "Batch nb = 3" begin
        nx, ny, nb = 32, 48, 3
        hx, hy = 1/nx, 1/ny
        αs = ((x, y) -> 1.0, (x, y) -> 1 + 100x*y, (x, y) -> 1 + x + 0.5sin(2π*y))
        rng = MersenneTwister(3)
        for (stencil_fn, coarsen_fn, bc, singular, solvers) in
                ((helmholtz_stencil, coarsen_arith, ntuple(_ -> Dirichlet(), 4), false, (GMRES(), CG())),
                 (diffusion_stencil, coarsen_harm, (N, N, P, P), true, (GMRES(), CG())))
            planes = [padded(α, nx, ny) for α in αs]
            coeff  = cat(planes...; dims=3)
            mk(c)  = FDOp(c, nx, ny, hx, hy, STENCIL_5PT, stencil_fn, coarsen_fn, bc)
            opb    = mk(coeff)
            Hb     = build_hierarchy(opb; singular)
            @test length(Hb.coarse_lu) == nb && size(Hb.levels[end].x, 3) == nb
            b = randn(rng, nx*ny*nb)
            n = nx*ny
            for solver in solvers
                xb, sb = solve(opb, b, Hb, solver; tol_rel=1e-12)
                for k in 1:nb
                    opk = mk(copy(planes[k]))
                    bk  = b[(k-1)*n+1:k*n]
                    xk  = solve(opk, bk, build_hierarchy(opk; singular), solver; tol_rel=1e-12)[1]
                    @test norm(xb[(k-1)*n+1:k*n] - xk) < 1e-9 * norm(xk)
                    singular && @test abs(sum(xb[(k-1)*n+1:k*n])) < 1e-10 * norm(xk)
                end
                println(rpad("Batch nb=3, singular=$singular, $(nameof(typeof(solver)))", 52),
                        "iterations ", sb.niter)
            end
        end
    end
end
