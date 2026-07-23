using LinearAlgebra
using Printf

# Build RHS f by applying A to the manufactured solution u_exact = sin(πx)sin(πy)
function make_rhs_and_exact(op::FDOp{T}) where T
    nx, ny = op.nx, op.ny
    u = zeros(T, nx+2, ny+2)
    @inbounds for j in 1:ny, i in 1:nx
        x = (i - T(0.5)) * op.hx
        y = (j - T(0.5)) * op.hy
        u[i+1, j+1] = sin(T(π)*x) * sin(T(π)*y)
    end
    b = zeros(T, nx+2, ny+2)
    apply!(b, op, u)
    b, u
end

# ---------------------------------------------------------------------------
# Helmholtz-specific solver entry point.
# Manufactures the RHS via the known exact solution, delegates the Krylov
# solve to JuliaGMG.solve(), and reports the L2 error vs the exact solution.
#
# solver: pass GMRES() or CG() — defaults to GMRES()
# Output format matches MFES --verbose 2 so parse_mfes() works without changes.
# ---------------------------------------------------------------------------
function run_solver(op::FDOp{T};
                    solver    = GMRES(),
                    tol_rel::T  = T(1e-10),
                    tol_abs::T  = T(1e-16),
                    max_iters::Int = 300,
                    npre::Int   = 2,
                    npost::Int  = 2,
                    verbose::Int = 2) where T

    nx, ny = op.nx, op.ny
    n = nx * ny
    label = solver isa CG ? "CG" : "GMRES"

    b_pad, u_exact_pad = make_rhs_and_exact(op)
    b_vec = zeros(T, n)
    padded_to_vec!(b_vec, b_pad, nx, ny)

    H = build_hierarchy(op; npre=npre, npost=npost)

    r0_norm = norm(b_vec)
    target  = max(tol_abs, tol_rel * r0_norm)

    if verbose >= 2
        @printf("[%s] ||r0|| = %.6e   target = %.6e (rel=%.1e abs=%.1e)\n",
                label, r0_norm, target, tol_rel, tol_abs)
    end

    x_sol, stats, residuals, solve_time =
        solve(op, b_vec, H, solver; tol_rel=tol_rel, tol_abs=tol_abs, max_iters=max_iters)

    n_iters = stats.niter
    final_r = isempty(residuals) ? T(NaN) : last(residuals)

    if verbose >= 2
        for (k, rk) in enumerate(residuals)
            if solver isa CG
                @printf("[CG] iter %3d: ||r|| = %.6e\n", k, rk)
            else
                @printf("[GMRES]  it %4d  ||r|| = %.6e\n", k, rk)
            end
        end
        solved = stats.solved || final_r ≤ target
        if solved
            @printf("[%s] converged in %d iterations, ||r|| = %.6e\n",
                    label, n_iters, final_r)
        else
            @printf("[%s] reached max iters (%d), ||r|| = %.6e\n",
                    label, n_iters, final_r)
        end
        println()
        @printf("%s iterations : %d\n", label, n_iters)
        @printf("solve time       : %.3f s\n", solve_time)

        x_pad = zeros(T, nx+2, ny+2)
        vec_to_padded!(x_pad, x_sol, nx, ny)
        err = zero(T)
        @inbounds for j in 1:ny, i in 1:nx
            err += (x_pad[i+1,j+1] - u_exact_pad[i+1,j+1])^2
        end
        l2_err = sqrt(op.hx * op.hy * err)
        @printf("L2 error vs exact: %.6e (hx = %.4e, hy = %.4e)\n",
                l2_err, op.hx, op.hy)
    end

    x_sol, stats, solve_time, n_iters, residuals
end
