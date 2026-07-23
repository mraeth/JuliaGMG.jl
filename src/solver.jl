using Krylov
using LinearOperators
using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# MGPreconditioner: wraps MGHierarchy; applies one V-cycle via mul!
# ---------------------------------------------------------------------------
struct MGPreconditioner{T, H<:MGHierarchy}
    hierarchy::H
end

MGPreconditioner(H::MGHierarchy{T}) where T = MGPreconditioner{T, typeof(H)}(H)

function LinearAlgebra.mul!(y::AbstractVector, P::MGPreconditioner, x::AbstractVector)
    H    = P.hierarchy
    lvl1 = H.levels[1]
    nx, ny = lvl1.op.nx, lvl1.op.ny
    vec_to_padded!(lvl1.b, x, nx, ny)
    fill!(lvl1.x, zero(eltype(lvl1.x)))
    vcycle!(H)
    padded_to_vec!(y, lvl1.x, nx, ny)
    y
end

Base.size(P::MGPreconditioner) = let n = P.hierarchy.levels[1].op.nx * P.hierarchy.levels[1].op.ny; (n,n) end
Base.size(P::MGPreconditioner, d) = size(P)[d]
Base.eltype(::MGPreconditioner{T}) where T = T


# ---------------------------------------------------------------------------
# Helper: vector ↔ padded array conversion
# ---------------------------------------------------------------------------
function _apply_flat!(y::AbstractVector, op::HelmholtzOp,
                      x_vec::AbstractVector,
                      x_pad::AbstractMatrix, y_pad::AbstractMatrix)
    nx, ny = op.nx, op.ny
    vec_to_padded!(x_pad, x_vec, nx, ny)
    apply!(y_pad, op, x_pad)
    padded_to_vec!(y, y_pad, nx, ny)
    y
end


# ---------------------------------------------------------------------------
# Main solver entry point
#
# GMRES: uses RIGHT preconditioning (Krylov.jl's `N=P`).
#   This minimises ‖r_k‖ directly, so stats.residuals contains true
#   absolute residuals [‖b‖, ‖r_1‖, ‖r_2‖, ...].
#
# CG:   uses LEFT preconditioning (Krylov.jl's `M=P`).
#   stats.residuals contains preconditioned residuals ‖M⁻¹r_k‖.
#   A callback computes true ‖r_k‖ from solver.x at each iteration.
#
# Output format matches MFES --verbose 2 so parse_mfes() works without changes.
# ---------------------------------------------------------------------------
function run_solver(op::HelmholtzOp{T};
                    solver_type::Symbol = :GMRES,
                    tol_rel::T  = T(1e-10),
                    tol_abs::T  = T(1e-16),
                    max_iters::Int = 300,
                    npre::Int   = 2,
                    npost::Int  = 2,
                    verbose::Int = 2) where T

    nx, ny = op.nx, op.ny
    n = nx * ny

    b_pad, u_exact_pad = make_rhs_and_exact(op)
    b_vec = zeros(T, n)
    padded_to_vec!(b_vec, b_pad, nx, ny)

    H  = build_hierarchy(op; npre=npre, npost=npost)
    P  = MGPreconditioner(H)

    x_pad_buf = zeros(T, nx+2, ny+2)
    y_pad_buf = zeros(T, nx+2, ny+2)
    Aop = LinearOperator(T, n, n, true, true,
        (y, x) -> _apply_flat!(y, op, x, x_pad_buf, y_pad_buf))

    r0_norm = norm(b_vec)
    target  = max(tol_abs, tol_rel * r0_norm)
    label   = solver_type == :CG ? "CG" : "GMRES"

    if verbose >= 2
        if solver_type == :CG
            @printf("[CG] ||r0|| = %.6e   target = %.6e (rel=%.1e abs=%.1e)\n",
                    r0_norm, target, tol_rel, tol_abs)
        else
            @printf("[GMRES] ||r0|| = %.6e   target = %.6e (rel=%.1e abs=%.1e)\n",
                    r0_norm, target, tol_rel, tol_abs)
        end
    end

    t0 = time_ns()
    local x_sol, stats, residuals

    if solver_type == :CG
        # CG with left preconditioning.
        # Krylov.jl's internal stopping criterion checks ‖M r_k‖ (preconditioned
        # residual), which with M ≈ A⁻¹ tracks the ERROR norm, not the true 2-norm
        # residual.  For large λ_max(A) this can stop far above the intended target.
        # Instead: disable Krylov.jl's check (rtol=atol=0) and let the callback
        # enforce ‖r_k‖₂ ≤ target — the same criterion as MFES and HYPRE.
        true_residuals = T[]
        function cg_callback(solver)
            rk = norm(solver.r)   # true residual; free — updated by CG recurrence
            push!(true_residuals, rk)
            return rk ≤ target
        end
        x_sol, stats = Krylov.cg(Aop, b_vec; M=P,
                                  rtol=zero(T), atol=zero(T),
                                  itmax=max_iters,
                                  callback=cg_callback,
                                  verbose=0)
        residuals = true_residuals

    else  # GMRES
        # Right preconditioning: minimises true ‖r_k‖; stats.residuals = [‖b‖, ‖r_1‖, ...]
        x_sol, stats = Krylov.gmres(Aop, b_vec; N=P,
                                     rtol=tol_rel, atol=tol_abs,
                                     itmax=max_iters,
                                     history=true,
                                     verbose=0)
        # stats.residuals[1] = ‖b‖ (initial true residual), [2:end] = per-iteration
        residuals = length(stats.residuals) > 1 ? collect(stats.residuals[2:end]) :
                                                   collect(stats.residuals)
    end

    solve_time = (time_ns() - t0) * 1e-9
    n_iters    = stats.niter
    final_r    = isempty(residuals) ? T(NaN) : last(residuals)

    if verbose >= 2
        for (k, rk) in enumerate(residuals)
            if solver_type == :CG
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
