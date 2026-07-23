using LinearAlgebra
using Krylov
using LinearOperators

# ---------------------------------------------------------------------------
# Solver dispatcher types — pass GMRES() or CG() to solve() to select algorithm
# ---------------------------------------------------------------------------
struct GMRES end
struct CG end

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
# _apply_flat!: bridges the padded matrix-free operator with Krylov flat vectors
# ---------------------------------------------------------------------------
function _apply_flat!(y::AbstractVector, op,
                      x_vec::AbstractVector,
                      x_pad::AbstractMatrix, y_pad::AbstractMatrix)
    nx, ny = op.nx, op.ny
    vec_to_padded!(x_pad, x_vec, nx, ny)
    apply!(y_pad, op, x_pad)
    padded_to_vec!(y, y_pad, nx, ny)
    y
end

# ---------------------------------------------------------------------------
# solve — generic Krylov solve with MGPreconditioner
# Returns (x_sol, stats, residuals, solve_time)
#   x_sol      : flat solution vector (length nx*ny)
#   stats      : Krylov.jl stats object
#   residuals  : true ‖r‖ per iteration
#   solve_time : wall-clock seconds
# ---------------------------------------------------------------------------
function solve(op, b_vec::AbstractVector{T}, H::MGHierarchy, ::GMRES;
               tol_rel::T = T(1e-10),
               tol_abs::T = T(1e-16),
               max_iters::Int = 300) where T
    nx, ny = op.nx, op.ny
    n = nx * ny
    P = MGPreconditioner(H)
    x_buf = zeros(T, nx+2, ny+2)
    y_buf = zeros(T, nx+2, ny+2)
    Aop = LinearOperator(T, n, n, true, true,
              (y, x) -> _apply_flat!(y, op, x, x_buf, y_buf))
    t0 = time_ns()
    x_sol, stats = Krylov.gmres(Aop, b_vec; N=P,
                                 rtol=tol_rel, atol=tol_abs,
                                 itmax=max_iters, history=true, verbose=0)
    residuals = length(stats.residuals) > 1 ? collect(stats.residuals[2:end]) :
                                               collect(stats.residuals)
    x_sol, stats, residuals, (time_ns() - t0) * 1e-9
end

function solve(op, b_vec::AbstractVector{T}, H::MGHierarchy, ::CG;
               tol_rel::T = T(1e-10),
               tol_abs::T = T(1e-16),
               max_iters::Int = 300) where T
    nx, ny = op.nx, op.ny
    n = nx * ny
    P      = MGPreconditioner(H)
    target = max(tol_abs, tol_rel * norm(b_vec))
    x_buf  = zeros(T, nx+2, ny+2)
    y_buf  = zeros(T, nx+2, ny+2)
    Aop = LinearOperator(T, n, n, true, true,
              (y, x) -> _apply_flat!(y, op, x, x_buf, y_buf))
    t0 = time_ns()
    true_residuals = T[]
    x_sol, stats = Krylov.cg(Aop, b_vec; M=P,
                               rtol=zero(T), atol=zero(T), itmax=max_iters,
                               callback = s -> (push!(true_residuals, norm(s.r));
                                                norm(s.r) ≤ target),
                               verbose=0)
    x_sol, stats, true_residuals, (time_ns() - t0) * 1e-9
end
