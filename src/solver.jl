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
    H.singular && _remove_mean!(lvl1.b, nx, ny)
    fill!(lvl1.x, zero(eltype(lvl1.x)))
    vcycle!(H)
    padded_to_vec!(y, lvl1.x, nx, ny)
    H.singular && _remove_mean!(y, nx * ny)
    y
end

Base.size(P::MGPreconditioner) = let n = _ndofs(P.hierarchy.levels[1].op); (n,n) end
Base.size(P::MGPreconditioner, d) = size(P)[d]
Base.eltype(::MGPreconditioner{T}) where T = T

# Number of unknowns nx*ny*nb (length of the flat Krylov vectors)
_ndofs(op) = op.nx * op.ny * size(op.coeff, 3)

# ---------------------------------------------------------------------------
# _apply_flat!: bridges the padded matrix-free operator with Krylov flat vectors
# ---------------------------------------------------------------------------
function _apply_flat!(y::AbstractVector, op,
                      x_vec::AbstractVector,
                      x_pad::AbstractArray, y_pad::AbstractArray)
    nx, ny = op.nx, op.ny
    vec_to_padded!(x_pad, x_vec, nx, ny)
    apply!(y_pad, op, x_pad)
    padded_to_vec!(y, y_pad, nx, ny)
    y
end

# ---------------------------------------------------------------------------
# solve — generic Krylov solve with MGPreconditioner
# Returns (x_sol, stats, residuals, solve_time)
#   x_sol      : flat solution vector (length nx*ny*nb, plane-major)
#   stats      : Krylov.jl stats object
#   residuals  : true ‖r‖ per iteration
#   solve_time : wall-clock seconds
# H.singular: the mean of each plane is projected out of b (the consistent part is
# solved) and the mean-free solution is returned.
# ---------------------------------------------------------------------------
function solve(op, b_vec::AbstractVector{T}, H::MGHierarchy, ::GMRES;
               tol_rel::T = T(1e-10),
               tol_abs::T = T(1e-16),
               max_iters::Int = 300) where T
    n = _ndofs(op)
    P = MGPreconditioner(H)
    x_buf = zeros(T, size(op.coeff))
    y_buf = zeros(T, size(op.coeff))
    Aop = LinearOperator(T, n, n, true, true,
              (y, x) -> _apply_flat!(y, op, x, x_buf, y_buf))
    H.singular && (b_vec = _remove_mean!(copy(b_vec), op.nx * op.ny))
    t0 = time_ns()
    x_sol, stats = Krylov.gmres(Aop, b_vec; N=P,
                                 rtol=tol_rel, atol=tol_abs,
                                 itmax=max_iters, history=true, verbose=0)
    H.singular && _remove_mean!(x_sol, op.nx * op.ny)
    residuals = length(stats.residuals) > 1 ? collect(stats.residuals[2:end]) :
                                               collect(stats.residuals)
    x_sol, stats, residuals, (time_ns() - t0) * 1e-9
end

# ---------------------------------------------------------------------------
# solve — matrix-free GMRES with a user right preconditioner
#   A!(y, x) : y = A x on flat vectors (A need not be symmetric)
#   P!(y, x) : y ≈ A⁻¹ x, e.g. a direct solve of a simplified operator, or
#              (y, x) -> mul!(y, MGPreconditioner(H), x) for a GMG cycle
# Returns (x_sol, stats, residuals, solve_time) as above.
# ---------------------------------------------------------------------------
function solve(A!, b_vec::AbstractVector{T}, P!, ::GMRES;
               tol_rel::T = T(1e-10),
               tol_abs::T = T(1e-16),
               max_iters::Int = 300) where T
    n   = length(b_vec)
    Aop = LinearOperator(T, n, n, false, false, A!)
    Pop = LinearOperator(T, n, n, false, false, P!)
    t0  = time_ns()
    x_sol, stats = Krylov.gmres(Aop, b_vec; N=Pop,
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
    n = _ndofs(op)
    P      = MGPreconditioner(H)
    H.singular && (b_vec = _remove_mean!(copy(b_vec), op.nx * op.ny))
    target = max(tol_abs, tol_rel * norm(b_vec))
    x_buf  = zeros(T, size(op.coeff))
    y_buf  = zeros(T, size(op.coeff))
    Aop = LinearOperator(T, n, n, true, true,
              (y, x) -> _apply_flat!(y, op, x, x_buf, y_buf))
    t0 = time_ns()
    true_residuals = T[]
    x_sol, stats = Krylov.cg(Aop, b_vec; M=P,
                               rtol=zero(T), atol=zero(T), itmax=max_iters,
                               callback = s -> (push!(true_residuals, norm(s.r));
                                                norm(s.r) ≤ target),
                               verbose=0)
    H.singular && _remove_mean!(x_sol, op.nx * op.ny)
    x_sol, stats, true_residuals, (time_ns() - t0) * 1e-9
end
