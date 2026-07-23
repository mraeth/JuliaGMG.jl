using Adapt
using LinearAlgebra: lu, ldiv!, LU

# Helmholtz operator: (α - ∇²)u = f on (0,1)², Dirichlet BCs
# Storage: (nx+2)×(ny+2) padded arrays; interior = [2:nx+1, 2:ny+1]; ghost ring = row/col 1 and nx+2/ny+2
struct HelmholtzOp{T, A<:AbstractMatrix{T}}
    alpha::A   # padded (nx+2)×(ny+2)
    nx::Int
    ny::Int
    hx::T
    hy::T
end

Adapt.@adapt_structure HelmholtzOp

struct Level{T, A<:AbstractMatrix{T}}
    x::A              # solution / correction
    b::A              # rhs / restricted residual
    r::A              # residual scratch buffer
    op::HelmholtzOp{T,A}
end

Adapt.@adapt_structure Level

struct MGHierarchy{T, L<:Level{T}, F}
    levels::Vector{L}
    npre::Int
    npost::Int
    coarse_lu::F      # pre-factorized LU of the coarse operator (always CPU Matrix)
end

# ---------------------------------------------------------------------------
# Assemble the dense matrix for a CPU HelmholtzOp via unit-vector application.
# Only called once at hierarchy build time on the tiny coarse grid.
# ---------------------------------------------------------------------------
function _assemble_dense(op::HelmholtzOp{T, <:Matrix}) where T
    nx, ny = op.nx, op.ny
    n = nx * ny
    A = zeros(T, n, n)
    tmp_x = zeros(T, nx+2, ny+2)
    tmp_y = zeros(T, nx+2, ny+2)
    for k in 1:n
        j_int = (k-1) ÷ nx + 1
        i_int = (k-1) % nx + 1
        fill!(tmp_x, zero(T))
        tmp_x[i_int+1, j_int+1] = one(T)
        apply!(tmp_y, op, tmp_x)
        for l in 1:n
            jl = (l-1) ÷ nx + 1
            il = (l-1) % nx + 1
            A[l, k] = tmp_y[il+1, jl+1]
        end
    end
    A
end

function build_hierarchy(op0::HelmholtzOp{T,A}; npre=2, npost=2, min_coarse=4) where {T, A}
    levels = Level{T,A}[]

    function make_level(op)
        nx, ny = op.nx, op.ny
        x = similar(op.alpha)
        b = similar(op.alpha)
        r = similar(op.alpha)
        fill!(x, zero(T)); fill!(b, zero(T)); fill!(r, zero(T))
        Level{T,A}(x, b, r, op)
    end

    push!(levels, make_level(op0))

    cur_op = op0
    while min(cur_op.nx, cur_op.ny) ÷ 2 >= min_coarse
        cnx = cur_op.nx ÷ 2
        cny = cur_op.ny ÷ 2
        chx = cur_op.hx * 2
        chy = cur_op.hy * 2
        calpha = similar(cur_op.alpha, cnx+2, cny+2)
        fill!(calpha, zero(T))
        fa = cur_op.alpha
        @views calpha[2:cnx+1, 2:cny+1] .= T(0.25) .* (
            fa[2:2:2*cnx+1, 2:2:2*cny+1] .+
            fa[3:2:2*cnx+2, 2:2:2*cny+1] .+
            fa[2:2:2*cnx+1, 3:2:2*cny+2] .+
            fa[3:2:2*cnx+2, 3:2:2*cny+2]
        )
        cur_op = HelmholtzOp{T,A}(calpha, cnx, cny, chx, chy)
        push!(levels, make_level(cur_op))
    end

    # Pre-factorize the coarse operator once.  Always done on CPU: the coarse grid
    # is ≤ min_coarse² DOFs — GPU kernel overhead would dwarf the actual solve time.
    coarse_op = last(levels).op
    nx_c, ny_c = coarse_op.nx, coarse_op.ny
    alpha_cpu  = Array(coarse_op.alpha)
    cpu_op     = HelmholtzOp{T, Matrix{T}}(alpha_cpu, nx_c, ny_c, coarse_op.hx, coarse_op.hy)
    coarse_lu  = lu(_assemble_dense(cpu_op))

    MGHierarchy{T, eltype(levels), typeof(coarse_lu)}(levels, npre, npost, coarse_lu)
end
