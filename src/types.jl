using Adapt
using LinearAlgebra: lu, ldiv!, LU

# Generic 5-point finite-difference operator.
#
# stencil_fn(i, j, coeff, ihx2, ihy2) -> (diag, wx_neg, wx_pos, wy_neg, wy_pos)
#   Returns the five stencil weights for interior cell (i,j), where (i,j) are
#   padded indices (interior ↔ [2:nx+1, 2:ny+1]).  ihx2 = 1/hx², ihy2 = 1/hy².
#
# coarsen_fn(fine_coeff, cnx, cny) -> coarse_coeff
#   Returns a padded (cnx+2)×(cny+2) coefficient array for the next coarser level.
struct FDOp{T, A<:AbstractMatrix{T}, StencilFn, CoarsenFn}
    coeff::A
    nx::Int
    ny::Int
    hx::T
    hy::T
    stencil_fn::StencilFn
    coarsen_fn::CoarsenFn
end

Adapt.@adapt_structure FDOp

struct Level{T, A<:AbstractMatrix{T}, Op}
    x::A
    b::A
    r::A
    op::Op
end

Adapt.@adapt_structure Level

struct MGHierarchy{T, L, F}
    levels::Vector{L}
    npre::Int
    npost::Int
    coarse_lu::F      # pre-factorized LU of the coarse operator (always CPU Matrix)
end

# ---------------------------------------------------------------------------
# Assemble the dense matrix for a CPU FDOp via unit-vector application.
# Only called once at hierarchy build time on the tiny coarse grid.
# ---------------------------------------------------------------------------
function _assemble_dense(op::FDOp{T, <:Matrix}) where T
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

function build_hierarchy(op0::FDOp{T,A}; npre=2, npost=2, min_coarse=4) where {T, A}
    function make_level(op)
        x = similar(op.coeff); fill!(x, zero(T))
        b = similar(op.coeff); fill!(b, zero(T))
        r = similar(op.coeff); fill!(r, zero(T))
        Level(x, b, r, op)
    end

    levels = [make_level(op0)]

    cur_op = op0
    while min(cur_op.nx, cur_op.ny) ÷ 2 >= min_coarse
        cnx  = cur_op.nx ÷ 2
        cny  = cur_op.ny ÷ 2
        chx  = cur_op.hx * 2
        chy  = cur_op.hy * 2
        ca   = cur_op.coarsen_fn(cur_op.coeff, cnx, cny)
        cur_op = FDOp(ca, cnx, cny, chx, chy, cur_op.stencil_fn, cur_op.coarsen_fn)
        push!(levels, make_level(cur_op))
    end

    # Pre-factorize the coarse operator once.  Always done on CPU: the coarse grid
    # is ≤ min_coarse² DOFs — GPU kernel overhead would dwarf the actual solve time.
    coarse_op  = last(levels).op
    nx_c, ny_c = coarse_op.nx, coarse_op.ny
    coeff_cpu  = convert(Matrix{T}, Array(coarse_op.coeff))
    cpu_op     = FDOp(coeff_cpu, nx_c, ny_c, coarse_op.hx, coarse_op.hy,
                      coarse_op.stencil_fn, coarse_op.coarsen_fn)
    coarse_lu  = lu(_assemble_dense(cpu_op))

    MGHierarchy{T, eltype(levels), typeof(coarse_lu)}(levels, npre, npost, coarse_lu)
end
