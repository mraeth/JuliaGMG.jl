using Adapt
using LinearAlgebra: lu, ldiv!, LU

# Stencil topology: N offsets (di,dj) in padded-index space; center_idx is the
# index of the (0,0) entry.  N is a compile-time constant so the inner loop
# in apply! compiles to fixed-size iteration without dynamic dispatch.
struct StencilSpec{N}
    offsets::NTuple{N, NTuple{2,Int}}
    center_idx::Int
    function StencilSpec(offsets::NTuple{N, NTuple{2,Int}}, center_idx::Int) where N
        @assert offsets[center_idx] == (0, 0) "center_idx must point to the (0,0) offset"
        new{N}(offsets, center_idx)
    end
end

# Canonical 5-point stencil: west, east, south, north, center.
const STENCIL_5PT = StencilSpec(((-1,0),(1,0),(0,-1),(0,1),(0,0)), 5)

# Boundary conditions, one per side, cell-centred ghost rules (see fill_bc!):
#   Dirichlet: u = 0 on the face, ghost = −interior
#   Neumann:   ∂u/∂n = 0 on the face, ghost = +interior
#   Periodic:  ghost = opposite interior row/column (must be paired)
abstract type BoundaryCondition end
struct Dirichlet <: BoundaryCondition end
struct Neumann   <: BoundaryCondition end
struct Periodic  <: BoundaryCondition end

const DIRICHLET_ALL = (Dirichlet(), Dirichlet(), Dirichlet(), Dirichlet())

# Generic N-point finite-difference operator.
#
# coeff is padded (nx+2)×(ny+2), or (nx+2)×(ny+2)×nb for nb independent planes (batch):
# every field array of the operator has the same shape, loops run over k in axes(x, 3).
#
# stencil_fn(i, j, coeff, ihx2, ihy2) -> NTuple{N, T}
#   Returns N stencil weights for interior cell (i,j) in padded-index space,
#   ordered to match stencil.offsets.  ihx2 = 1/hx², ihy2 = 1/hy².
#   coeff is always one 2D plane (the array itself, or a view of plane k for a batch).
#
# coarsen_fn(fine_coeff, cnx, cny) -> coarse_coeff
#   Returns a padded (cnx+2)×(cny+2) coefficient array for the next coarser level.
#   fine_coeff is one 2D plane; each direction is either halved or kept, so the factor
#   is (size(fine_coeff, 1) - 2) ÷ cnx ∈ {1, 2} in x, likewise in y.
#
# bc = (west, east, south, north), default all Dirichlet().  For Periodic sides the
# constructor wraps the ghost ring of coeff in place (face means across the boundary).
struct FDOp{T, A<:AbstractArray{T}, N, StencilFn, CoarsenFn, B<:NTuple{4,BoundaryCondition}}
    coeff::A
    nx::Int
    ny::Int
    hx::T
    hy::T
    stencil::StencilSpec{N}
    stencil_fn::StencilFn
    coarsen_fn::CoarsenFn
    bc::B
    function FDOp(coeff::A, nx, ny, hx::T, hy::T, stencil::StencilSpec{N}, stencil_fn::S,
                  coarsen_fn::C, bc::B = DIRICHLET_ALL) where {T, A<:AbstractArray{T}, N, S, C,
                                                              B<:NTuple{4,BoundaryCondition}}
        (bc[1] isa Periodic) == (bc[2] isa Periodic) ||
            throw(ArgumentError("Periodic must be set on both west and east"))
        (bc[3] isa Periodic) == (bc[4] isa Periodic) ||
            throw(ArgumentError("Periodic must be set on both south and north"))
        _fill_ghosts!(coeff, nx, ny, bc, _ghost_coeff)
        new{T, A, N, S, C, B}(coeff, nx, ny, hx, hy, stencil, stencil_fn, coarsen_fn, bc)
    end
end

Adapt.@adapt_structure FDOp

struct Level{T, A<:AbstractArray{T}, Op}
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
    coarse_lu::F      # pre-factorized LU of the coarse operator, one per plane (CPU)
    singular::Bool    # operator has constants in its null space: work mean-free
end

# Coefficient plane k (the array itself for 2D)
_plane(c::AbstractMatrix, k) = c
_plane(c::AbstractArray, k) = view(c, :, :, k)

# Coarsen the coefficients plane by plane with the user's coarsen_fn
_coarsen(f, c::AbstractMatrix, cnx, cny) = f(c, cnx, cny)
function _coarsen(f, c::AbstractArray, cnx, cny)
    ca = similar(c, cnx+2, cny+2, size(c, 3))
    for k in axes(c, 3)
        ca[:, :, k] .= f(_plane(c, k), cnx, cny)
    end
    ca
end

# ---------------------------------------------------------------------------
# Assemble the dense matrix for a 2D CPU FDOp via unit-vector application.
# Only called once at hierarchy build time on the small coarse grid.
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

# Semi-coarsening: each direction is halved while its size is even and the halved size is
# ≥ min_coarse, otherwise kept (with its spacing); stop when neither can be halved.
function build_hierarchy(op0::FDOp{T,A}; npre=2, npost=2, min_coarse=4,
                         singular::Bool=false) where {T, A}
    function make_level(op)
        x = similar(op.coeff); fill!(x, zero(T))
        b = similar(op.coeff); fill!(b, zero(T))
        r = similar(op.coeff); fill!(r, zero(T))
        Level(x, b, r, op)
    end

    levels = [make_level(op0)]

    cur_op = op0
    while true
        cx = iseven(cur_op.nx) && cur_op.nx ÷ 2 >= min_coarse
        cy = iseven(cur_op.ny) && cur_op.ny ÷ 2 >= min_coarse
        (cx || cy) || break
        cnx = cx ? cur_op.nx ÷ 2 : cur_op.nx
        cny = cy ? cur_op.ny ÷ 2 : cur_op.ny
        chx = cx ? cur_op.hx * 2 : cur_op.hx
        chy = cy ? cur_op.hy * 2 : cur_op.hy
        ca  = _coarsen(cur_op.coarsen_fn, cur_op.coeff, cnx, cny)
        cur_op = FDOp(ca, cnx, cny, chx, chy, cur_op.stencil, cur_op.stencil_fn, cur_op.coarsen_fn,
                      cur_op.bc)
        push!(levels, make_level(cur_op))
    end

    # Pre-factorize the coarse operator of each plane once, on CPU (small dense system).
    coarse_op  = last(levels).op
    nx_c, ny_c = coarse_op.nx, coarse_op.ny
    nx_c * ny_c > 4096 &&
        @warn "JuliaGMG: coarse direct solve has $(nx_c)×$(ny_c) = $(nx_c*ny_c) unknowns per plane " *
              "(sizes not divisible by 2 down to min_coarse=$min_coarse); dense LU will be slow"
    coeff_cpu  = convert(Array{T}, Array(coarse_op.coeff))
    coarse_lu  = map(axes(coeff_cpu, 3)) do k
        cpu_op = FDOp(coeff_cpu[:, :, k], nx_c, ny_c, coarse_op.hx, coarse_op.hy,
                      coarse_op.stencil, coarse_op.stencil_fn, coarse_op.coarsen_fn, coarse_op.bc)
        Ac = _assemble_dense(cpu_op)
        # Singular: A + 11ᵀ/n is non-singular and, for mean-free b, returns the mean-free solution.
        singular && (Ac .+= one(T) / (nx_c * ny_c))
        lu(Ac)
    end

    MGHierarchy{T, eltype(levels), typeof(coarse_lu)}(levels, npre, npost, coarse_lu, singular)
end
