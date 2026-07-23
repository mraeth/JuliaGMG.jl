using LinearAlgebra: ldiv!

# ---------------------------------------------------------------------------
# Coarse-level direct solve via the pre-factorized LU stored in the hierarchy.
# The coarse grid is ≤ min_coarse² DOFs; Array()/copyto! cost is negligible.
# ---------------------------------------------------------------------------
function coarse_solve!(H::MGHierarchy, lvl_idx::Int)
    lvl    = H.levels[lvl_idx]
    nx, ny = lvl.op.nx, lvl.op.ny
    n      = nx * ny
    T      = eltype(lvl.x)

    b_cpu = Array(lvl.b)
    b_vec = Vector{T}(undef, n)
    @inbounds for j in 1:ny, i in 1:nx
        b_vec[(j-1)*nx + i] = b_cpu[i+1, j+1]
    end

    ldiv!(H.coarse_lu, b_vec)

    x_cpu = zeros(T, nx+2, ny+2)
    @inbounds for j in 1:ny, i in 1:nx
        x_cpu[i+1, j+1] = b_vec[(j-1)*nx + i]
    end
    copyto!(lvl.x, x_cpu)
end


# ---------------------------------------------------------------------------
# V-cycle (recursive)
# ---------------------------------------------------------------------------
function vcycle!(H::MGHierarchy, lvl_idx::Int=1)
    lvl = H.levels[lvl_idx]

    smooth!(lvl, H.npre)

    if lvl_idx == length(H.levels)
        coarse_solve!(H, lvl_idx)
    else
        residual!(lvl.r, lvl)

        next = H.levels[lvl_idx+1]
        restrict!(next.b, lvl.r, lvl.op)
        fill!(next.x, zero(eltype(next.x)))

        vcycle!(H, lvl_idx + 1)

        prolong_add!(lvl.x, next.x, next.op)
    end

    smooth!(lvl, H.npost)
end
