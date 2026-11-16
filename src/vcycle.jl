using LinearAlgebra: ldiv!

# ---------------------------------------------------------------------------
# Coarse-level direct solve via the pre-factorized LUs stored in the hierarchy
# (one per plane).  Array()/copyto! cost is negligible on the small coarse grid.
# ---------------------------------------------------------------------------
function coarse_solve!(H::MGHierarchy, lvl_idx::Int)
    lvl    = H.levels[lvl_idx]
    nx, ny = lvl.op.nx, lvl.op.ny
    T      = eltype(lvl.x)

    b_cpu = Array(lvl.b)
    x_cpu = zeros(T, size(b_cpu))
    b_vec = Vector{T}(undef, nx * ny)
    for k in axes(b_cpu, 3)
        @inbounds for j in 1:ny, i in 1:nx
            b_vec[(j-1)*nx + i] = b_cpu[i+1, j+1, k]
        end
        ldiv!(H.coarse_lu[k], b_vec)
        @inbounds for j in 1:ny, i in 1:nx
            x_cpu[i+1, j+1, k] = b_vec[(j-1)*nx + i]
        end
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

    smooth!(lvl, H.npost; reverse=true)   # reverse colour order: symmetric V-cycle
end
